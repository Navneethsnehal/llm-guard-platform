"""Request orchestration: limits, guardrails, cache, routing, and accounting."""

import asyncio
import hashlib
import logging
import time
import uuid
from collections.abc import AsyncIterator, Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field

from opentelemetry import trace
from opentelemetry.trace import Span, Tracer

from llm_guard.auth import Tenant
from llm_guard.budget import (
    BudgetLedger,
    Reservation,
    micro_to_usd,
    reservation_micro,
    usage_cost_micro,
    usd_to_micro,
)
from llm_guard.cache import SemanticCache
from llm_guard.config import Settings
from llm_guard.errors import GatewayError, GuardrailBlocked, PayloadTooLargeError
from llm_guard.guardrails.grounding import grounding_report
from llm_guard.guardrails.pii import PiiVault
from llm_guard.guardrails.pipeline import GuardContext, GuardrailPipeline
from llm_guard.observability import (
    AuditLog,
    GatewayMetrics,
    audit_event,
    request_id_var,
    traces_to_audit,
)
from llm_guard.ratelimit import RateLimiter
from llm_guard.routing import ModelRouter, ModelSpec, StreamStart
from llm_guard.sse import completion_chunk, iter_buffered, sse, sse_done
from llm_guard.types import (
    CompletionRequest,
    CompletionResult,
    GuardTrace,
    Message,
    ProviderEvent,
    ProviderRequest,
    Usage,
    estimate_tokens,
    render_messages,
    usage_for,
)

logger = logging.getLogger("llm_guard.gateway")


@dataclass
class _Prepared:
    request_id: str
    started: float
    tenant: Tenant
    request: CompletionRequest
    messages: tuple[Message, ...]
    reservation: Reservation
    traces: list[GuardTrace]
    prompt_hash: str
    context: str
    vault: PiiVault
    blocked_stage: str | None
    blocked_reasons: tuple[str, ...]
    input_decision: str
    pii_fingerprint: str = ""
    committed: bool = False
    model_name: str = ""
    cache_result: str = "bypass"
    cost_usd: float = 0.0
    usage: Usage = field(default_factory=lambda: Usage(0, 0))


class Gateway:
    def __init__(
        self,
        *,
        settings: Settings,
        limiter: RateLimiter,
        budget: BudgetLedger,
        cache: SemanticCache,
        router: ModelRouter,
        guardrails: GuardrailPipeline,
        audit: AuditLog,
        metrics: GatewayMetrics,
        tracer: Tracer | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.settings = settings
        self.limiter = limiter
        self.budget = budget
        self.cache = cache
        self.router = router
        self.guardrails = guardrails
        self.audit = audit
        self.metrics = metrics
        self._tracer = tracer or trace.get_tracer("llm_guard")
        self._clock = clock or time.time

    async def execute(self, tenant: Tenant, request: CompletionRequest) -> CompletionResult:
        prepared: _Prepared | None = None
        status = "error"
        try:
            with self._span("gateway.request") as span:
                span.set_attribute("llm_guard.tenant", tenant.id)
                span.set_attribute("llm_guard.model", request.model)
                prepared = self._preflight(tenant, request)
                if prepared.blocked_stage is not None:
                    raise GuardrailBlocked(prepared.blocked_stage, list(prepared.blocked_reasons))
                cached = await self._read_cache(prepared)
                if cached is not None:
                    status = "ok"
                    return cached
                result = await self._complete(prepared)
                status = "ok"
                return result
        except GatewayError as exc:
            status = exc.code
            raise
        finally:
            self._finish(prepared, request, status, tenant)

    async def stream(self, tenant: Tenant, request: CompletionRequest) -> tuple[dict[str, str], AsyncIterator[str]]:
        """Start a stream.

        Buffered mode finishes provider I/O and output guardrails before the
        first byte, so a block is still an HTTP error. Incremental mode can
        only fail closed between chunks.
        """

        mode = tenant.stream_mode or self.settings.guardrails.stream_mode
        if mode == "buffered":
            result = await self.execute(tenant, replace_stream(request, stream=False))
            return result_headers(result), _async_iter(iter_buffered(result, include_usage=request.include_usage))
        return await self._open_incremental(tenant, request)

    def _preflight(self, tenant: Tenant, request: CompletionRequest) -> _Prepared:
        started = time.perf_counter()
        total_chars = sum(len(message.content) for message in request.messages)
        if total_chars > self.settings.server.max_body_chars:
            raise PayloadTooLargeError(f"Request exceeds {self.settings.server.max_body_chars} characters")
        self.limiter.consume(tenant.id)
        prompt_tokens = estimate_tokens(render_messages(request.messages))
        reserved = reservation_micro(
            prompt_tokens=prompt_tokens,
            reserved_output_tokens=max(request.max_tokens, self.settings.server.reservation_output_tokens),
            input_per_million=self._max_input_price(request.model),
            output_per_million=self._max_output_price(request.model),
        )
        reservation = self.budget.reserve(
            tenant.id,
            reserved,
            cap_micro=usd_to_micro(tenant.daily_budget_usd),
        )
        try:
            return self._guard_input(tenant, request, reservation, started)
        except Exception:
            self.budget.release(reservation)
            raise

    def _guard_input(
        self,
        tenant: Tenant,
        request: CompletionRequest,
        reservation: Reservation,
        started: float,
    ) -> _Prepared:
        request_id = request_id_var.get() or _new_id("req")
        prompt_hash = hashlib.sha256(render_messages(request.messages).encode("utf-8")).hexdigest()
        context = "\n".join(message.content for message in request.messages if message.role == "system")
        vault = PiiVault()
        ctx = GuardContext(
            tenant=tenant,
            direction="input",
            context=context,
            response_format=request.response_format,
            vault=vault,
        )
        with self._span("guardrails.input"):
            outcome = self.guardrails.run_input(request.messages, ctx)
        traces_to_audit(
            self.audit,
            outcome.traces,
            request_id=request_id,
            tenant_id=tenant.id,
            model=request.model,
            prompt_sha256=prompt_hash,
        )
        self.metrics.record_guardrails(outcome.traces)
        return _Prepared(
            request_id=request_id,
            started=started,
            tenant=tenant,
            request=request,
            messages=outcome.messages,
            reservation=reservation,
            traces=list(outcome.traces),
            prompt_hash=prompt_hash,
            context=context,
            vault=vault,
            blocked_stage=outcome.stage,
            blocked_reasons=outcome.reasons,
            input_decision=outcome.decision,
            pii_fingerprint=vault.fingerprint(),
            model_name=request.model,
        )

    async def _read_cache(self, prepared: _Prepared) -> CompletionResult | None:
        if not _cacheable(prepared.request):
            prepared.cache_result = "bypass"
            return None
        with self._span("cache.lookup"):
            hit = await self.cache.lookup(
                _cache_namespace(prepared),
                render_messages(prepared.messages),
            )
        if hit is None:
            prepared.cache_result = "miss"
            return None
        prepared.cache_result = "hit"
        prepared.model_name = hit.model
        prepared.usage = Usage(hit.prompt_tokens, hit.completion_tokens)
        self.budget.commit(prepared.reservation, 0)
        prepared.committed = True
        return CompletionResult(
            id=_new_id("chatcmpl"),
            created=int(self._clock()),
            model=hit.model,
            text=hit.text,
            usage=prepared.usage,
            cost_usd=0.0,
            cache="hit",
            decision=hit.decision,
            traces=tuple(prepared.traces),
        )

    async def _complete(self, prepared: _Prepared) -> CompletionResult:
        with self._span("provider.call"):
            spec, response = await self.router.invoke(prepared.request.model, _provider_request(prepared))
        prepared.model_name = spec.name
        with self._span("guardrails.output"):
            outcome = self.guardrails.run_output(response.text, self._output_context(prepared))
        self._record_stage(prepared, outcome.traces)
        if outcome.blocked and outcome.stage is not None:
            raise GuardrailBlocked(outcome.stage, list(outcome.reasons))
        self._charge(prepared, spec, response.usage)
        decision = _merge_decision(prepared.input_decision, outcome.decision)
        result = CompletionResult(
            id=_new_id("chatcmpl"),
            created=int(self._clock()),
            model=spec.name,
            text=outcome.text,
            usage=response.usage,
            cost_usd=prepared.cost_usd,
            cache=prepared.cache_result,
            decision=decision,
            traces=tuple(prepared.traces),
        )
        if prepared.cache_result == "miss":
            await self.cache.store(
                _cache_namespace(prepared),
                render_messages(prepared.messages),
                response_text=outcome.text,
                model=spec.name,
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                decision=decision,
            )
        return result

    async def _open_incremental(
        self, tenant: Tenant, request: CompletionRequest
    ) -> tuple[dict[str, str], AsyncIterator[str]]:
        prepared: _Prepared | None = None
        try:
            prepared = self._preflight(tenant, request)
            if prepared.blocked_stage is not None:
                raise GuardrailBlocked(prepared.blocked_stage, list(prepared.blocked_reasons))
            prepared.cache_result = "bypass"
            start = await self.router.open_stream(request.model, _provider_request(prepared))
            prepared.model_name = start.spec.name
        except GatewayError as exc:
            self._finish(prepared, request, exc.code, tenant)
            raise
        headers = result_headers_partial(prepared)

        async def body() -> AsyncIterator[str]:
            assert prepared is not None
            status = "error"
            try:
                async for line in self._incremental_lines(prepared, start):
                    yield line
                status = "guardrail_blocked" if prepared.blocked_stage else "ok"
            except GatewayError as exc:
                status = exc.code
                raise
            except (GeneratorExit, asyncio.CancelledError):
                status = "cancelled"
                raise
            finally:
                self._finish(prepared, request, status, tenant)

        return headers, body()

    async def _incremental_lines(self, prepared: _Prepared, start: StreamStart) -> AsyncIterator[str]:
        completion_id = _new_id("chatcmpl")
        created = int(self._clock())
        model = start.spec.name
        failed = False
        try:
            yield sse(
                completion_chunk(
                    completion_id=completion_id,
                    created=created,
                    model=model,
                    delta={"role": "assistant"},
                    finish_reason=None,
                )
            )
            accumulated = ""
            usage: Usage | None = None
            async for event in _chain(start.first, start.rest):
                if event.usage is not None:
                    usage = event.usage
                if not event.text:
                    continue
                prospective = accumulated + event.text
                scan = self.guardrails.scan_partial(prospective, self._output_context(prepared))
                if scan.blocked:
                    prepared.blocked_stage = scan.stage
                    prepared.blocked_reasons = scan.reasons
                    self._record_stage(prepared, scan.traces)
                    self._charge(prepared, start.spec, usage or usage_for(prepared.messages, accumulated))
                    yield sse(
                        completion_chunk(
                            completion_id=completion_id,
                            created=created,
                            model=model,
                            delta=None,
                            finish_reason="content_filter",
                        )
                    )
                    yield sse_done()
                    return
                accumulated = prospective
                yield sse(
                    completion_chunk(
                        completion_id=completion_id,
                        created=created,
                        model=model,
                        delta={"content": event.text},
                        finish_reason=None,
                    )
                )
            self._note_grounding(prepared, accumulated)
            final_usage = usage or usage_for(prepared.messages, accumulated)
            self._charge(prepared, start.spec, final_usage)
            include = prepared.request.include_usage
            yield sse(
                completion_chunk(
                    completion_id=completion_id,
                    created=created,
                    model=model,
                    delta=None,
                    finish_reason="stop",
                    usage=final_usage if include else None,
                )
            )
            yield sse_done()
        except Exception:
            failed = True
            raise
        finally:
            if failed:
                start.fail()
            else:
                start.succeed()
            await _aclose(start.rest)

    def _note_grounding(self, prepared: _Prepared, text: str) -> None:
        if not self.settings.guardrails.grounding.enabled:
            return
        started = time.perf_counter()
        report = grounding_report(text, prepared.context)
        minimum = (
            prepared.tenant.grounding_min_score
            if prepared.tenant.grounding_min_score is not None
            else self.settings.guardrails.grounding.min_score
        )
        decision = _grounding_decision(report.score, report.reasons, minimum)
        trace = GuardTrace(
            stage="grounding",
            direction="output",
            decision=decision,
            reasons=report.reasons,
            score=report.score,
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        self._record_stage(prepared, (trace,))

    def _charge(self, prepared: _Prepared, spec: ModelSpec, usage: Usage) -> None:
        cost_micro = usage_cost_micro(
            usage,
            input_per_million=spec.input_per_million,
            output_per_million=spec.output_per_million,
        )
        if not prepared.committed:
            self.budget.commit(prepared.reservation, cost_micro)
            prepared.committed = True
        prepared.usage = usage
        prepared.cost_usd = micro_to_usd(cost_micro)

    def _output_context(self, prepared: _Prepared) -> GuardContext:
        return GuardContext(
            tenant=prepared.tenant,
            direction="output",
            context=prepared.context,
            response_format=prepared.request.response_format,
            vault=prepared.vault,
        )

    def _record_stage(self, prepared: _Prepared, traces: tuple[GuardTrace, ...]) -> None:
        prepared.traces.extend(traces)
        traces_to_audit(
            self.audit,
            traces,
            request_id=prepared.request_id,
            tenant_id=prepared.tenant.id,
            model=prepared.model_name or prepared.request.model,
            prompt_sha256=prepared.prompt_hash,
        )
        self.metrics.record_guardrails(traces)

    def _finish(
        self,
        prepared: _Prepared | None,
        request: CompletionRequest,
        status: str,
        tenant: Tenant,
    ) -> None:
        if prepared is not None and not prepared.committed:
            self.budget.release(prepared.reservation)
        latency = 0.0 if prepared is None else time.perf_counter() - prepared.started
        model = request.model if prepared is None or not prepared.model_name else prepared.model_name
        cache_result = "bypass" if prepared is None else prepared.cache_result
        cost = 0.0 if prepared is None else prepared.cost_usd
        usage = Usage(0, 0) if prepared is None else prepared.usage
        self.metrics.record_request(
            tenant=tenant.id,
            model=model,
            status=status,
            latency_s=latency,
            cache_result=cache_result,
            cost_usd=cost,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
        )
        self.metrics.sync_circuits({name: breaker.state for name, breaker in self.router.breakers.items()})
        if prepared is None:
            return
        self.audit.write(
            audit_event(
                request_id=prepared.request_id,
                tenant_id=tenant.id,
                stage="gateway",
                direction="request",
                decision=status,
                reasons=(f"cache:{cache_result}",),
                model=model,
                latency_ms=latency * 1000,
                prompt_sha256=prepared.prompt_hash,
            )
        )
        logger.info(
            "request completed",
            extra={
                "request_id": prepared.request_id,
                "tenant": tenant.id,
                "model": model,
                "status": status,
                "latency_ms": round(latency * 1000, 3),
                "cache": cache_result,
                "cost_usd": cost,
            },
        )

    def _max_input_price(self, requested: str) -> float:
        return max(spec.input_per_million for spec in self.router.candidates(requested))

    def _max_output_price(self, requested: str) -> float:
        return max(spec.output_per_million for spec in self.router.candidates(requested))

    @contextmanager
    def _span(self, name: str) -> Iterator[Span]:
        with self._tracer.start_as_current_span(name) as span:
            yield span


def result_headers(result: CompletionResult) -> dict[str, str]:
    headers = {
        "x-llm-guard-cache": result.cache,
        "x-llm-guard-model": result.model,
        "x-llm-guard-decision": result.decision,
        "x-llm-guard-cost-usd": f"{result.cost_usd:.8f}",
    }
    request_id = request_id_var.get()
    if request_id:
        headers["x-request-id"] = request_id
    return headers


def result_headers_partial(prepared: _Prepared) -> dict[str, str]:
    return {
        "x-request-id": prepared.request_id,
        "x-llm-guard-cache": "bypass",
        "x-llm-guard-model": prepared.model_name,
        "x-llm-guard-decision": "streaming",
    }


def replace_stream(request: CompletionRequest, *, stream: bool) -> CompletionRequest:
    return CompletionRequest(
        model=request.model,
        messages=request.messages,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
        stream=stream,
        response_format=request.response_format,
        include_usage=request.include_usage,
        use_cache=request.use_cache,
    )


def _provider_request(prepared: _Prepared) -> ProviderRequest:
    return ProviderRequest(
        model=prepared.request.model,
        messages=prepared.messages,
        temperature=prepared.request.temperature,
        max_tokens=prepared.request.max_tokens,
    )


def _cacheable(request: CompletionRequest) -> bool:
    return request.use_cache and request.temperature == 0.0 and not request.stream


def _cache_namespace(prepared: _Prepared) -> str:
    """Tenant, requested model, and the input PII values.

    Redaction maps different emails onto the same placeholder. Without the
    fingerprint, those prompts share a hit and the cached completion can
    contain the other caller's restored value.
    """

    base = f"{prepared.tenant.id}:{prepared.request.model}"
    if not prepared.pii_fingerprint:
        return base
    return f"{base}:{prepared.pii_fingerprint}"


def _merge_decision(input_decision: str, output_decision: str) -> str:
    if "block" in {input_decision, output_decision}:
        return "block"
    if "redact" in {input_decision, output_decision}:
        return "redact"
    return "allow"


def _grounding_decision(score: float, reasons: tuple[str, ...], minimum: float) -> str:
    if reasons in {("no_context",), ("abstained",)}:
        return "allow"
    if score < minimum:
        return "warn"
    return "allow"


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:24]}"


async def _aclose(iterator: AsyncIterator[object]) -> None:
    close = getattr(iterator, "aclose", None)
    if close is None:
        return
    try:
        await close()
    except RuntimeError:
        return


async def _chain(first: ProviderEvent, rest: AsyncIterator[ProviderEvent]) -> AsyncIterator[ProviderEvent]:
    yield first
    async for event in rest:
        yield event


def _async_iter(lines: Iterator[str]) -> AsyncIterator[str]:
    async def generate() -> AsyncIterator[str]:
        for line in lines:
            yield line

    return generate()
