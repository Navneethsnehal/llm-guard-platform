"""Run guardrail stages in order and collect a decision for each one."""

import time
from dataclasses import dataclass, field

from llm_guard.auth import Tenant
from llm_guard.config import Settings
from llm_guard.guardrails.classifier import InjectionModel
from llm_guard.guardrails.grounding import grounding_report
from llm_guard.guardrails.injection import InjectionStage
from llm_guard.guardrails.pii import PiiVault, detect_pii, redact
from llm_guard.guardrails.schema_check import check_schema
from llm_guard.guardrails.topics import TopicsStage
from llm_guard.types import GuardTrace, Message, ResponseFormat


@dataclass(frozen=True, slots=True)
class PipelineOutcome:
    messages: tuple[Message, ...]
    text: str
    blocked: bool
    stage: str | None
    reasons: tuple[str, ...]
    traces: tuple[GuardTrace, ...]
    decision: str


@dataclass
class GuardContext:
    tenant: Tenant
    direction: str
    context: str
    response_format: ResponseFormat | None
    vault: PiiVault = field(default_factory=PiiVault)


class GuardrailPipeline:
    def __init__(
        self,
        settings: Settings,
        topics: TopicsStage,
        injection: InjectionStage,
    ) -> None:
        self._settings = settings
        self._topics = topics
        self._injection = injection

    def run_input(self, messages: tuple[Message, ...], ctx: GuardContext) -> PipelineOutcome:
        traces: list[GuardTrace] = []
        transcript = "\n".join(f"{message.role}: {message.content}" for message in messages)
        caller_text = "\n".join(message.content for message in messages if message.role != "system")
        if self._settings.guardrails.injection.enabled:
            blocked = self._apply_injection(transcript, "input", traces)
            if blocked is not None:
                return self._blocked(messages, transcript, blocked, traces)
        blocked = self._apply_topics(caller_text, ctx.tenant.policy_pack, "input", traces)
        if blocked is not None:
            return self._blocked(messages, transcript, blocked, traces)
        rewritten: list[Message] = []
        kinds: list[str] = []
        if self._settings.guardrails.pii.enabled:
            mode = ctx.tenant.pii_mode or self._settings.guardrails.pii.mode
            started = time.perf_counter()
            for message in messages:
                redacted, found = redact(message.content, ctx.vault)
                if found and mode == "block":
                    trace = _trace(
                        "pii",
                        "input",
                        "block",
                        found,
                        None,
                        started,
                    )
                    traces.append(trace)
                    return self._blocked(messages, transcript, trace, traces)
                if found:
                    kinds.extend(found)
                rewritten.append(Message(role=message.role, content=redacted))
            decision = "redact" if kinds else "allow"
            traces.append(_trace("pii", "input", decision, tuple(dict.fromkeys(kinds)), None, started))
        else:
            rewritten = list(messages)
        text = "\n".join(message.content for message in rewritten)
        return PipelineOutcome(
            messages=tuple(rewritten),
            text=text,
            blocked=False,
            stage=None,
            reasons=(),
            traces=tuple(traces),
            decision=_aggregate(traces),
        )

    def run_output(self, text: str, ctx: GuardContext) -> PipelineOutcome:
        traces: list[GuardTrace] = []
        current = text
        if self._settings.guardrails.pii.enabled:
            mode = ctx.tenant.pii_mode or self._settings.guardrails.pii.mode
            started = time.perf_counter()
            redacted, found = redact(current, ctx.vault)
            if found and mode == "block":
                trace = _trace("pii", "output", "block", found, None, started)
                traces.append(trace)
                return self._blocked((), current, trace, traces)
            decision = "redact" if found else "allow"
            traces.append(_trace("pii", "output", decision, tuple(dict.fromkeys(found)), None, started))
            current = redacted
        if self._settings.guardrails.injection.enabled:
            blocked = self._apply_injection(current, "output", traces)
            if blocked is not None:
                return self._blocked((), current, blocked, traces)
        blocked = self._apply_topics(current, ctx.tenant.policy_pack, "output", traces)
        if blocked is not None:
            return self._blocked((), current, blocked, traces)
        if self._settings.guardrails.schema_validation.enabled and ctx.response_format is not None:
            started = time.perf_counter()
            checked = check_schema(
                current,
                kind=ctx.response_format.type,
                schema=ctx.response_format.json_schema,
            )
            traces.append(_trace("schema", "output", checked.decision, checked.reasons, None, started))
            if checked.decision == "block":
                return self._blocked((), current, traces[-1], traces)
            current = checked.text
        if self._settings.guardrails.grounding.enabled:
            started = time.perf_counter()
            report = grounding_report(current, ctx.context)
            minimum = (
                ctx.tenant.grounding_min_score
                if ctx.tenant.grounding_min_score is not None
                else self._settings.guardrails.grounding.min_score
            )
            if report.reasons == ("no_context",) or report.reasons == ("abstained",):
                decision = "allow"
            elif report.score < minimum:
                decision = "block"
            else:
                decision = "allow"
            trace = _trace("grounding", "output", decision, report.reasons, report.score, started)
            traces.append(trace)
            if decision == "block":
                return self._blocked((), current, trace, traces)
        reversible = set(self._reversible(ctx.tenant))
        rendered = ctx.vault.render(current, reversible)
        return PipelineOutcome(
            messages=(),
            text=rendered,
            blocked=False,
            stage=None,
            reasons=(),
            traces=tuple(traces),
            decision=_aggregate(traces),
        )

    def scan_partial(self, text: str, ctx: GuardContext) -> PipelineOutcome:
        """Lightweight output scan used by incremental streaming.

        Grounding and schema run only on the buffered path, where the gateway
        can still withhold the response. Incremental mode fail-closes if this
        scan blocks, and records grounding separately after the flush.
        """

        traces: list[GuardTrace] = []
        if self._settings.guardrails.injection.enabled:
            blocked = self._apply_injection(text, "output", traces)
            if blocked is not None:
                return self._blocked((), text, blocked, traces)
        blocked = self._apply_topics(text, ctx.tenant.policy_pack, "output", traces)
        if blocked is not None:
            return self._blocked((), text, blocked, traces)
        if self._settings.guardrails.pii.enabled and detect_pii(text):
            started = time.perf_counter()
            trace = _trace("pii", "output", "block", ("partial_pii",), None, started)
            traces.append(trace)
            return self._blocked((), text, trace, traces)
        return PipelineOutcome(
            messages=(),
            text=text,
            blocked=False,
            stage=None,
            reasons=(),
            traces=tuple(traces),
            decision="allow",
        )

    def _apply_injection(self, text: str, direction: str, traces: list[GuardTrace]) -> GuardTrace | None:
        started = time.perf_counter()
        decision = self._injection.evaluate(text)
        trace = _trace(
            "injection",
            direction,
            decision.decision,
            decision.reasons,
            decision.score,
            started,
        )
        traces.append(trace)
        if decision.decision == "block":
            return trace
        return None

    def _apply_topics(self, text: str, pack_id: str, direction: str, traces: list[GuardTrace]) -> GuardTrace | None:
        started = time.perf_counter()
        decision = self._topics.evaluate(text, pack_id)
        trace = _trace("topics", direction, decision.decision, decision.reasons, None, started)
        traces.append(trace)
        if decision.decision == "block":
            return trace
        return None

    def _reversible(self, tenant: Tenant) -> tuple[str, ...]:
        if tenant.reversible_pii is not None:
            return tenant.reversible_pii
        return tuple(self._settings.guardrails.pii.reversible_types)

    def _blocked(
        self,
        messages: tuple[Message, ...],
        text: str,
        trace: GuardTrace,
        traces: list[GuardTrace],
    ) -> PipelineOutcome:
        return PipelineOutcome(
            messages=messages,
            text=text,
            blocked=True,
            stage=trace.stage,
            reasons=trace.reasons,
            traces=tuple(traces),
            decision="block",
        )


def _aggregate(traces: list[GuardTrace]) -> str:
    if any(trace.decision == "block" for trace in traces):
        return "block"
    if any(trace.decision == "redact" for trace in traces):
        return "redact"
    return "allow"


def _trace(
    stage: str,
    direction: str,
    decision: str,
    reasons: tuple[str, ...],
    score: float | None,
    started: float,
) -> GuardTrace:
    return GuardTrace(
        stage=stage,
        direction=direction,
        decision=decision,
        reasons=reasons,
        score=score,
        latency_ms=(time.perf_counter() - started) * 1000,
    )


def load_injection_model(settings: Settings) -> InjectionModel | None:
    if not settings.guardrails.injection.enabled:
        return None
    path = settings.resolve_path(settings.guardrails.injection.model_path)
    if not path.is_file():
        return None
    return InjectionModel.load(path)
