"""Policy routing, jittered retries, and circuit breakers."""

import asyncio
import random
import threading
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import TypeVar

from llm_guard.config import ModelConfig
from llm_guard.errors import ProviderError, UnknownModelError, UpstreamError
from llm_guard.providers import ChatProvider
from llm_guard.ratelimit import compute_delay
from llm_guard.types import ProviderEvent, ProviderRequest, ProviderResponse


class CircuitBreaker:
    """Closed, open, and half-open breaker with an injectable clock."""

    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        recovery_s: float = 30.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if failure_threshold < 1:
            raise ValueError("failure_threshold must be at least 1")
        self.failure_threshold = failure_threshold
        self.recovery_s = recovery_s
        self._clock = clock or time.monotonic
        self._state = "closed"
        self._failures = 0
        self._opened_at = 0.0
        self._half_open_inflight = False
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        with self._lock:
            self._maybe_half_open()
            return self._state

    def _maybe_half_open(self) -> None:
        if self._state == "open" and self._clock() - self._opened_at >= self.recovery_s:
            self._state = "half_open"
            self._half_open_inflight = False

    def allow(self) -> bool:
        with self._lock:
            self._maybe_half_open()
            if self._state == "closed":
                return True
            if self._state == "open":
                return False
            if self._half_open_inflight:
                return False
            self._half_open_inflight = True
            return True

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._half_open_inflight = False
            self._state = "closed"

    def record_failure(self) -> None:
        with self._lock:
            self._half_open_inflight = False
            self._failures += 1
            if self._state == "half_open" or self._failures >= self.failure_threshold:
                self._state = "open"
                self._opened_at = self._clock()


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    attempts: int
    base_delay: float
    max_delay: float


Sleep = Callable[[float], Awaitable[None]]
Rng = Callable[[], float]
T = TypeVar("T")


async def call_with_retry(
    fn: Callable[[], Awaitable[T]],
    policy: RetryPolicy,
    *,
    sleep: Sleep = asyncio.sleep,
    rng: Rng = random.random,
) -> T:
    """Retry retryable provider errors with full jitter.

    The circuit breaker sees one outcome for the whole sequence: retries absorb
    blips, and the breaker opens only when the attempts are exhausted.
    """

    if policy.attempts < 1:
        raise ValueError("attempts must be at least 1")
    last: ProviderError | None = None
    for attempt in range(policy.attempts):
        try:
            return await fn()
        except ProviderError as exc:
            last = exc
            if not exc.retryable or attempt == policy.attempts - 1:
                raise
            delay = compute_delay(attempt, policy.base_delay, policy.max_delay, rng())
            await sleep(delay)
    raise ProviderError(last.message if last else "retry failed", retryable=True)


@dataclass(frozen=True, slots=True)
class ModelSpec:
    name: str
    provider: str
    upstream_model: str
    behavior: str
    timeout_s: float
    input_per_million: float
    output_per_million: float
    latency_s: float
    fail_times: int

    @classmethod
    def from_config(cls, name: str, config: ModelConfig) -> "ModelSpec":
        return cls(
            name=name,
            provider=config.provider,
            upstream_model=config.upstream_model,
            behavior=config.behavior,
            timeout_s=config.timeout_seconds,
            input_per_million=config.input_per_million_usd,
            output_per_million=config.output_per_million_usd,
            latency_s=config.latency_seconds,
            fail_times=config.fail_times,
        )


class ModelRouter:
    def __init__(
        self,
        models: dict[str, ModelSpec],
        routes: dict[str, list[str]],
        providers: dict[str, ChatProvider],
        breakers: dict[str, CircuitBreaker],
        retry: RetryPolicy,
        *,
        sleeper: Sleep = asyncio.sleep,
        rng: Rng = random.random,
    ) -> None:
        self.models = models
        self.routes = routes
        self.providers = providers
        self.breakers = breakers
        self.retry = retry
        self._sleeper = sleeper
        self._rng = rng

    def candidates(self, requested: str) -> list[ModelSpec]:
        if requested in self.routes:
            names = list(self.routes[requested])
        elif requested in self.models:
            names = [requested]
            for route_names in self.routes.values():
                if requested in route_names:
                    names.extend(name for name in route_names if name != requested)
                    break
        else:
            raise UnknownModelError(f"Unknown model '{requested}'")
        ordered: list[str] = []
        for name in names:
            if name not in ordered:
                ordered.append(name)
        missing = [name for name in ordered if name not in self.models]
        if missing:
            raise UnknownModelError(f"Route references unknown model '{missing[0]}'")
        return [self.models[name] for name in ordered]

    def known_ids(self) -> list[str]:
        ids = list(self.routes)
        for name in self.models:
            if name not in ids:
                ids.append(name)
        return ids

    async def invoke(self, requested: str, request: ProviderRequest) -> tuple[ModelSpec, ProviderResponse]:
        errors: list[str] = []
        for spec in self.candidates(requested):
            breaker = self.breakers[spec.name]
            if not breaker.allow():
                errors.append(f"{spec.name}: circuit open")
                continue
            provider = self.providers[spec.name]
            upstream = _upstream_request(spec, request)

            async def attempt(
                provider: ChatProvider = provider,
                spec: ModelSpec = spec,
                upstream: ProviderRequest = upstream,
            ) -> ProviderResponse:
                return await _invoke_provider(provider, upstream, spec.timeout_s)

            settled = False
            try:
                try:
                    response = await call_with_retry(
                        attempt,
                        self.retry,
                        sleep=self._sleeper,
                        rng=self._rng,
                    )
                except ProviderError as exc:
                    breaker.record_failure()
                    settled = True
                    errors.append(f"{spec.name}: {exc.message}")
                    continue
                breaker.record_success()
                settled = True
                return spec, response
            finally:
                if not settled:
                    breaker.record_failure()
        detail = "; ".join(errors) if errors else "no candidates"
        raise UpstreamError(f"No upstream model available ({detail})")

    async def open_stream(self, requested: str, request: ProviderRequest) -> "StreamStart":
        """Return the first upstream that yields an event.

        Fallback happens only before the first event. The caller records the
        breaker outcome when the stream finishes. An exception after ``allow``
        still settles the breaker, so a half-open trial cannot stick.
        """

        errors: list[str] = []
        for spec in self.candidates(requested):
            breaker = self.breakers[spec.name]
            if not breaker.allow():
                errors.append(f"{spec.name}: circuit open")
                continue
            provider = self.providers[spec.name]
            upstream = _upstream_request(spec, request)

            async def open_first(
                provider: ChatProvider = provider,
                spec: ModelSpec = spec,
                upstream: ProviderRequest = upstream,
            ) -> tuple[ProviderEvent, AsyncIterator[ProviderEvent]]:
                return await _open_first_event(provider, upstream, spec.timeout_s)

            settled = False
            try:
                try:
                    first, rest = await call_with_retry(
                        open_first,
                        self.retry,
                        sleep=self._sleeper,
                        rng=self._rng,
                    )
                except ProviderError as exc:
                    breaker.record_failure()
                    settled = True
                    errors.append(f"{spec.name}: {exc.message}")
                    continue
                settled = True
                return StreamStart(spec=spec, first=first, rest=rest, breaker=breaker)
            finally:
                if not settled:
                    breaker.record_failure()
        detail = "; ".join(errors) if errors else "no candidates"
        raise UpstreamError(f"No upstream model available ({detail})")


@dataclass(slots=True)
class StreamStart:
    spec: ModelSpec
    first: "ProviderEvent"
    rest: AsyncIterator["ProviderEvent"]
    breaker: CircuitBreaker

    def succeed(self) -> None:
        self.breaker.record_success()

    def fail(self) -> None:
        self.breaker.record_failure()


def _upstream_request(spec: ModelSpec, request: ProviderRequest) -> ProviderRequest:
    return ProviderRequest(
        model=spec.upstream_model,
        messages=request.messages,
        temperature=request.temperature,
        max_tokens=request.max_tokens,
    )


async def _aclose(iterator: AsyncIterator[object]) -> None:
    close = getattr(iterator, "aclose", None)
    if close is None:
        return
    try:
        await close()
    except RuntimeError:
        return


async def _open_first_event(
    provider: ChatProvider,
    request: ProviderRequest,
    timeout_s: float,
) -> tuple["ProviderEvent", AsyncIterator["ProviderEvent"]]:
    iterator = provider.stream(request, timeout=timeout_s)
    try:
        first = await asyncio.wait_for(iterator.__anext__(), timeout=timeout_s)
    except StopAsyncIteration as exc:
        await _aclose(iterator)
        raise ProviderError("provider returned an empty stream", retryable=True) from exc
    except TimeoutError as exc:
        await _aclose(iterator)
        raise ProviderError(f"provider timed out after {timeout_s:.3f}s", retryable=True) from exc
    except Exception:
        await _aclose(iterator)
        raise
    return first, iterator


async def _invoke_provider(
    provider: ChatProvider,
    request: ProviderRequest,
    timeout_s: float,
) -> ProviderResponse:
    try:
        return await asyncio.wait_for(provider.complete(request, timeout=timeout_s), timeout=timeout_s)
    except TimeoutError as exc:
        raise ProviderError(f"provider timed out after {timeout_s:.3f}s", retryable=True) from exc
