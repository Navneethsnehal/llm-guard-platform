"""Tracing, Prometheus metrics, JSON logs, and the guardrail audit log."""

import json
import logging
import threading
from collections import deque
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram, generate_latest

from llm_guard.types import GuardTrace

request_id_var: ContextVar[str] = ContextVar("llm_guard_request_id", default="")

_LOG_ATTRS = frozenset(
    {
        "args",
        "asctime",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
        "taskName",
    }
)
_TRACING_READY = False


def configure_tracing() -> None:
    """Install a process-wide tracer provider once.

    Spans are recorded for every request and guardrail stage. Set a span
    processor on the global provider before startup to export them. Without
    one, the SDK still produces valid span context for logs.
    """

    global _TRACING_READY
    if _TRACING_READY:
        return
    provider = TracerProvider(resource=Resource.create({"service.name": "llm-guard-gateway"}))
    trace.set_tracer_provider(provider)
    _TRACING_READY = True


def configure_logging(level: str) -> None:
    logger = logging.getLogger("llm_guard")
    logger.setLevel(level.upper())
    logger.propagate = False
    if logger.handlers:
        return
    handler = logging.StreamHandler()
    handler.setFormatter(JsonFormatter())
    logger.addHandler(handler)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key not in _LOG_ATTRS and not key.startswith("_"):
                payload[key] = value
        return json.dumps(payload, default=str, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class AuditEvent:
    ts: str
    request_id: str
    tenant_id: str
    stage: str
    direction: str
    decision: str
    reasons: tuple[str, ...]
    model: str
    latency_ms: float
    prompt_sha256: str

    def as_dict(self) -> dict[str, object]:
        return {
            "ts": self.ts,
            "request_id": self.request_id,
            "tenant_id": self.tenant_id,
            "stage": self.stage,
            "direction": self.direction,
            "decision": self.decision,
            "reasons": list(self.reasons),
            "model": self.model,
            "latency_ms": round(self.latency_ms, 3),
            "prompt_sha256": self.prompt_sha256,
        }


class AuditLog:
    """Append-only decision log. Prompts themselves are not stored."""

    def __init__(self, path: Path | None = None, *, maxlen: int = 1000) -> None:
        self._path = path
        self._events: deque[AuditEvent] = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def write(self, event: AuditEvent) -> None:
        line = json.dumps(event.as_dict(), separators=(",", ":"), sort_keys=True)
        with self._lock:
            self._events.append(event)
            if self._path is not None:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")

    def recent(self, limit: int = 100) -> list[AuditEvent]:
        with self._lock:
            if limit <= 0:
                return []
            return list(self._events)[-limit:]


def audit_event(
    *,
    request_id: str,
    tenant_id: str,
    stage: str,
    direction: str,
    decision: str,
    reasons: tuple[str, ...],
    model: str,
    latency_ms: float,
    prompt_sha256: str,
) -> AuditEvent:
    return AuditEvent(
        ts=datetime.now(UTC).isoformat(),
        request_id=request_id,
        tenant_id=tenant_id,
        stage=stage,
        direction=direction,
        decision=decision,
        reasons=reasons,
        model=model,
        latency_ms=latency_ms,
        prompt_sha256=prompt_sha256,
    )


def traces_to_audit(
    log: AuditLog,
    traces: tuple[GuardTrace, ...] | list[GuardTrace],
    *,
    request_id: str,
    tenant_id: str,
    model: str,
    prompt_sha256: str,
) -> None:
    for item in traces:
        log.write(
            audit_event(
                request_id=request_id,
                tenant_id=tenant_id,
                stage=item.stage,
                direction=item.direction,
                decision=item.decision,
                reasons=item.reasons,
                model=model,
                latency_ms=item.latency_ms,
                prompt_sha256=prompt_sha256,
            )
        )


class GatewayMetrics:
    def __init__(self, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        labels = ["tenant", "model", "status"]
        self.latency = Histogram(
            "llm_guard_request_latency_seconds",
            "End-to-end gateway latency.",
            labels,
            registry=self.registry,
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5),
        )
        self.requests = Counter(
            "llm_guard_requests_total",
            "Chat completion requests.",
            labels,
            registry=self.registry,
        )
        self.cache = Counter(
            "llm_guard_cache_lookups_total",
            "Semantic cache lookups.",
            ["result"],
            registry=self.registry,
        )
        self.guardrails = Counter(
            "llm_guard_guardrail_decisions_total",
            "Guardrail stage decisions.",
            ["stage", "direction", "decision"],
            registry=self.registry,
        )
        self.cost = Counter(
            "llm_guard_cost_usd_total",
            "Provider cost charged to tenants.",
            ["tenant", "model"],
            registry=self.registry,
        )
        self.tokens = Counter(
            "llm_guard_tokens_total",
            "Tokens observed by the gateway.",
            ["tenant", "direction"],
            registry=self.registry,
        )
        self.circuit_open = Gauge(
            "llm_guard_circuit_open",
            "1 when a model circuit breaker is open.",
            ["model"],
            registry=self.registry,
        )
        self._lock = threading.Lock()
        self._request_counts: dict[str, int] = {}
        self._cache_counts: dict[str, int] = {}
        self._cost_usd: dict[str, float] = {}

    def record_request(
        self,
        *,
        tenant: str,
        model: str,
        status: str,
        latency_s: float,
        cache_result: str,
        cost_usd: float,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> None:
        self.latency.labels(tenant=tenant, model=model, status=status).observe(latency_s)
        self.requests.labels(tenant=tenant, model=model, status=status).inc()
        self.cache.labels(result=cache_result).inc()
        if cost_usd:
            self.cost.labels(tenant=tenant, model=model).inc(cost_usd)
        if prompt_tokens:
            self.tokens.labels(tenant=tenant, direction="input").inc(prompt_tokens)
        if completion_tokens:
            self.tokens.labels(tenant=tenant, direction="output").inc(completion_tokens)
        with self._lock:
            self._request_counts[status] = self._request_counts.get(status, 0) + 1
            self._cache_counts[cache_result] = self._cache_counts.get(cache_result, 0) + 1
            self._cost_usd[tenant] = self._cost_usd.get(tenant, 0.0) + cost_usd

    def record_guardrails(self, traces: tuple[GuardTrace, ...] | list[GuardTrace]) -> None:
        for item in traces:
            self.guardrails.labels(
                stage=item.stage,
                direction=item.direction,
                decision=item.decision,
            ).inc()

    def sync_circuits(self, states: Mapping[str, str]) -> None:
        for model, state in states.items():
            self.circuit_open.labels(model=model).set(1.0 if state == "open" else 0.0)

    def render(self) -> bytes:
        payload: bytes = generate_latest(self.registry)
        return payload

    def counts(self) -> dict[str, object]:
        with self._lock:
            return {
                "requests": dict(self._request_counts),
                "cache": dict(self._cache_counts),
                "cost_usd": dict(self._cost_usd),
            }
