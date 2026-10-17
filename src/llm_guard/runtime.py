"""Wire settings into a gateway runtime."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

from opentelemetry.trace import Tracer

from llm_guard.auth import Tenant, TenantRegistry
from llm_guard.budget import BudgetLedger
from llm_guard.cache import AsyncRedisAdapter, Embedder, MemoryStore, RedisStore, SemanticCache
from llm_guard.config import Settings
from llm_guard.gateway import Gateway
from llm_guard.guardrails.classifier import InjectionModel
from llm_guard.guardrails.injection import InjectionStage
from llm_guard.guardrails.pipeline import GuardrailPipeline, load_injection_model
from llm_guard.guardrails.topics import TopicsStage
from llm_guard.observability import AuditLog, GatewayMetrics, configure_logging, configure_tracing
from llm_guard.providers import AnthropicProvider, ChatProvider, FakeProvider, OpenAIProvider
from llm_guard.ratelimit import RateLimiter
from llm_guard.routing import CircuitBreaker, ModelRouter, ModelSpec, RetryPolicy

logger = logging.getLogger("llm_guard.runtime")


@dataclass
class Runtime:
    settings: Settings
    tenants: TenantRegistry
    gateway: Gateway
    metrics: GatewayMetrics
    audit: AuditLog
    router: ModelRouter
    redis: AsyncRedisAdapter | None = None

    async def aclose(self) -> None:
        if self.redis is not None:
            await self.redis.aclose()

    def snapshot(self, *, audit_limit: int = 50) -> dict[str, object]:
        counts = self.metrics.counts()
        return {
            "requests": counts["requests"],
            "cache": counts["cache"],
            "cost_usd": counts["cost_usd"],
            "circuits": {name: breaker.state for name, breaker in self.router.breakers.items()},
            "audit": [event.as_dict() for event in self.audit.recent(audit_limit)],
        }


def build_runtime(settings: Settings, *, tracer: Tracer | None = None) -> Runtime:
    configure_tracing()
    configure_logging(settings.server.log_level)
    tenants = [Tenant.from_config(config) for config in settings.tenants]
    key_hashes = {config.id: [key.sha256 for key in config.api_keys] for config in settings.tenants}
    registry = TenantRegistry(tenants, key_hashes)
    limiter = RateLimiter()
    for tenant in tenants:
        limiter.configure(
            tenant.id,
            requests_per_minute=tenant.requests_per_minute,
            burst=tenant.burst,
        )
    budget = BudgetLedger()
    redis: AsyncRedisAdapter | None = None
    if settings.cache.backend == "redis":
        redis = AsyncRedisAdapter(settings.cache.redis_url)
        store: MemoryStore | RedisStore = RedisStore(redis)
    else:
        store = MemoryStore()
    cache = SemanticCache(
        store,
        Embedder(settings.cache.embed_dim),
        threshold=settings.cache.similarity_threshold,
        ttl_s=settings.cache.ttl_seconds,
        max_entries=settings.cache.max_entries,
    )
    models = {name: ModelSpec.from_config(name, config) for name, config in settings.models.items()}
    providers = _providers(settings, models)
    breakers = {
        name: CircuitBreaker(
            failure_threshold=settings.circuit_breaker.failure_threshold,
            recovery_s=settings.circuit_breaker.recovery_seconds,
        )
        for name in models
    }
    router = ModelRouter(
        models,
        {name: list(candidates) for name, candidates in settings.routes.items()},
        providers,
        breakers,
        RetryPolicy(
            attempts=settings.retry.attempts,
            base_delay=settings.retry.base_delay_seconds,
            max_delay=settings.retry.max_delay_seconds,
        ),
    )
    loaded = load_injection_model(settings)
    if loaded is not None and settings.guardrails.injection.threshold < loaded.threshold:
        logger.warning(
            "injection threshold %.3f is below the calibrated %.3f",
            settings.guardrails.injection.threshold,
            loaded.threshold,
        )
    pipeline = GuardrailPipeline(
        settings,
        TopicsStage.from_packs(settings.policy_packs),
        InjectionStage(loaded, _injection_threshold(settings, loaded)),
    )
    audit_path = _audit_path(settings)
    audit = AuditLog(audit_path)
    metrics = GatewayMetrics()
    gateway = Gateway(
        settings=settings,
        limiter=limiter,
        budget=budget,
        cache=cache,
        router=router,
        guardrails=pipeline,
        audit=audit,
        metrics=metrics,
        tracer=tracer,
    )
    logger.info(
        "runtime ready",
        extra={
            "tenants": len(tenants),
            "models": len(models),
            "cache": settings.cache.backend,
        },
    )
    return Runtime(
        settings=settings,
        tenants=registry,
        gateway=gateway,
        metrics=metrics,
        audit=audit,
        router=router,
        redis=redis,
    )


def _injection_threshold(settings: Settings, model: InjectionModel | None) -> float:
    configured = settings.guardrails.injection.threshold
    if model is None:
        return configured
    return max(configured, model.threshold)


def _audit_path(settings: Settings) -> Path | None:
    raw = settings.server.audit_log_path.strip()
    if not raw:
        return None
    return settings.resolve_path(raw)


def _providers(settings: Settings, models: dict[str, ModelSpec]) -> dict[str, ChatProvider]:
    built: dict[str, ChatProvider] = {}
    openai: OpenAIProvider | None = None
    anthropic: AnthropicProvider | None = None
    for name, spec in models.items():
        if spec.provider == "fake":
            built[name] = FakeProvider(
                name,
                behavior=spec.behavior,
                latency_s=spec.latency_s,
                fail_times=spec.fail_times,
            )
            continue
        if spec.provider == "openai":
            if openai is None:
                key, base_url = _credential(settings, "openai", "OPENAI_API_KEY", "https://api.openai.com/v1")
                openai = OpenAIProvider(api_key=key, base_url=base_url)
            built[name] = openai
            continue
        if anthropic is None:
            key, base_url = _credential(settings, "anthropic", "ANTHROPIC_API_KEY", "https://api.anthropic.com")
            anthropic = AnthropicProvider(api_key=key, base_url=base_url)
        built[name] = anthropic
    return built


def _credential(settings: Settings, name: str, env: str, default_url: str) -> tuple[str, str]:
    configured = settings.providers.get(name)
    if configured is None:
        return os.environ.get(env, ""), default_url
    return os.environ.get(configured.api_key_env, ""), configured.base_url
