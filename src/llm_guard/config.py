"""YAML configuration for tenants, routes, providers, and guardrails."""

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

StreamMode = Literal["buffered", "incremental"]
PiiMode = Literal["redact", "block"]
ProviderName = Literal["fake", "openai", "anthropic"]
FakeBehavior = Literal["faithful", "leak", "echo", "ungrounded", "json", "fail"]


class TopicRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    action: Literal["block"] = "block"
    patterns: list[str]


class PolicyPack(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    deny_topics: list[TopicRule] = Field(default_factory=list)


class ApiKeyRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    sha256: str


class TenantConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    policy_pack: str
    daily_budget_usd: float
    requests_per_minute: float
    burst: float
    stream_mode: StreamMode | None = None
    grounding_min_score: float | None = None
    pii_mode: PiiMode | None = None
    reversible_pii: list[str] | None = None
    api_keys: list[ApiKeyRecord]


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider: ProviderName
    upstream_model: str
    behavior: FakeBehavior = "faithful"
    timeout_seconds: float = 10.0
    input_per_million_usd: float
    output_per_million_usd: float
    latency_seconds: float = 0.0
    fail_times: int = 0


class ServerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    host: str = "0.0.0.0"
    port: int = 8080
    admin_api_key: str = "lg_admin_local_dev"
    expose_diagnostics: bool = False
    audit_log_path: str = "var/audit/decisions.jsonl"
    max_body_chars: int = 32_000
    reservation_output_tokens: int = 256
    log_level: str = "INFO"


class CacheSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    backend: Literal["memory", "redis"] = "memory"
    redis_url: str = "redis://localhost:6379/0"
    ttl_seconds: float = 300.0
    similarity_threshold: float = 0.86
    max_entries: int = 200
    embed_dim: int = 128


class RetrySettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    attempts: int = 2
    base_delay_seconds: float = 0.05
    max_delay_seconds: float = 0.2


class BreakerSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    failure_threshold: int = 3
    recovery_seconds: float = 15.0


class ProviderSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    api_key_env: str
    base_url: str


class InjectionSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    threshold: float = 0.5
    model_path: str = "data/models/injection_classifier.json"


class PiiSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    mode: PiiMode = "redact"
    reversible_types: list[str] = Field(default_factory=lambda: ["email"])


class GroundingSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    min_score: float = 0.45


class SchemaSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True


class GuardrailSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    injection: InjectionSettings = Field(default_factory=InjectionSettings)
    pii: PiiSettings = Field(default_factory=PiiSettings)
    grounding: GroundingSettings = Field(default_factory=GroundingSettings)
    schema_validation: SchemaSettings = Field(default_factory=SchemaSettings)
    stream_mode: StreamMode = "buffered"


class Settings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    server: ServerConfig = Field(default_factory=ServerConfig)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    retry: RetrySettings = Field(default_factory=RetrySettings)
    circuit_breaker: BreakerSettings = Field(default_factory=BreakerSettings)
    providers: dict[str, ProviderSettings] = Field(default_factory=dict)
    models: dict[str, ModelConfig]
    routes: dict[str, list[str]]
    guardrails: GuardrailSettings = Field(default_factory=GuardrailSettings)
    policy_packs: dict[str, PolicyPack]
    tenants: list[TenantConfig]

    def resolve_path(self, relative: str) -> Path:
        candidate = Path(relative)
        if candidate.is_file():
            return candidate
        return repo_root() / relative


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_settings(path: Path | None = None) -> Settings:
    config_path = path or Path(os.environ.get("LLM_GUARD_CONFIG", repo_root() / "configs" / "gateway.yaml"))
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"config {config_path} must be a mapping")
    document: dict[str, object] = {str(key): value for key, value in raw.items()}
    policy_dir = config_path.parent / "policies"
    packs: dict[str, object] = {}
    inline = document.get("policy_packs")
    if isinstance(inline, dict):
        packs.update({str(key): value for key, value in inline.items()})
    if policy_dir.is_dir():
        for file in sorted(policy_dir.glob("*.yaml")):
            loaded = yaml.safe_load(file.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict) or not isinstance(loaded.get("id"), str):
                raise ValueError(f"policy pack {file} needs an id")
            packs[str(loaded["id"])] = loaded
    document["policy_packs"] = packs
    settings = Settings.model_validate(document)
    return _apply_env(settings)


def _apply_env(settings: Settings) -> Settings:
    server_updates: dict[str, object] = {}
    admin = os.environ.get("LLM_GUARD_ADMIN_KEY")
    if admin:
        server_updates["admin_api_key"] = admin
    cache_updates: dict[str, object] = {}
    redis_url = os.environ.get("REDIS_URL")
    if redis_url:
        cache_updates["redis_url"] = redis_url
    backend = os.environ.get("LLM_GUARD_CACHE")
    if backend in {"memory", "redis"}:
        cache_updates["backend"] = backend
    updates: dict[str, object] = {}
    if server_updates:
        updates["server"] = settings.server.model_copy(update=server_updates)
    if cache_updates:
        updates["cache"] = settings.cache.model_copy(update=cache_updates)
    merged = settings.model_copy(update=updates) if updates else settings
    _validate(merged)
    return merged


def _validate(settings: Settings) -> None:
    seen_tenants: set[str] = set()
    seen_hashes: set[str] = set()
    for tenant in settings.tenants:
        if tenant.id in seen_tenants:
            raise ValueError(f"duplicate tenant id {tenant.id}")
        seen_tenants.add(tenant.id)
        if tenant.policy_pack not in settings.policy_packs:
            raise ValueError(f"tenant {tenant.id} references unknown policy {tenant.policy_pack}")
        for key in tenant.api_keys:
            if key.sha256 in seen_hashes:
                raise ValueError(f"duplicate api key hash on tenant {tenant.id}")
            seen_hashes.add(key.sha256)
    for route, names in settings.routes.items():
        for name in names:
            if name not in settings.models:
                raise ValueError(f"route {route} references unknown model {name}")
