"""Tenant registry and constant-time API key authentication."""

import hashlib
import hmac
from dataclasses import dataclass

from llm_guard.config import TenantConfig
from llm_guard.errors import AuthenticationError


def hash_api_key(raw_key: str) -> str:
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class Tenant:
    id: str
    policy_pack: str
    daily_budget_usd: float
    requests_per_minute: float
    burst: float
    stream_mode: str | None
    grounding_min_score: float | None
    pii_mode: str | None
    reversible_pii: tuple[str, ...] | None

    @classmethod
    def from_config(cls, config: TenantConfig) -> "Tenant":
        reversible = None if config.reversible_pii is None else tuple(config.reversible_pii)
        return cls(
            id=config.id,
            policy_pack=config.policy_pack,
            daily_budget_usd=config.daily_budget_usd,
            requests_per_minute=config.requests_per_minute,
            burst=config.burst,
            stream_mode=config.stream_mode,
            grounding_min_score=config.grounding_min_score,
            pii_mode=config.pii_mode,
            reversible_pii=reversible,
        )


@dataclass(frozen=True, slots=True)
class _KeyEntry:
    digest: str
    tenant: Tenant


class TenantRegistry:
    def __init__(self, tenants: list[Tenant], key_hashes: dict[str, list[str]]) -> None:
        self._by_id = {tenant.id: tenant for tenant in tenants}
        entries: list[_KeyEntry] = []
        for tenant_id, digests in key_hashes.items():
            tenant = self._by_id[tenant_id]
            for digest in digests:
                entries.append(_KeyEntry(digest=digest, tenant=tenant))
        self._entries = tuple(entries)

    def get(self, tenant_id: str) -> Tenant:
        try:
            return self._by_id[tenant_id]
        except KeyError as exc:
            raise KeyError(tenant_id) from exc

    def ids(self) -> tuple[str, ...]:
        return tuple(self._by_id)

    def authenticate(self, raw_key: str | None) -> Tenant:
        if raw_key is None or raw_key == "":
            raise AuthenticationError("Missing API key")
        digest = hash_api_key(raw_key)
        found: Tenant | None = None
        for entry in self._entries:
            if hmac.compare_digest(digest, entry.digest):
                found = entry.tenant
        if found is None:
            raise AuthenticationError("Invalid API key")
        return found

    def authenticate_admin(self, raw_key: str | None, admin_key: str) -> None:
        if raw_key is None or raw_key == "":
            raise AuthenticationError("Missing admin API key")
        if not hmac.compare_digest(hash_api_key(raw_key), hash_api_key(admin_key)):
            raise AuthenticationError("Invalid admin API key")
