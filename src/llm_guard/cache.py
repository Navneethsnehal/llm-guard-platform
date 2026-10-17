"""Tenant-scoped semantic cache.

Embeddings are hashed lexical vectors (word unigrams, bigrams, character
trigrams). They catch near-duplicate prompts without a hosted embedding
model. The store interface is small enough to back with memory or Redis.
"""

import asyncio
import hashlib
import itertools
import json
import math
import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

_TOKEN = re.compile(r"[a-z0-9]+")


class Embedder:
    def __init__(self, dim: int = 128) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def embed(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        tokens = _TOKEN.findall(text.lower())
        if not tokens:
            return vector
        for token in tokens:
            _accumulate(vector, "w:" + token, 1.0)
            for index in range(len(token) - 2):
                _accumulate(vector, "c:" + token[index : index + 3], 0.5)
        for left, right in itertools.pairwise(tokens):
            _accumulate(vector, "b:" + left + "_" + right, 1.0)
        return _normalize(vector)


def _accumulate(vector: list[float], feature: str, weight: float) -> None:
    digest = hashlib.sha256(feature.encode("utf-8")).digest()
    index = int.from_bytes(digest[:4], "little") % len(vector)
    sign = 1.0 if digest[4] % 2 == 0 else -1.0
    vector[index] += sign * weight


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for x_val, y_val in zip(left, right, strict=True):
        dot += x_val * y_val
        left_norm += x_val * x_val
        right_norm += y_val * y_val
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / math.sqrt(left_norm * right_norm)


class CacheStore(Protocol):
    async def get_all(self, namespace: str) -> list[str]: ...

    async def put(
        self,
        namespace: str,
        entry_json: str,
        *,
        ttl_s: float,
        max_entries: int,
    ) -> None: ...


class MemoryStore:
    def __init__(self, clock: Callable[[], float] | None = None) -> None:
        self._clock = clock or time.time
        self._items: dict[str, list[tuple[str, float]]] = {}
        self._lock = asyncio.Lock()

    async def get_all(self, namespace: str) -> list[str]:
        async with self._lock:
            now = self._clock()
            kept: list[tuple[str, float]] = []
            for payload, expires_at in self._items.get(namespace, []):
                if expires_at > now:
                    kept.append((payload, expires_at))
            self._items[namespace] = kept
            return [payload for payload, _expires in kept]

    async def put(
        self,
        namespace: str,
        entry_json: str,
        *,
        ttl_s: float,
        max_entries: int,
    ) -> None:
        async with self._lock:
            now = self._clock()
            expires_at = now + ttl_s
            current = [item for item in self._items.get(namespace, []) if item[1] > now]
            current.append((entry_json, expires_at))
            self._items[namespace] = current[-max_entries:]


class KeyValueClient(Protocol):
    async def get(self, key: str) -> str | None: ...

    async def set(self, key: str, value: str, ex: int | None = None) -> None: ...


class RedisStore:
    """One JSON list per namespace.

    Adequate for a few hundred near-duplicate entries per tenant and model.
    A vector index is the upgrade when cardinality outgrows a single key.
    """

    def __init__(self, client: KeyValueClient) -> None:
        self._client = client

    async def get_all(self, namespace: str) -> list[str]:
        raw = await self._client.get(_redis_key(namespace))
        if not raw:
            return []
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            return []
        return [item for item in parsed if isinstance(item, str)]

    async def put(
        self,
        namespace: str,
        entry_json: str,
        *,
        ttl_s: float,
        max_entries: int,
    ) -> None:
        current = await self.get_all(namespace)
        current.append(entry_json)
        trimmed = current[-max_entries:]
        await self._client.set(
            _redis_key(namespace),
            json.dumps(trimmed, separators=(",", ":")),
            ex=max(1, int(ttl_s)),
        )


def _redis_key(namespace: str) -> str:
    return f"llm-guard:semcache:{namespace}"


class AsyncRedisAdapter:
    """Thin adapter so the gateway depends on our protocol, not redis types."""

    def __init__(self, url: str) -> None:
        from redis.asyncio import Redis

        self._redis = Redis.from_url(url, decode_responses=True)

    async def get(self, key: str) -> str | None:
        value = await self._redis.get(key)
        if isinstance(value, str):
            return value
        return None

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        await self._redis.set(key, value, ex=ex)

    async def aclose(self) -> None:
        await self._redis.aclose()


@dataclass(frozen=True, slots=True)
class CacheHit:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    decision: str
    similarity: float


class SemanticCache:
    def __init__(
        self,
        store: CacheStore,
        embedder: Embedder,
        *,
        threshold: float,
        ttl_s: float,
        max_entries: int,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._store = store
        self._embedder = embedder
        self._threshold = threshold
        self._ttl_s = ttl_s
        self._max_entries = max_entries
        self._clock = clock or time.time

    async def lookup(self, namespace: str, text: str) -> CacheHit | None:
        query = self._embedder.embed(text)
        best: CacheHit | None = None
        best_score = self._threshold
        now = self._clock()
        for raw in await self._store.get_all(namespace):
            hit = _parse_entry(raw, query, now)
            if hit is None:
                continue
            if hit.similarity >= best_score:
                best_score = hit.similarity
                best = hit
        return best

    async def store(
        self,
        namespace: str,
        text: str,
        *,
        response_text: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        decision: str,
    ) -> None:
        payload = {
            "embedding": self._embedder.embed(text),
            "expires_at": self._clock() + self._ttl_s,
            "text": response_text,
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "decision": decision,
        }
        await self._store.put(
            namespace,
            json.dumps(payload, separators=(",", ":")),
            ttl_s=self._ttl_s,
            max_entries=self._max_entries,
        )


def _parse_entry(raw: str, query: Sequence[float], now: float) -> CacheHit | None:
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    expires_at = parsed.get("expires_at")
    embedding = parsed.get("embedding")
    if not isinstance(expires_at, (int, float)) or float(expires_at) <= now:
        return None
    if not isinstance(embedding, list) or not all(isinstance(item, (int, float)) for item in embedding):
        return None
    vector = [float(item) for item in embedding]
    similarity = cosine(query, vector)
    text = parsed.get("text")
    model = parsed.get("model")
    if not isinstance(text, str) or not isinstance(model, str):
        return None
    prompt_tokens = parsed.get("prompt_tokens")
    completion_tokens = parsed.get("completion_tokens")
    decision = parsed.get("decision")
    if not isinstance(prompt_tokens, int) or not isinstance(completion_tokens, int):
        return None
    if not isinstance(decision, str):
        return None
    return CacheHit(
        text=text,
        model=model,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        decision=decision,
        similarity=similarity,
    )
