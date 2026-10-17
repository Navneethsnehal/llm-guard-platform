"""Token-bucket rate limiter with an injectable clock."""

import threading
import time
from collections.abc import Callable

from llm_guard.errors import RateLimitError


class TokenBucket:
    """Classic token bucket.

    ``rate`` is tokens added per second and ``capacity`` is the burst size.
    ``try_consume`` never drives the level negative and never above capacity.
    """

    def __init__(
        self,
        *,
        rate: float,
        capacity: float,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if capacity < 0:
            raise ValueError("capacity must be non-negative")
        if rate < 0:
            raise ValueError("rate must be non-negative")
        self.rate = rate
        self.capacity = capacity
        self._clock = clock or time.monotonic
        self._tokens = capacity
        self._updated = self._clock()
        self._lock = threading.Lock()

    @property
    def level(self) -> float:
        with self._lock:
            self._refill()
            return self._tokens

    def _refill(self) -> None:
        now = self._clock()
        elapsed = now - self._updated
        if elapsed < 0:
            elapsed = 0.0
        self._updated = now
        if self.rate == 0 or elapsed == 0:
            return
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)

    def try_consume(self, amount: float = 1.0) -> bool:
        if amount < 0:
            raise ValueError("amount must be non-negative")
        with self._lock:
            self._refill()
            if amount > self.capacity or self._tokens < amount:
                return False
            self._tokens -= amount
            return True

    def retry_after(self, amount: float = 1.0) -> float:
        with self._lock:
            self._refill()
            deficit = amount - self._tokens
            if deficit <= 0:
                return 0.0
            if self.rate <= 0:
                return 60.0
            return deficit / self.rate


def compute_delay(attempt: int, base_delay: float, max_delay: float, unit_random: float) -> float:
    """Full jitter: uniform(0, min(max_delay, base_delay * 2**attempt)).

    ``unit_random`` is in ``[0, 1]`` so tests can pin the draw.
    """

    if not 0.0 <= unit_random <= 1.0:
        raise ValueError("unit_random must be within [0, 1]")
    ceiling = min(max_delay, base_delay * (2.0**attempt))
    if ceiling < 0:
        return 0.0
    return unit_random * ceiling


class RateLimiter:
    """One bucket per tenant. Request cost is one bucket token."""

    def __init__(self, clock: Callable[[], float] | None = None) -> None:
        self._clock = clock
        self._buckets: dict[str, TokenBucket] = {}
        self._lock = threading.Lock()

    def configure(self, tenant_id: str, *, requests_per_minute: float, burst: float) -> None:
        rate = requests_per_minute / 60.0
        with self._lock:
            self._buckets[tenant_id] = TokenBucket(rate=rate, capacity=burst, clock=self._clock)

    def bucket(self, tenant_id: str) -> TokenBucket:
        with self._lock:
            try:
                return self._buckets[tenant_id]
            except KeyError as exc:
                raise KeyError(f"no rate limit configured for {tenant_id}") from exc

    def consume(self, tenant_id: str, amount: float = 1.0) -> None:
        bucket = self.bucket(tenant_id)
        if bucket.try_consume(amount):
            return
        retry = bucket.retry_after(amount)
        raise RateLimitError(
            f"Rate limit exceeded for tenant '{tenant_id}'",
            retry_after=retry,
        )
