"""Rate limiter, budget ledger, and retry jitter."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from llm_guard.budget import (
    BudgetExceededError,
    BudgetLedger,
    micro_to_usd,
    reservation_micro,
    usage_cost_micro,
    usd_to_micro,
)
from llm_guard.errors import BudgetExceededError as BudgetErrorAlias
from llm_guard.errors import RateLimitError
from llm_guard.ratelimit import RateLimiter, TokenBucket, compute_delay
from llm_guard.types import Usage


class Clock:
    def __init__(self, now: float = 0.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


def test_token_bucket_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError):
        TokenBucket(rate=1, capacity=-1)
    with pytest.raises(ValueError):
        TokenBucket(rate=-1, capacity=1)
    bucket = TokenBucket(rate=1, capacity=1)
    with pytest.raises(ValueError):
        bucket.try_consume(-1)


def test_token_bucket_refill_and_clock_rewind() -> None:
    clock = Clock()
    bucket = TokenBucket(rate=2.0, capacity=4.0, clock=clock)
    assert bucket.try_consume(4)
    assert bucket.retry_after(1) == pytest.approx(0.5)
    clock.now = -5
    assert bucket.level == 0
    clock.now = -4
    assert bucket.level == pytest.approx(2)
    clock.now = 100
    assert bucket.level == 4
    zero = TokenBucket(rate=0, capacity=1, clock=clock)
    assert zero.try_consume(1) is True
    assert zero.retry_after(1) == 60.0
    assert zero.retry_after(0) == 0.0


def test_rate_limiter_unknown_tenant_and_limit() -> None:
    limiter = RateLimiter(clock=Clock())
    with pytest.raises(KeyError):
        limiter.consume("missing")
    limiter.configure("t", requests_per_minute=0, burst=1)
    limiter.consume("t")
    with pytest.raises(RateLimitError) as exc:
        limiter.consume("t")
    assert exc.value.retry_after == 60.0


def test_compute_delay_bounds() -> None:
    with pytest.raises(ValueError):
        compute_delay(1, 0.1, 1.0, 1.1)
    assert compute_delay(3, 0.2, 0.5, 1.0) == 0.5
    assert compute_delay(1, -1.0, -0.5, 1.0) == 0.0
    assert compute_delay(2, 0.25, 10.0, 0.0) == 0.0


@given(
    rate=st.floats(min_value=0, max_value=100, allow_nan=False),
    capacity=st.floats(min_value=0, max_value=50, allow_nan=False),
    amounts=st.lists(st.floats(min_value=0, max_value=20, allow_nan=False), max_size=12),
    elapsed=st.floats(min_value=0, max_value=10, allow_nan=False),
)
@settings(max_examples=40)
def test_bucket_never_exceeds_capacity(rate: float, capacity: float, amounts: list[float], elapsed: float) -> None:
    clock = Clock()
    bucket = TokenBucket(rate=rate, capacity=capacity, clock=clock)
    for amount in amounts:
        bucket.try_consume(amount)
        assert 0 <= bucket.level <= capacity + 1e-9
        clock.now += elapsed
    assert 0 <= bucket.level <= capacity + 1e-9


def test_budget_reserve_commit_and_release() -> None:
    clock = Clock(1_700_000_000)
    ledger = BudgetLedger(clock=clock)
    assert ledger.spent_micro("acme") == 0
    assert ledger.inflight_micro("acme") == 0
    with pytest.raises(ValueError):
        ledger.reserve("acme", -1, cap_micro=10)
    reservation = ledger.reserve("acme", 4, cap_micro=10)
    assert ledger.inflight_micro("acme") == 4
    with pytest.raises(BudgetExceededError):
        ledger.reserve("acme", 7, cap_micro=10)
    with pytest.raises(ValueError):
        ledger.commit(reservation, -1)
    ledger.commit(reservation, 3)
    assert ledger.spent_micro("acme") == 3
    assert ledger.inflight_micro("acme") == 0
    extra = ledger.reserve("acme", 2, cap_micro=10)
    ledger.release(extra)
    ledger.release(extra)
    assert ledger.inflight_micro("acme") == 0
    clock.now += 86_400
    assert ledger.spent_micro("acme") == 0


def test_cost_helpers() -> None:
    usage = Usage(prompt_tokens=1_000_000, completion_tokens=2)
    assert usage.total_tokens == 1_000_002
    assert usage_cost_micro(usage, input_per_million=1.5, output_per_million=3) == 1_500_006
    assert (
        reservation_micro(
            prompt_tokens=10,
            reserved_output_tokens=0,
            input_per_million=1,
            output_per_million=1,
        )
        == 10
    )
    assert usd_to_micro(1.25) == 1_250_000
    assert micro_to_usd(2_000_000) == 2
    assert BudgetErrorAlias is BudgetExceededError
