"""Per-tenant daily spend caps, accounted in integer micro-USD."""

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from llm_guard.errors import BudgetExceededError
from llm_guard.types import Usage

MICRO = 1_000_000


def usd_to_micro(amount: float) -> int:
    return round(amount * MICRO)


def micro_to_usd(amount: int) -> float:
    return amount / MICRO


def usage_cost_micro(usage: Usage, *, input_per_million: float, output_per_million: float) -> int:
    """Exact integer cost: tokens * USD-per-million == micro-USD."""

    input_cost = round(usage.prompt_tokens * input_per_million)
    output_cost = round(usage.completion_tokens * output_per_million)
    return input_cost + output_cost


def reservation_micro(
    *,
    prompt_tokens: int,
    reserved_output_tokens: int,
    input_per_million: float,
    output_per_million: float,
) -> int:
    usage = Usage(prompt_tokens=prompt_tokens, completion_tokens=reserved_output_tokens)
    return usage_cost_micro(
        usage,
        input_per_million=input_per_million,
        output_per_million=output_per_million,
    )


@dataclass(frozen=True, slots=True)
class Reservation:
    tenant_id: str
    day: str
    amount_micro: int


class BudgetLedger:
    """Daily cap with in-flight reservations.

    A request reserves an upper bound before the provider call and commits the
    actual cost afterwards. Reservations count against the cap so concurrent
    requests cannot all sneak under a nearly exhausted budget.
    """

    def __init__(self, clock: Callable[[], float] | None = None) -> None:
        self._clock = clock or time.time
        self._spent: dict[tuple[str, str], int] = {}
        self._inflight: dict[tuple[str, str], int] = {}
        self._lock = threading.Lock()

    def _key(self, tenant_id: str) -> tuple[str, str]:
        stamp = time.gmtime(self._clock())
        return (tenant_id, time.strftime("%Y-%m-%d", stamp))

    def spent_micro(self, tenant_id: str) -> int:
        key = self._key(tenant_id)
        with self._lock:
            return self._spent.get(key, 0)

    def inflight_micro(self, tenant_id: str) -> int:
        key = self._key(tenant_id)
        with self._lock:
            return self._inflight.get(key, 0)

    def reserve(self, tenant_id: str, amount_micro: int, *, cap_micro: int) -> Reservation:
        if amount_micro < 0:
            raise ValueError("reservation must be non-negative")
        key = self._key(tenant_id)
        with self._lock:
            spent = self._spent.get(key, 0)
            inflight = self._inflight.get(key, 0)
            if spent + inflight + amount_micro > cap_micro:
                raise BudgetExceededError(
                    f"Daily budget exceeded for tenant '{tenant_id}' "
                    f"(spent ${micro_to_usd(spent):.6f}, "
                    f"in flight ${micro_to_usd(inflight):.6f}, "
                    f"cap ${micro_to_usd(cap_micro):.6f})"
                )
            self._inflight[key] = inflight + amount_micro
        return Reservation(tenant_id=tenant_id, day=key[1], amount_micro=amount_micro)

    def commit(self, reservation: Reservation, actual_micro: int) -> None:
        if actual_micro < 0:
            raise ValueError("actual cost must be non-negative")
        key = (reservation.tenant_id, reservation.day)
        with self._lock:
            inflight = self._inflight.get(key, 0)
            self._inflight[key] = max(0, inflight - reservation.amount_micro)
            self._spent[key] = self._spent.get(key, 0) + actual_micro

    def release(self, reservation: Reservation) -> None:
        key = (reservation.tenant_id, reservation.day)
        with self._lock:
            inflight = self._inflight.get(key, 0)
            self._inflight[key] = max(0, inflight - reservation.amount_micro)
