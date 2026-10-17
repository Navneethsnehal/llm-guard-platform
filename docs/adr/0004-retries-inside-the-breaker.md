# ADR 0004: Retries inside the circuit breaker

## Status

Accepted

## Context

A single 429 should not open a breaker, and a breaker that opens on the first failure will fail over so eagerly that the primary model never gets a retry. Opening after every attempt, on the other hand, counts one incident as several.

## Decision

`call_with_retry` is the unit the breaker observes. Retryable provider errors sleep with full jitter (`uniform(0, min(cap, base * 2^attempt))`) and try again up to `attempts`. A non-retryable error, or the last attempt, is one failure. The next candidate is used only after that. Half-open allows a single trial after `recovery_seconds`.

Streaming may fall back only before the first event. Once a byte is available, the selected model's breaker is settled when the generator finishes.

## Consequences

Timeouts use the same retry budget as HTTP 429 and 5xx. The demo config uses a short base delay so CI does not wait on the failure path. Production configs should raise `base_delay_seconds` and `max_delay_seconds` with the upstream's retry advice.
