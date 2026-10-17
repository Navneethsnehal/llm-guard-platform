# ADR 0005: Buffer output guardrails before streaming

## Status

Accepted

## Context

Incremental SSE cannot take back a sentence the client has already rendered. A PII leak or a policy violation in the first chunk is already an incident if the only control runs at the end.

## Decision

The default `stream_mode` is `buffered`. The gateway executes the full non-streaming path, including output guardrails, and only then writes SSE frames. A block is still HTTP 400 with no model text. Tenants that need time-to-first-token can set `incremental`. That mode scans each partial for injection, topics, and raw PII, stops with `finish_reason=content_filter` when one fires, and records grounding as a warning after the flush because the bytes have already left.

## Consequences

Buffered streaming adds the provider latency before the first byte. That is the point. Incremental mode is a conscious weakening and is not the demo default for the regulated tenant.
