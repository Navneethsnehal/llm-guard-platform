# ADR 0001: OpenAI-compatible facade

## Status

Accepted

## Context

Bank and insurance callers already speak the chat-completions API through existing SDKs. A private schema would force every application to take a new client before any control could be inserted.

## Decision

`POST /v1/chat/completions` accepts the fields those SDKs send: `model`, `messages`, `temperature`, `max_tokens`, `stream`, `response_format`, and `stream_options`. `n` other than 1 and `tools` are rejected. Unknown extra fields are ignored so SDK drift does not become an outage. Streaming uses the same SSE chunk shape, including `data: [DONE]`.

Diagnostics are not part of the OpenAI body unless `server.expose_diagnostics` is on. Cost, cache, model, and decision are always response headers, because eval and audit need them even when the body must stay compatible.

## Consequences

Tool-calling applications cannot use this gateway until a later version understands tools. The compatibility surface stays small enough to test against the fake provider in CI with no network.
