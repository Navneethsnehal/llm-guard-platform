# Engineering notes

Review of this tree against the code and the ADRs in `docs/adr`. Nothing here is a claim about a deployed system. Live OpenAI, live Anthropic, a real Redis server, Docker Compose, Prometheus, Grafana, Streamlit, and Locust are not what CI runs.

## Files

| Path | What it is |
| --- | --- |
| `src/llm_guard/__init__.py` | Package version (`0.1.0`). |
| `src/llm_guard/__main__.py` | `python -m llm_guard` entry. |
| `src/llm_guard/api/app.py` | FastAPI app: `/health`, `/v1/models`, `/v1/chat/completions`, `/metrics`, admin snapshot and audit. Rejects `n != 1` and `tools`. |
| `src/llm_guard/api/schemas.py` | Pydantic models for the chat-completions subset the gateway accepts. |
| `src/llm_guard/auth.py` | Loads tenants and compares API-key SHA-256 digests. |
| `src/llm_guard/gateway.py` | Orders auth limits, guardrails, cache, routing, charging, and the audit write. |
| `src/llm_guard/runtime.py` | Builds the gateway from settings: providers, breakers, cache backend, injection threshold. |
| `src/llm_guard/config.py` | Parses `configs/gateway.yaml` and policy packs. |
| `src/llm_guard/types.py` | Messages, usage, requests, results, and guard traces. Token estimates are one token per four characters, minimum one for a non-empty string. |
| `src/llm_guard/errors.py` | Gateway errors and the HTTP status each one maps to. `ProviderError` is not a client-facing error. |
| `src/llm_guard/routing.py` | Route expansion, full-jitter retries, circuit breaker, and stream fallback before the first event. |
| `src/llm_guard/providers.py` | `FakeProvider` (deterministic, offline), `OpenAIProvider`, `AnthropicProvider`. |
| `src/llm_guard/cache.py` | Hashed lexical embeddings, cosine lookup, memory store, Redis list store. |
| `src/llm_guard/budget.py` | Per-tenant daily cap in integer micro-USD: reserve, commit, release. |
| `src/llm_guard/ratelimit.py` | In-process token bucket. |
| `src/llm_guard/sse.py` | SSE framing for chat-completion chunks. |
| `src/llm_guard/guardrails/pipeline.py` | Stage order and the incremental partial scan. |
| `src/llm_guard/guardrails/injection.py` | High-precision regexes plus the local classifier, scored on normalized variants. |
| `src/llm_guard/guardrails/normalize.py` | NFKC, homoglyphs, zero-width removal, leetspeak, base64, rot13. |
| `src/llm_guard/guardrails/classifier.py` | Hashed n-gram logistic regression. Weights load from JSON. |
| `src/llm_guard/guardrails/train.py` | Seeded synthetic set and `build_model()`. |
| `src/llm_guard/guardrails/topics.py` | Compiles deny patterns from a policy pack. System text is not passed in. |
| `src/llm_guard/guardrails/pii.py` | Email, phone, SSN, card (Luhn), and account-number detection. Placeholders are `[[PII:kind:n]]`. |
| `src/llm_guard/guardrails/schema_check.py` | A small JSON Schema subset for model output. |
| `src/llm_guard/guardrails/grounding.py` | Lexical overlap and number membership against the supplied context. |
| `src/llm_guard/observability.py` | Prometheus metrics, JSON logs, the audit log, and a process-wide tracer provider. |
| `src/llm_guard/evals.py` | Golden-set scoring and the paired-bootstrap release gate. |
| `src/llm_guard/redteam.py` | The 28-case regression pack and its success rules. |
| `src/llm_guard/bench.py` | In-process async benchmark. |
| `src/llm_guard/cli.py` | `redteam`, `eval-gate`, `bench`, `train-classifier`, `serve`. |
| `src/llm_guard/dashboard.py` | Pure functions the Streamlit app renders. Tested without a browser. |
| `ui/app.py` | Streamlit page over `/admin/snapshot` and `var/reports/*.json`. |
| `configs/gateway.yaml` | Server, cache, retry, breaker, fake models, two live-provider models, routes, guardrails, two tenants. |
| `configs/policies/regulated.yaml` | Topic pack: weapons, insider trading, crime assistance. |
| `data/evals/golden.jsonl` | 12 golden cases. |
| `data/models/injection_classifier.json` | Checked-in weights and the stored holdout block. |
| `data/datasets/injection.jsonl` | Seeded train/holdout lines. |
| `tests/conftest.py` | In-process app fixture. |
| `tests/unit/` | Guardrails, cache, routing, providers, budget, eval, red team, CLI. |
| `tests/integration/test_api.py` | ASGI tests for the HTTP contract. |
| `benchmarks/locustfile.py` | Optional external load profile. Not used by `make ci`. |
| `deploy/prometheus/prometheus.yml` | Scrape config for Compose. |
| `deploy/grafana/` | Datasource and dashboard provisioning for Compose. |
| `docker-compose.yml` | Gateway, Redis, Prometheus, Grafana, Streamlit. Not started by tests. |
| `Dockerfile` | Python 3.12 image. Installs `.[ui]` and runs `llm-guard serve`. |
| `Makefile` | `ci` is lint, mypy, pytest, red-team smoke, eval-gate smoke. |
| `.github/workflows/ci.yml` | Ubuntu, Python 3.12, `pip install -e ".[dev]"`, `make ci`. |
| `docs/architecture.md` | Component and sequence diagrams, HTTP status table. |
| `docs/adr/0001`–`0007` | Decisions recorded with the code. |
| `docs/threat-model.md` | Assets, boundaries, and what the red team is not. |
| `SECURITY.md` | Demo keys and operational notes. |

## Decisions and tradeoffs

These match the ADRs and the current code.

**OpenAI-shaped API (ADR 0001).** Callers post chat completions. Diagnostics go in response headers when the config exposes them. The schema does not try to cover the whole OpenAI surface: tools and `n != 1` are hard errors.

**Staged guardrails (ADR 0002).** Input order is injection on the full transcript, topics on non-system messages, then PII per message. Output order is PII, injection, topics, schema, grounding, then vault render. The first block returns. Placeholders are rendered only after the later stages, so a reversible email is not restored into a response that is about to be blocked. Topic rules are YAML. System text is skipped for topics because the operator wrote it and it often quotes the policy. The cost is that a system message is still scanned for injection, and a topic violation hidden only in the system role is not a topic block.

**Cache namespace (ADR 0003).** A hit requires cosine similarity at or above the configured threshold (0.86 in the YAML), a live TTL, `temperature == 0`, and caching left on. Embeddings are local hashes so CI never calls an embedding API. Paraphrases that share no tokens miss. A Redis list will not scale to a large distinct-prompt set; the interface is the same either way. The namespace is `tenant:model`, plus a SHA-256 of input replacements when the vault is non-empty. The fingerprint is taken after input redaction and before output redaction, so the lookup key and the store key see the same vault. Incremental streaming sets the cache result to `bypass`. Buffered streaming calls `execute` with `stream=False`, so a buffered stream can hit.

**Retries inside the breaker (ADR 0004).** `call_with_retry` is one outcome. Retryable errors sleep `uniform(0, min(cap, base * 2^attempt))`. The YAML uses a short delay so CI does not wait. Half-open allows one trial after `recovery_seconds`. An empty stream is a retryable `ProviderError`, then the next candidate. After the first event the caller settles the breaker: success if the generator finishes or the client disconnects, failure if a provider exception escapes while the rest is read. Any other exception after `allow()` records a failure so `_half_open_inflight` cannot stay true.

**Buffer, then stream (ADR 0005).** Default `stream_mode` is `buffered`. A block is HTTP 400 and no model text. Incremental mode scans each partial for injection, topics, and raw PII, stops with `finish_reason=content_filter`, and records grounding after the flush because those bytes have already left. `scan_partial` blocks on any detected PII, including when the tenant mode is `redact`. That is the fail-closed choice in the ADR, not an accident. Incremental content-filter stops call `_charge`. Buffered output blocks raise before `_charge`, and `_finish` releases the reservation.

**Bootstrap gate (ADR 0006).** `GateConfig.default()` uses 1000 resamples, seed 42, alpha 0.05. Quality and faithfulness block on a drop past 0.02 when the interval says so. Refusal correctness blocks on any drop. Cost blocks on an increase past `1e-9`. Latency blocks only when the p95 interval lies above zero and the p95 regression exceeds `max(50 ms, 50% of baseline)`. Comparing a config to itself still produces a latency delta, because the two passes are timed separately.

**Local classifier (ADR 0007).** Heuristics block on their own. Otherwise the model blocks at `max(YAML threshold, holdout threshold)`, so the YAML cannot undercut the calibration. `build_model()` on this tree returned threshold 0.48, holdout n 16, precision 1.0, recall 1.0, F1 1.0, false-positive rate 0.0, seed 42, dimension 256. The YAML threshold is 0.50, so the operating threshold is 0.50. The holdout is the seeded split, not a sample of production traffic. The red-team attack success rate is the other regression number, and it is also only the cases in `redteam.py`.

**Money.** Prices are USD per million tokens. `usage_cost_micro` rounds token count times that price into integer micro-USD. Reservations use the max input and output price on the route and `max(request max_tokens, reservation_output_tokens)`. Cache hits commit 0. The ledger and the rate limiter are in-process. Two processes do not share them.

**Providers.** `FakeProvider` synthesizes text from the prompt: faithful overlap, echo, JSON, an ungrounded sentence, a leak behavior, or injected failures. OpenAI streaming sends `stream_options.include_usage` and keeps a usage object even when `choices` is empty. Anthropic `stream()` calls `complete()` and yields one event. Missing API keys raise a non-retryable `ProviderError`.

## Bugs fixed

Each fix has a regression test. Behavior elsewhere was left as it was.

1. **Redacted PII shared a cache entry.** Two prompts that differ only by email redact to the same `[[PII:email:1]]` and the same embedding. A later caller could receive the earlier caller's restored address. The cache namespace now includes `PiiVault.fingerprint()`, a SHA-256 of token, kind, and value, snapshotted after input guardrails. Test: `test_cache_does_not_reuse_another_callers_pii`.

2. **A cache cap of zero stored every row.** `list[-0:]` is the whole list. `_trim` returns an empty list when `max_entries <= 0`. Test: `test_cache_cap_of_zero_stores_nothing`.

3. **Redis kept expired rows and raised on bad JSON.** `RedisStore.put` appended without dropping `expires_at`, so dead rows occupied `max_entries`. `json.loads` of a corrupt value failed the lookup. `get_all` now returns `[]` for non-JSON or a non-list, and `put` keeps only live entries before trimming. The lock is one process; the docstring says so. Test: `test_redis_drops_expired_entries_and_bad_json`.

4. **An abstention sentence grounded the whole answer.** Any abstention phrase returned score 1.0 and `("abstained",)`, including when a later sentence was unsupported. Abstention phrases are stripped per sentence. A sentence with nothing left is skipped. Other sentences are scored. A fully abstaining answer is still `("abstained",)`. Test: `test_grounding_scores` (mixed answer and a claim that only starts with "I don't know").

5. **An empty stream was a success and skipped fallback.** The router recorded success and then raised `UpstreamError`, so the next candidate never ran. An empty iterator is now a retryable `ProviderError`, which stays inside one breaker outcome and then falls through. Test: `test_empty_stream_falls_back_and_counts_as_one_failure`.

6. **A non-provider exception could stick half-open.** `allow()` set the in-flight flag, and only `ProviderError` cleared it. A `RuntimeError` left the breaker half-open until process restart. The `finally` records one failure when the attempt did not settle. Test: `test_unexpected_error_does_not_stick_half_open`.

7. **Closing an incremental stream counted as a provider failure.** The upstream generator's `finally` runs after the gateway's `aclose()` returns, so the gateway recorded a failure and then the provider cleanup ran. The gateway now settles success when an event was already produced, and the request status is `cancelled` on `GeneratorExit` or `CancelledError`. Test: `test_abandoned_incremental_stream_is_not_a_provider_failure`.

8. **OpenAI streaming dropped usage.** Usage-only SSE chunks were ignored, and the request did not set `stream_options.include_usage`, so incremental cost fell back to the character estimate. The payload requests usage, and a chunk with empty `choices` still returns a `ProviderEvent` when usage parses. Test: `test_openai_stream_keeps_usage_and_requests_it`.

9. **Policy-evasion success used the HTTP body length.** Any body longer than 80 characters counted, so a JSON envelope around `"No."` was a success. The check uses assistant `content` when the body is a chat completion, and still uses the raw body when it is not. Test: `test_attack_scoring_and_catalog`.

## Tests added

The parent revision collected 43 tests. This tree collects 55. Twelve are new functions. A few existing tests gained assertions for the same bugs or for a boundary that the new code made worth pinning.

| Test | Why it is there |
| --- | --- |
| `test_cache_does_not_reuse_another_callers_pii` | The two-email cache leak, through ASGI, including a third call that still hits the first caller's entry. |
| `test_cache_cap_of_zero_stores_nothing` | `max_entries=0` on both stores. |
| `test_redis_drops_expired_entries_and_bad_json` | TTL filter on put, and a `{` value does not raise. |
| `test_empty_stream_falls_back_and_counts_as_one_failure` | Empty stream retries once, records one failure, and opens the next model. |
| `test_stream_timeout_falls_back` | A slow first event falls through to the next candidate. |
| `test_unexpected_error_does_not_stick_half_open` | `RuntimeError` opens the breaker and a later trial is allowed after recovery. |
| `test_retries_are_one_breaker_outcome` | Two injected failures inside `attempts=3` are not two breaker failures. |
| `test_abandoned_incremental_stream_is_not_a_provider_failure` | Client `aclose` leaves the breaker closed and the metric status `cancelled`. |
| `test_blocked_output_releases_the_reservation` | Ungrounded buffered output is 400, spend stays 0, and a later call can still reserve. |
| `test_direct_model_uses_the_rest_of_its_route` | `fake-unstable` fails over to `fake-fast` on the same route. |
| `test_payload_limit_boundary` | Body length equal to `max_body_chars` is 200; one character over is 413. |
| `test_openai_stream_keeps_usage_and_requests_it` | Usage-only chunk is kept, `include_usage` is sent, malformed usage raises. |

Assertions added inside existing tests: PII fingerprints differ when the values differ; a bool is `type_mismatch:$.n` in the schema checker; a token-bucket consume larger than the level does not drain it; a budget reserve of exactly the cap commits and the next unit fails; dashboard `cache_rows` and `circuit_rows` follow the snapshot, and a `content_filter` request status is not counted as `guardrail_blocked`.

`python -m pytest` on this tree: 55 passed, coverage **95.20%** with branch coverage, fail-under 85.

## What changed without a behavior change

- Removed `MutableTraces`. Nothing constructed it. Traces are a list on the prepared request.
- `ui/app.py` renders `dashboard.cache_rows` instead of a second copy of that loop.
- The dashboard headline counts status `guardrail_blocked`. That is the status the gateway records. `content_filter` is the OpenAI error type and the incremental `finish_reason`, not a request counter.
- Package text no longer says "production". The budget helper's docstring says the cost is rounded.
- Small helpers where a branch was hard to follow: `_trim`, `_decode_entries`, `_entry_is_live`, `_claim_text`, `_cache_namespace`, `_upstream_request`, `_assistant_content`, `_optional_usage`, `_delta_text`.

## Not done

These are left as they are. They are visible in the code and are not covered by a test that would fail if they changed.

- **Real Redis, Compose, and the operator UI.** CI never starts them. `AsyncRedisAdapter` is only lightly touched. Prometheus, Grafana, and Streamlit have no browser or scrape test.
- **Live providers.** OpenAI and Anthropic are tested with `httpx.MockTransport`. No test uses a network key. Anthropic streaming is one buffered completion.
- **Redis races.** The store lock is process-local. Two processes can lose an update on the same list.
- **`FakeProvider._fail_remaining`.** The counter is not locked. Concurrent calls can under-count injected failures.
- **Audit append.** `AuditLog.write` holds its lock across the file append.
- **Shared process state.** The budget ledger, rate limiter, and tracer provider are in-process. `configure_tracing` is global.
- **Incremental PII.** Partials always block, including under `pii.mode: redact` (ADR 0005). A test could pin a redacted email on the latency tenant if that choice is ever revisited.
- **Incremental charging.** A content-filter stop commits provider usage. A buffered block does not. That split is untested as a stated policy beyond the buffered release test.
- **Chunks after the first event.** Only the first event is wrapped in the router's `wait_for`. Later chunks depend on the provider's own timeout.
- **Grounding.** Overlap will not catch a paraphrase. Numbers must appear as the same digit string. Abstention phrases are a fixed list.
- **Classifier and red team.** Both are synthetic. There is no test that a new attack family outside the templates is caught, because the design does not claim that.
- **Policy scoring.** `_substantive` still treats the raw body as a refusal when it contains the substrings `content_filter` or `blocked by guardrail`, before it parses assistant text.
- **`_parse_entry`.** A boolean `expires_at` passes `isinstance(..., (int, float))` because `bool` is an `int`. The Redis live check rejects bools. The two paths disagree on that value.
- **PII token ids.** `[[PII:kind:n]]` is a per-request counter. It is not a secret.
- **Locust.** `benchmarks/locustfile.py` is not part of `make ci`. The benchmark numbers in the README come from `llm-guard bench`.
- **OTLP.** Spans are created. Nothing in the repo exports them.
- **Tools.** Rejected in the API. There is no passthrough.
