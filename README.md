# llm-guard-platform

Production LLM gateway for teams that have to explain, in an audit, why a model call was allowed.

[![CI](https://github.com/Navneethsnehal/llm-guard-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/Navneethsnehal/llm-guard-platform/actions/workflows/ci.yml)
[![coverage](https://img.shields.io/badge/coverage-94.5%25-brightgreen)](https://github.com/Navneethsnehal/llm-guard-platform)
[![Python](https://img.shields.io/badge/python-3.11%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

An OpenAI-compatible front door with policy routing, guardrails, a regression red-team, an eval gate that can block a release, and an audit log that stores decisions rather than prompts. Built by Navneeth Jadav ([Navneethsnehal](https://github.com/Navneethsnehal)) as a flagship example of the control plane around RAG and tool-using agents in banking and insurance.

The default path uses a deterministic fake provider, so CI never calls OpenAI or Anthropic. Those providers are wired and fail closed when their API keys are absent.

## Why this exists

A regulated team cannot treat a chat completion as a private function call. The caller, the tenant, the policy pack, the spend, and the reason a prompt was blocked all have to be reconstructable later. This repo is that control plane:

- one API the existing OpenAI SDKs already speak
- per-tenant keys, budgets, and topic packs
- a written decision for every guardrail stage
- a release check that fails the build when quality, refusal, faithfulness, or cost regresses

## Architecture

```mermaid
flowchart LR
    Client[Client] --> API[FastAPI gateway]
    API --> Guards[Guardrails]
    API --> Cache[Semantic cache]
    API --> Router[Router and breakers]
    Router --> Models[Fake / OpenAI / Anthropic]
    API --> Audit[Audit log and metrics]
```

The request sequence and the component boundaries are in [docs/architecture.md](docs/architecture.md). Tradeoffs are recorded in [docs/adr](docs/adr).

## Features

- `POST /v1/chat/completions`, including SSE streaming
- Routes for cost, latency, and quality, with jittered retries and circuit breakers
- Semantic cache (hashed lexical embeddings, TTL, memory or Redis), isolated per tenant and model
- Token-bucket rate limits and daily budgets in integer micro-USD
- Guardrails: PII tokenization, injection heuristics plus a seeded local classifier, YAML topic packs, JSON schema checks, lexical grounding
- Red-team generator (jailbreaks, document injection, encoding, PII exfiltration, policy evasion) with attack success rate by category
- Golden-set eval gate with paired bootstrap confidence intervals
- OpenTelemetry spans, Prometheus metrics, structured logs, Grafana dashboard, append-only audit log
- Streamlit view of traffic, blocks, red-team results, and gate history

## Quickstart

```bash
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev,ui]"
make ci
llm-guard serve
```

`make ci` is what GitHub Actions runs: `ruff check`, `ruff format --check`, `mypy`, `pytest` (coverage gate 85%), a red-team smoke run, and the eval gate smoke run.

Docker Compose brings up the gateway, Redis, Prometheus, Grafana, and the UI:

```bash
docker compose up --build
```

| Service | URL |
| --- | --- |
| Gateway | http://127.0.0.1:8080 |
| Metrics | http://127.0.0.1:8080/metrics |
| Grafana | http://127.0.0.1:3000 |
| UI | http://127.0.0.1:8501 |

The demo tenant key below is local-only. See [SECURITY.md](SECURITY.md).

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer lg_test_regulated_demo_key_0001" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "quality",
    "temperature": 0,
    "messages": [
      {"role": "system", "content": "The platinum card APR is 2.9 percent for the first twelve months."},
      {"role": "user", "content": "What is the platinum card APR for the first twelve months?"}
    ]
  }'
```

Streaming is the same payload with `"stream": true`. The regulated demo tenant buffers the provider response, runs output guardrails, and only then emits SSE, so a block is still HTTP 400.

## Measured results

Numbers below are from this tree, in-process, against the fake provider. They are not hosted-model scores.

### Red team

`python -m llm_guard.cli redteam` — 28 cases. Attack success means the gateway returned the thing the attack asked for (canary, raw PII literal, or a substantive answer to a denied topic). Benign successes would be false positives.

| Category | Cases | Attack successes | Rate |
| --- | ---: | ---: | ---: |
| Jailbreak | 8 | 0 | 0.00 |
| Document injection | 4 | 0 | 0.00 |
| Encoding / obfuscation | 5 | 0 | 0.00 |
| PII exfiltration | 3 | 0 | 0.00 |
| Policy evasion | 4 | 0 | 0.00 |
| Benign (false positives) | 4 | 0 | 0.00 |
| **Overall ASR / FPR** | **28** | **0** | **0.00 / 0.00** |

This pack is a regression suite. A zero on these 28 cases is not a claim about every jailbreak in the wild. The holdout F1 of the injection classifier on its own synthetic split is 1.0 at a calibrated threshold of 0.48 (16 holdout templates, false-positive rate 0). The gateway operates at `max(0.50, calibrated)`.

### Eval gate

`python -m llm_guard.cli eval-gate` on `data/evals/golden.jsonl` (12 cases). Baseline and candidate were the same config, compared with a paired bootstrap of 1000 resamples, seed 42, alpha 0.05. Decision: **pass**.

| Metric | Baseline | Candidate | Blocks |
| --- | ---: | ---: | --- |
| Quality | 1.000 | 1.000 | no |
| Refusal correctness | 1.000 | 1.000 | no |
| Faithfulness | 0.972 | 0.972 | no |
| Mean cost (USD) | 0.000251 | 0.000251 | no |
| Latency p95 | 7.12 ms | 2.85 ms | no |

Faithfulness is lexical overlap with the supplied context and is omitted when the case was refused. The latency interval includes zero (ci low -9.51 ms, ci high 0.15 ms). The second pass was warmer. The gate blocks latency only when p95 regresses by more than `max(50ms, 50% of baseline)` and the bootstrap interval lies above zero, so this run does not block and would not have been reported as a speedup the gate requires.

### Gateway overhead

`python -m llm_guard.cli bench --requests 400 --concurrency 20` against the in-process ASGI app, model route `latency`, fake provider, semantic cache disabled. The bench command raises the demo tenant's token bucket for the measurement so the sample is gateway work rather than HTTP 429. All 400 requests returned 200.

| | |
| --- | ---: |
| Success | 400 / 400 |
| Elapsed | 0.848 s |
| Throughput | 471.7 req/s |
| p50 | 37.84 ms |
| p95 | 52.74 ms |
| p99 | 55.44 ms |

A Locust profile for the same payload is in `benchmarks/locustfile.py`. The README figures come from the async bench, which CI can run without Locust.

### Tests

43 tests (unit, ASGI integration, Hypothesis properties on the token bucket and email redaction). Coverage.py total, branch coverage on: **94.5%**. The gate in `pyproject.toml` fails under 85%.

## Project structure

```
src/llm_guard/          gateway, providers, guardrails, eval, red team, CLI
configs/                gateway.yaml and policy packs
data/evals/             golden set
data/models/            injection classifier weights
data/datasets/          seeded train/holdout jsonl
tests/                  unit, property, and integration tests
ui/                     Streamlit operator view
deploy/                 Prometheus scrape and Grafana dashboard
docs/architecture.md    sequence and component diagrams
docs/adr/               tradeoffs
docs/threat-model.md
benchmarks/locustfile.py
```

## Design decisions

- **OpenAI facade.** Callers do not take a new client. Unsupported fields (`n != 1`, tools) are rejected. Diagnostics stay in headers unless explicitly enabled. [ADR 0001](docs/adr/0001-openai-compatible-facade.md)
- **Stages with reasons.** Injection, topics, PII, schema, and grounding each record a decision. System text is not scanned for topics, because the operator wrote it and it often quotes the policy. [ADR 0002](docs/adr/0002-composable-guardrails.md)
- **Tenant-scoped cache.** A hit never crosses tenants. Embeddings are local hashes so CI stays offline. [ADR 0003](docs/adr/0003-tenant-scoped-cache.md)
- **Retries inside the breaker.** One incident is one failure. Fallback happens after the retry budget, and only before the first streamed byte. [ADR 0004](docs/adr/0004-retries-inside-the-breaker.md)
- **Buffer, then stream.** The regulated default finishes output guardrails before the first SSE byte. [ADR 0005](docs/adr/0005-buffer-then-stream.md)
- **Bootstrap gate.** Means alone flap on a small golden set. The gate needs a confidence interval and a minimum effect. [ADR 0006](docs/adr/0006-bootstrap-release-gate.md)
- **Local classifier.** Weights are JSON in the repo, training is seeded, and the red-team ASR is the number that matters. [ADR 0007](docs/adr/0007-local-injection-classifier.md)

Budgets are integer micro-USD. A request reserves the most expensive model on its route, commits the actual cost, and releases the hold on failure. Cache hits commit zero.

## Configuration

Demo keys (plaintext only here and in tests; YAML stores SHA-256):

| Key | Tenant | Notes |
| --- | --- | --- |
| `lg_test_regulated_demo_key_0001` | regulated-demo | Buffered streaming, $25/day, 600 rpm, burst 60 |
| `lg_test_latency_demo_key_0001` | latency-demo | Incremental streaming, $50/day |
| `lg_admin_local_dev` | admin | Local admin API only |

`LLM_GUARD_ADMIN_KEY`, `REDIS_URL`, and `LLM_GUARD_CACHE=memory|redis` override the YAML. Real providers read `OPENAI_API_KEY` and `ANTHROPIC_API_KEY` and are not on the default routes.

## Roadmap

- Tool and function-call passthrough with the same output checks on tool arguments
- A vector index behind the same cache interface once a tenant outgrows one Redis list
- Optional OTLP export wired from environment variables, still off by default
- Golden cases that call a hosted model in a manual workflow, kept out of the offline CI job
- A larger red-team corpus with a published ASR trend, still scored as a regression pack rather than a universe estimate

## License

MIT. Copyright 2026 Navneeth Jadav.
