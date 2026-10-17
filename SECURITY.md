# Security

## Reporting

Report vulnerabilities privately to the repository owner, Navneeth Jadav (GitHub: Navneethsnehal). Please include a request transcript, the tenant, and the guardrail decision if one was returned. Do not open a public issue that contains a working bypass or customer data.

## Demo credentials

These keys are hashed in `configs/gateway.yaml` and are for local runs and CI only:

| Key | Tenant | Purpose |
| --- | --- | --- |
| `lg_test_regulated_demo_key_0001` | regulated-demo | Buffered streaming, $25 daily cap |
| `lg_test_latency_demo_key_0001` | latency-demo | Incremental streaming |
| `lg_admin_local_dev` | admin | `/admin/snapshot` and `/admin/audit` |

Set `LLM_GUARD_ADMIN_KEY` in any shared environment. Do not reuse the demo admin key. Rotate a tenant by replacing the SHA-256 in the tenant's `api_keys` list. Plaintext keys are not stored in the config.

## Operational expectations

- Terminate TLS in front of the gateway. The process speaks HTTP.
- Do not expose `/metrics` or the admin routes on a public ingress. `/metrics` has no credential because scrapers on a private network should not need the admin key, and it does not contain prompts.
- Provider keys are read from the environment named in the config (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`). They are not written to logs.
- The audit log records the SHA-256 of the rendered prompt, the stage, the decision, and reason codes. It does not record the prompt or the completion. Treat the file as compliance evidence and restrict its filesystem permissions.
- The semantic cache stores completion text. Use the Redis backend with network policy that matches the sensitivity of those completions, or keep the memory backend on a single host.
- The injection classifier and the red-team pack are regression controls. A green CI run means this suite was blocked, not that every future prompt will be.

## Threat model

See [docs/threat-model.md](docs/threat-model.md).
