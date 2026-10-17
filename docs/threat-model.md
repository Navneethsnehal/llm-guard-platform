# Threat model

This document scopes the gateway as a control in front of a model, not as a guarantee that the model is safe. The red-team pack is a regression suite. It is not an estimate of jailbreak prevalence.

## Assets

- Tenant API keys and the local admin key.
- Prompts and completions that may contain customer PII or confidential product rules.
- Daily spend caps and per-tenant rate limits.
- The audit log used to show what the control decided.
- Provider credentials (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`).

## Trust boundaries

1. The caller is authenticated only by a bearer key. Possession of the key is the tenant.
2. System messages are treated as operator-supplied. Topic rules do not scan them, because a policy pack quoted in the system prompt would otherwise block legitimate questions about that policy.
3. User and assistant messages, and any document text the caller pastes into them, are untrusted.
4. Provider responses are untrusted. Output stages run before the body is returned on the buffered path.
5. `/metrics` is unauthenticated. It is meant for a private network. It does not include prompt text.
6. `/admin/snapshot` and `/admin/audit` require the admin key. The admin key in the demo config is for local development.

## Adversaries

| Adversary | Capability | What the gateway does |
| --- | --- | --- |
| Tenant user | Sends jailbreaks, encoded instructions, documents that override the prompt, or asks for disallowed topics | Input heuristics, normalized views (homoglyphs, zero-width, leetspeak, base64, rot13), local classifier, topic pack |
| Tenant user | Embeds PII and asks the model to echo it | Detection and tokenization before the provider call. Reversible types are restored only on the way out. Other types stay redacted |
| Compromised or weak model | Emits secrets, ungrounded claims, or non-conforming JSON | Output injection, topics, schema check, lexical grounding against the system context |
| Noisy upstream | Timeouts, 429s, 5xx | Jittered retries inside the candidate, then the next model. The breaker opens only after the retry budget is exhausted |
| Curious operator or log scraper | Reads logs and the audit file | Structured logs and the audit file store a SHA-256 of the prompt, stage, decision, and reason codes |

## Out of scope

- Training-time data poisoning of a hosted model.
- A jailbreak that uses a novel phrasing the heuristics and the small classifier both miss. That is why the suite is re-run in CI and why a failed case blocks the red-team command.
- Cross-tenant cache leakage. Namespaces include tenant id and requested model. Do not point two tenants at one shared store key prefix by editing `_redis_key` without keeping that prefix.
- TLS termination, key management, and network policy. Run the process behind the platform's ingress. Do not publish the admin key or the demo tenant keys.

## Residual risk

Lexical grounding misses paraphrases that are still unsupported, and it can block a true sentence that shares too few content words with the context. The injection classifier is trained on a seeded synthetic set. Holdout F1 on that generator is not the end-to-end number. The end-to-end number is the red-team attack success rate.

Reason strings are codes (`heuristic:ignore_previous_instructions`, `ungrounded_sentence_count:N`, `topic:weapons`). They are not copies of the prompt, so a blocked PII value is not written into the audit log.
