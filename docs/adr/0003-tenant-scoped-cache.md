# ADR 0003: Tenant-scoped semantic cache

## Status

Accepted

## Context

Repeated policy questions dominate cost. A shared cache would also let one tenant retrieve another tenant's completion, which is unacceptable for account data that survived redaction or for tenant-specific policy text.

## Decision

Cache keys are namespaced by tenant id and requested model. When input guardrails replace PII, the namespace also includes a SHA-256 of each placeholder, its kind, and the original value. Redaction would otherwise map different values onto the same `[[PII:kind:n]]` text. Embeddings are hashed lexical vectors (word unigrams, bigrams, character trigrams) so CI does not call an embedding API. A hit requires cosine similarity at or above the configured threshold, a live TTL, and `temperature == 0` with caching enabled. Incremental streaming bypasses the cache. Buffered streaming runs the non-streaming path first, so it can hit. The same store interface is implemented in memory and as one Redis list per namespace.

Hits commit a zero cost. The reservation is still taken before the lookup so a burst of identical requests cannot all pass a nearly exhausted cap and then discover they were hits.

## Consequences

Paraphrases that do not share tokens will miss. A Redis list will not scale to millions of distinct prompts per tenant. That is a later index, not a different security boundary. The namespace stays mandatory either way.
