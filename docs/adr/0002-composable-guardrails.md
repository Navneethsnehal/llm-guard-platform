# ADR 0002: Composable guardrail stages

## Status

Accepted

## Context

PII, injection, topic policy, schema, and grounding fail for different reasons and need different evidence in an audit. A single boolean "safe" cannot be explained to a model-risk reviewer.

## Decision

Each stage returns a decision and reason codes. Input order is injection on the full transcript, topics on non-system messages, then PII per message. Output order is PII, injection, topics, schema, grounding, then vault render. The first block stops the pipeline. PII placeholders are rendered only after the later stages, so a reversible email is not restored into a response that is about to be blocked.

Topic rules are YAML per policy pack. Tenants point at a pack. System text is not scanned for topics: the operator writes it, and it often quotes the policy the user is allowed to ask about.

## Consequences

A new control is a stage with a trace, not a branch in the router. Stages do not see each other's private state except the per-request PII vault. Grounding is lexical. It will not catch a fluent paraphrase that shares no content words with the context, and it will abstain cleanly when the model says it does not know.
