# ADR 0006: Bootstrap confidence intervals for the release gate

## Status

Accepted

## Context

A one-point drop on a 12-case golden set can be noise, especially for latency measured in-process. A gate that blocks on any decrease will flake. A gate that only compares means will ship a real refusal regression.

## Decision

The harness scores quality, refusal correctness, faithfulness, cost, and latency on a fixed JSONL set. The gate pairs baseline and candidate by case id and draws a paired bootstrap (default 1000 resamples, seed 42, alpha 0.05). Higher-is-better metrics block only when the confidence interval lies entirely below zero and the mean drop exceeds a minimum effect. Cost blocks when the interval lies entirely above zero and the increase exceeds `1e-9` USD. Latency blocks only when the p95 regression is both significant and larger than `max(50ms, 50% of baseline p95)`.

The CLI exits 1 when the decision is `block`. CI runs the smoke subset. The full file is the release check.

## Consequences

Identical configs pass. Two fast in-process runs will not fail the latency gate because of a few milliseconds. A refusal that flips from correct to incorrect has a minimum effect of zero, so a significant drop blocks even when it is a single case, provided the interval excludes zero. With very small sets the interval is wide, which is why the golden file has to stay large enough that a real regression is visible. Seed and resample count are part of the contract so two machines agree.
