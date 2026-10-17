# ADR 0007: Local injection classifier

## Status

Accepted

## Context

High-precision regular expressions miss paraphrases and encoded payloads. Calling another hosted model to judge the prompt adds a dependency, a latency tax, and a data-processing question for every request in a bank.

## Decision

Heuristics block immediately on a small set of high-precision patterns, including matches against normalized variants (NFKC, homoglyphs, zero-width removal, mixed-token leetspeak, base64 segments, suspicious rot13). A hashed n-gram logistic regression, trained in-process with a fixed seed, covers the rest. Weights are JSON in the repository. `train-classifier` regenerates them. The operating threshold is the maximum of the YAML value and the threshold calibrated on the holdout split, so a config file cannot silently drop below the measured false-positive constraint.

The training text is synthetic and seeded. Holdout templates are the last eight benign and attack strings and are not passed through the train wrappers. The reported holdout F1 describes that split only. The red-team attack success rate is the number that gates a change.

## Consequences

The model is auditable: a reviewer can diff the weight file. It will not track a new attack family until the templates or the heuristics are updated and the red-team pack grows a case. That update is a code change with a CI run, which is the property we want for a control in a regulated environment.
