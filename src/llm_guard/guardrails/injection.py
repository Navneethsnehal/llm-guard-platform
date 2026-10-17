"""Prompt-injection stage: high-precision heuristics plus the local classifier."""

import re
from dataclasses import dataclass

from llm_guard.guardrails.classifier import InjectionModel
from llm_guard.guardrails.normalize import attack_variants

_HEURISTICS: tuple[tuple[str, re.Pattern[str]], ...] = tuple(
    (name, re.compile(pattern))
    for name, pattern in (
        (
            "ignore_previous_instructions",
            r"(?i)ignore (?:all |any |the )?(?:previous|prior|above|earlier) "
            r"(?:instructions|prompts|rules|guidelines)",
        ),
        (
            "disregard_rules",
            r"(?i)disregard (?:all |any |every |the )?(?:previous |prior |earlier )?"
            r"(?:instructions|rules|guidelines|policies|safety policy)",
        ),
        (
            "reveal_prompt",
            r"(?i)(?:reveal|print|show|output|repeat|dump|exfiltrate|leak).{0,48}"
            r"(?:system prompt|hidden instructions|hidden prompt|canary|secret)",
        ),
        ("developer_mode", r"(?i)developer mode"),
        ("dan_jailbreak", r"(?i)\b(?:do anything now|\bdan\b)"),
        (
            "bypass_safety",
            r"(?i)bypass (?:the )?(?:safety|guardrails?|guards?|policy filters?|"
            r"content filter|filters?)",
        ),
        (
            "no_restrictions",
            r"(?i)pretend (?:you have no|there are no) (?:restrictions|rules|policies)",
        ),
        ("system_tag", r"(?i)<\s*/?\s*(?:system|instructions)\s*>"),
        ("new_instructions", r"(?i)(?:new instructions|new system prompt)\s*:"),
        (
            "override_prompt",
            r"(?i)override (?:the )?(?:system prompt|safety|policy|previous instructions)",
        ),
        ("unfiltered", r"(?i)\bunfiltered\b"),
        (
            "forget_policies",
            r"(?i)forget (?:the |all )?(?:policies|instructions|rules) you were given",
        ),
        (
            "do_not_follow",
            r"(?i)do not follow (?:the )?(?:previous|prior|system) instructions",
        ),
        (
            "document_override",
            r"(?i)end of (?:policy|document).{0,40}new instructions",
        ),
    )
)


@dataclass(frozen=True, slots=True)
class InjectionDecision:
    decision: str
    reasons: tuple[str, ...]
    score: float


class InjectionStage:
    name = "injection"

    def __init__(self, model: InjectionModel | None, threshold: float) -> None:
        self.model = model
        self.threshold = threshold

    def evaluate(self, text: str) -> InjectionDecision:
        variants = attack_variants(text)
        reasons: list[str] = []
        for variant in variants:
            for name, pattern in _HEURISTICS:
                if pattern.search(variant):
                    reasons.append(f"heuristic:{name}")
        deduped = tuple(dict.fromkeys(reasons))
        probability = 0.0
        if self.model is not None and variants:
            probability = max(self.model.probability(variant) for variant in variants)
        if deduped:
            return InjectionDecision("block", deduped, max(probability, 1.0))
        if probability >= self.threshold:
            return InjectionDecision("block", (f"classifier:{probability:.3f}",), probability)
        return InjectionDecision("allow", (), probability)
