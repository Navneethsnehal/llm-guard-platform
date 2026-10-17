"""Per-tenant topic restrictions compiled from policy packs."""

import re
from dataclasses import dataclass

from llm_guard.config import PolicyPack
from llm_guard.guardrails.normalize import fold


@dataclass(frozen=True, slots=True)
class CompiledTopic:
    id: str
    pattern: re.Pattern[str]


@dataclass(frozen=True, slots=True)
class TopicDecision:
    decision: str
    reasons: tuple[str, ...]


class TopicsStage:
    name = "topics"

    def __init__(self, packs: dict[str, tuple[CompiledTopic, ...]]) -> None:
        self._packs = packs

    @classmethod
    def from_packs(cls, packs: dict[str, PolicyPack]) -> "TopicsStage":
        compiled: dict[str, tuple[CompiledTopic, ...]] = {}
        for pack_id, pack in packs.items():
            rules: list[CompiledTopic] = []
            for topic in pack.deny_topics:
                for index, pattern in enumerate(topic.patterns):
                    try:
                        compiled_pattern = re.compile(pattern)
                    except re.error as exc:
                        raise ValueError(f"invalid pattern {pack_id}.{topic.id}[{index}]") from exc
                    rules.append(CompiledTopic(id=topic.id, pattern=compiled_pattern))
            compiled[pack_id] = tuple(rules)
        return cls(compiled)

    def evaluate(self, text: str, pack_id: str) -> TopicDecision:
        folded = fold(text)
        reasons: list[str] = []
        for rule in self._packs.get(pack_id, ()):
            if rule.pattern.search(text) or rule.pattern.search(folded):
                reasons.append(f"topic:{rule.id}")
        deduped = tuple(dict.fromkeys(reasons))
        if deduped:
            return TopicDecision("block", deduped)
        return TopicDecision("allow", ())
