"""Lexical grounding check of a completion against supplied context."""

import re

_STOP = frozenset(
    {
        "a",
        "an",
        "the",
        "of",
        "to",
        "and",
        "or",
        "in",
        "on",
        "for",
        "with",
        "without",
        "from",
        "by",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "it",
        "this",
        "that",
        "these",
        "those",
        "as",
        "at",
        "if",
        "then",
        "than",
        "into",
        "over",
        "after",
        "before",
        "not",
        "no",
        "do",
        "does",
        "did",
        "can",
        "could",
        "should",
        "would",
        "will",
        "just",
        "about",
        "what",
        "which",
        "who",
        "whom",
        "whose",
        "how",
        "when",
        "where",
        "why",
        "please",
    }
)
_SENTENCE = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_NUMBER = re.compile(r"\d+(?:\.\d+)?%?")
_ABSTENTION = re.compile(
    r"(?i)\b(i don't know|i do not know|not in the provided context|"
    r"cannot determine from the context|don't know based on the provided context)\b"
)


def sentences(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    return [part for part in _SENTENCE.split(stripped) if part]


def content_tokens(text: str) -> list[str]:
    return [token for token in _WORD.findall(text.lower()) if token not in _STOP and not token.isdigit()]


class GroundingReport:
    def __init__(self, score: float, reasons: tuple[str, ...]) -> None:
        self.score = score
        self.reasons = reasons


def grounding_report(answer: str, context: str) -> GroundingReport:
    """Score how much of ``answer`` is supported by ``context``.

    Numbers in the answer must appear in the context. Sentences with little
    content-word overlap are counted as unsupported. Abstentions are treated
    as fully grounded: declining to answer is not a hallucination.
    """

    if not answer.strip():
        return GroundingReport(1.0, ("empty",))
    if not context.strip():
        return GroundingReport(1.0, ("no_context",))
    if _ABSTENTION.search(answer):
        return GroundingReport(1.0, ("abstained",))

    context_tokens = set(content_tokens(context))
    context_numbers = set(_NUMBER.findall(context))
    scores: list[float] = []
    unsupported = 0
    for sentence in sentences(answer):
        tokens = content_tokens(sentence)
        numbers = _NUMBER.findall(sentence)
        if not tokens and not numbers:
            continue
        overlap = (len(set(tokens) & context_tokens) / len(set(tokens))) if tokens else 1.0
        numbers_ok = all(number in context_numbers for number in numbers) if numbers else True
        sentence_score = overlap if numbers_ok else overlap * 0.25
        scores.append(sentence_score)
        if sentence_score < 0.35:
            unsupported += 1
    if not scores:
        return GroundingReport(1.0, ("no_claims",))
    mean = sum(scores) / len(scores)
    reasons: list[str] = []
    if unsupported:
        reasons.append(f"ungrounded_sentence_count:{unsupported}")
    return GroundingReport(mean, tuple(reasons))
