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
# Longest phrase first. A short match must not hide the rest of a canned abstention.
_ABSTENTION_PHRASES = (
    re.compile(r"(?i)\bi don't know based on the provided context\b"),
    re.compile(r"(?i)\bdon't know based on the provided context\b"),
    re.compile(r"(?i)\bcannot determine from the context\b"),
    re.compile(r"(?i)\bnot in the provided context\b"),
    re.compile(r"(?i)\bi do not know\b"),
    re.compile(r"(?i)\bi don't know\b"),
)


def _claim_text(sentence: str) -> str | None:
    """Return the claim left after abstention phrases, or None when nothing remains."""

    remainder = sentence
    matched = False
    for pattern in _ABSTENTION_PHRASES:
        if pattern.search(remainder):
            matched = True
            remainder = pattern.sub(" ", remainder)
    if not matched:
        return sentence
    if not content_tokens(remainder) and not _NUMBER.findall(remainder):
        return None
    return remainder


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
    content-word overlap are counted as unsupported. A sentence that only
    declines to answer is not a claim. Other sentences in the same answer
    are still scored.
    """

    if not answer.strip():
        return GroundingReport(1.0, ("empty",))
    if not context.strip():
        return GroundingReport(1.0, ("no_context",))

    context_tokens = set(content_tokens(context))
    context_numbers = set(_NUMBER.findall(context))
    scores: list[float] = []
    unsupported = 0
    saw_abstention = False
    for sentence in sentences(answer):
        claim = _claim_text(sentence)
        if claim is None:
            saw_abstention = True
            continue
        tokens = content_tokens(claim)
        numbers = _NUMBER.findall(claim)
        if not tokens and not numbers:
            continue
        overlap = (len(set(tokens) & context_tokens) / len(set(tokens))) if tokens else 1.0
        numbers_ok = all(number in context_numbers for number in numbers) if numbers else True
        sentence_score = overlap if numbers_ok else overlap * 0.25
        scores.append(sentence_score)
        if sentence_score < 0.35:
            unsupported += 1
    if not scores:
        if saw_abstention:
            return GroundingReport(1.0, ("abstained",))
        return GroundingReport(1.0, ("no_claims",))
    mean = sum(scores) / len(scores)
    reasons: list[str] = []
    if unsupported:
        reasons.append(f"ungrounded_sentence_count:{unsupported}")
    return GroundingReport(mean, tuple(reasons))
