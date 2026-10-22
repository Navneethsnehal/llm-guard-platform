"""PII detection with optional reversible tokenization."""

import hashlib
import re
from dataclasses import dataclass

_EMAIL = re.compile(r"\b([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})\b")
_PHONE = re.compile(r"(?<!\w)(?:\+1[\s.\-]?)?(?:\(\d{3}\)|\d{3})[\s.\-]\d{3}[\s.\-]\d{4}(?!\w)")
_SSN = re.compile(r"(?<!\d)(\d{3}-\d{2}-\d{4})(?!\d)")
_CARD_GROUPED = re.compile(r"(?<!\d)((?:\d{4}[\s-]){3}\d{4})(?!\d)")
_CARD_PLAIN = re.compile(r"(?<!\d)(\d{13,19})(?!\d)")
_ACCOUNT = re.compile(r"(?i)\b(?:account|acct)\s*(?:number|no\.?|#)?\s*[:#]?\s*(\d{6,17})\b")
_TOKEN = re.compile(r"\[\[PII:([a-z_]+):(\d+)\]\]")

_PRIORITY = {
    "credit_card": 0,
    "ssn": 1,
    "email": 2,
    "account_number": 3,
    "phone": 4,
}


@dataclass(frozen=True, slots=True)
class PiiSpan:
    start: int
    end: int
    kind: str
    value: str


class PiiVault:
    """Per-request map from placeholder tokens back to original values."""

    def __init__(self) -> None:
        self._values: dict[str, tuple[str, str]] = {}
        self._counts: dict[str, int] = {}

    def put(self, kind: str, value: str) -> str:
        self._counts[kind] = self._counts.get(kind, 0) + 1
        token = f"[[PII:{kind}:{self._counts[kind]}]]"
        self._values[token] = (kind, value)
        return token

    def fingerprint(self) -> str:
        """Stable id for the values replaced in this request.

        Token text is not enough to tell two redacted prompts apart: the first
        email is always ``[[PII:email:1]]``.
        """

        if not self._values:
            return ""
        material = "\n".join(f"{token}\0{kind}\0{value}" for token, (kind, value) in self._values.items())
        return hashlib.sha256(material.encode("utf-8")).hexdigest()

    def render(self, text: str, reversible: set[str]) -> str:
        def replace(match: re.Match[str]) -> str:
            token = match.group(0)
            stored = self._values.get(token)
            if stored is None:
                return token
            kind, value = stored
            if kind in reversible:
                return value
            return f"[REDACTED:{kind}]"

        return _TOKEN.sub(replace, text)


def luhn_ok(number: str) -> bool:
    digits = [int(char) for char in number if char.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for index, digit in enumerate(reversed(digits)):
        if index % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def detect_pii(text: str) -> list[PiiSpan]:
    spans: list[PiiSpan] = []
    spans.extend(_spans(_EMAIL, text, "email"))
    spans.extend(_spans(_PHONE, text, "phone"))
    spans.extend(_spans(_SSN, text, "ssn"))
    spans.extend(_card_spans(text))
    spans.extend(_spans(_ACCOUNT, text, "account_number"))
    spans.sort(key=lambda span: (span.start, _PRIORITY.get(span.kind, 9), -(span.end - span.start)))
    kept: list[PiiSpan] = []
    cursor = -1
    for span in spans:
        if span.start < cursor:
            continue
        kept.append(span)
        cursor = span.end
    return kept


def redact(text: str, vault: PiiVault) -> tuple[str, tuple[str, ...]]:
    spans = detect_pii(text)
    if not spans:
        return text, ()
    pieces: list[str] = []
    cursor = 0
    kinds: list[str] = []
    for span in spans:
        pieces.append(text[cursor : span.start])
        pieces.append(vault.put(span.kind, span.value))
        kinds.append(span.kind)
        cursor = span.end
    pieces.append(text[cursor:])
    return "".join(pieces), tuple(kinds)


def _spans(pattern: re.Pattern[str], text: str, kind: str) -> list[PiiSpan]:
    found: list[PiiSpan] = []
    for match in pattern.finditer(text):
        group = 1 if match.lastindex else 0
        found.append(PiiSpan(match.start(group), match.end(group), kind, match.group(group)))
    return found


def _card_spans(text: str) -> list[PiiSpan]:
    found: list[PiiSpan] = []
    for pattern in (_CARD_GROUPED, _CARD_PLAIN):
        for match in pattern.finditer(text):
            value = match.group(1)
            if luhn_ok(value):
                found.append(PiiSpan(match.start(1), match.end(1), "credit_card", value))
    return found
