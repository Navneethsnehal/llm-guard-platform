"""Normalize obfuscated prompts before injection scoring."""

import base64
import codecs
import re
import unicodedata

_ZERO_WIDTH = frozenset("\u200b\u200c\u200d\ufeff\u2060")
_HOMOGLYPHS = str.maketrans(
    {
        "\u0430": "a",
        "\u0435": "e",
        "\u043e": "o",
        "\u0440": "p",
        "\u0441": "c",
        "\u0445": "x",
        "\u0443": "y",
        "\u0456": "i",
        "\u0410": "A",
        "\u0415": "E",
        "\u041e": "O",
        "\u0420": "P",
        "\u0421": "C",
    }
)
_LEET = str.maketrans({"0": "o", "1": "i", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a"})
_BASE64 = re.compile(r"(?<![A-Za-z0-9+/])([A-Za-z0-9+/]{16,}={0,2})(?![A-Za-z0-9+/])")
_MIXED = re.compile(r"[A-Za-z0-9@]+")
_SUSPICIOUS = re.compile(r"(?i)ignore|system prompt|instruction|canary|jailbreak|bypass")


def fold(text: str) -> str:
    """Strip zero-width characters, apply NFKC, and map common homoglyphs."""

    stripped = "".join(char for char in text if char not in _ZERO_WIDTH)
    normalized = unicodedata.normalize("NFKC", stripped)
    return normalized.translate(_HOMOGLYPHS).lower()


def leet_fold(text: str) -> str:
    """Map leetspeak only inside tokens that mix letters and digits."""

    def replace(match: re.Match[str]) -> str:
        token = match.group(0)
        letters = sum(char.isalpha() for char in token)
        digits = sum(char.isdigit() for char in token)
        if letters == 0 or digits == 0:
            return token
        return token.translate(_LEET)

    return _MIXED.sub(replace, text)


def decoded_base64_segments(text: str) -> list[str]:
    found: list[str] = []
    for match in _BASE64.finditer(text):
        blob = match.group(1)
        padded = blob + ("=" * ((4 - len(blob) % 4) % 4))
        try:
            raw = base64.b64decode(padded, validate=True)
            decoded = raw.decode("utf-8")
        except (ValueError, UnicodeDecodeError):
            continue
        if not decoded or not re.search(r"[A-Za-z]", decoded):
            continue
        printable = sum(char.isprintable() for char in decoded) / len(decoded)
        if printable > 0.9:
            found.append(decoded)
    return found


def rot13_if_suspicious(text: str) -> str | None:
    transformed = codecs.decode(text, "rot_13")
    if transformed != text and _SUSPICIOUS.search(transformed):
        return transformed
    return None


def attack_variants(text: str) -> list[str]:
    """Return the original text plus decoded views used by the injection stage."""

    folded = fold(text)
    ordered = [text, folded, leet_fold(folded)]
    ordered.extend(decoded_base64_segments(text))
    rotated = rot13_if_suspicious(folded)
    if rotated is not None:
        ordered.append(rotated)
    unique: list[str] = []
    seen: set[str] = set()
    for item in ordered:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique
