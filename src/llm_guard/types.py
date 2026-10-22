"""Shared value types for the gateway, providers, and guardrails."""

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Message:
    role: str
    content: str


@dataclass(frozen=True, slots=True)
class Usage:
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True, slots=True)
class ResponseFormat:
    """Subset of the OpenAI response_format object the gateway enforces."""

    type: str = "text"
    json_schema: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class CompletionRequest:
    model: str
    messages: tuple[Message, ...]
    temperature: float = 0.0
    max_tokens: int = 256
    stream: bool = False
    response_format: ResponseFormat | None = None
    include_usage: bool = False
    use_cache: bool = True


@dataclass(frozen=True, slots=True)
class CompletionResult:
    id: str
    created: int
    model: str
    text: str
    usage: Usage
    cost_usd: float
    cache: str
    decision: str
    traces: tuple["GuardTrace", ...] = ()


@dataclass(frozen=True, slots=True)
class GuardTrace:
    stage: str
    direction: str
    decision: str
    reasons: tuple[str, ...]
    score: float | None
    latency_ms: float


@dataclass(frozen=True, slots=True)
class ProviderRequest:
    model: str
    messages: tuple[Message, ...]
    temperature: float
    max_tokens: int


@dataclass(frozen=True, slots=True)
class ProviderResponse:
    text: str
    model: str
    usage: Usage
    finish_reason: str = "stop"


@dataclass(frozen=True, slots=True)
class ProviderEvent:
    text: str = ""
    usage: Usage | None = None


def render_messages(messages: tuple[Message, ...] | list[Message]) -> str:
    return "\n".join(f"{message.role}: {message.content}" for message in messages)


def estimate_tokens(text: str) -> int:
    """Approximate token count used when a provider does not report usage.

    The estimate is deliberately simple and stable: one token per four
    characters, with a floor of one token for any non-empty string. Cost
    accounting and reservations share this function so they cannot drift.
    """

    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def usage_for(messages: tuple[Message, ...] | list[Message], completion: str) -> Usage:
    prompt = estimate_tokens(render_messages(messages))
    completion_tokens = estimate_tokens(completion)
    return Usage(prompt_tokens=prompt, completion_tokens=completion_tokens)
