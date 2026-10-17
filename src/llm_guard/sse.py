"""Server-sent events for the OpenAI chat-completions stream."""

import json
from collections.abc import Iterator

from llm_guard.types import CompletionResult, Usage


def chunk_text(text: str, size: int = 48) -> list[str]:
    if size < 1:
        raise ValueError("size must be positive")
    if text == "":
        return []
    return [text[index : index + size] for index in range(0, len(text), size)]


def sse(payload: object) -> str:
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


def sse_done() -> str:
    return "data: [DONE]\n\n"


def completion_chunk(
    *,
    completion_id: str,
    created: int,
    model: str,
    delta: dict[str, str] | None,
    finish_reason: str | None,
    usage: Usage | None = None,
) -> dict[str, object]:
    choice: dict[str, object] = {
        "index": 0,
        "delta": delta if delta is not None else {},
        "finish_reason": finish_reason,
    }
    payload: dict[str, object] = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [choice],
    }
    if usage is not None:
        payload["usage"] = {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }
    return payload


def iter_buffered(result: CompletionResult, *, include_usage: bool) -> Iterator[str]:
    completion_id = result.id
    created = result.created
    model = result.model
    yield sse(
        completion_chunk(
            completion_id=completion_id,
            created=created,
            model=model,
            delta={"role": "assistant"},
            finish_reason=None,
        )
    )
    for piece in chunk_text(result.text):
        yield sse(
            completion_chunk(
                completion_id=completion_id,
                created=created,
                model=model,
                delta={"content": piece},
                finish_reason=None,
            )
        )
    usage = result.usage if include_usage else None
    yield sse(
        completion_chunk(
            completion_id=completion_id,
            created=created,
            model=model,
            delta=None,
            finish_reason="stop",
            usage=usage,
        )
    )
    yield sse_done()
