"""Pluggable chat providers. The fake provider is deterministic and offline."""

import asyncio
import json
import re
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

import httpx

from llm_guard.errors import ProviderError
from llm_guard.guardrails.grounding import content_tokens, sentences
from llm_guard.types import (
    Message,
    ProviderEvent,
    ProviderRequest,
    ProviderResponse,
    Usage,
    usage_for,
)

_LEAK = re.compile(
    r"(?i)\b(reveal|print|show|repeat|output|dump)\b.{0,40}\b("
    r"system prompt|hidden instructions|canary|secret|above)\b"
)


class ChatProvider(ABC):
    name: str

    @abstractmethod
    async def complete(self, request: ProviderRequest, *, timeout: float) -> ProviderResponse:
        raise NotImplementedError

    async def stream(self, request: ProviderRequest, *, timeout: float) -> AsyncIterator[ProviderEvent]:
        response = await self.complete(request, timeout=timeout)
        yield ProviderEvent(text=response.text, usage=response.usage)


class FakeProvider(ChatProvider):
    """Deterministic model used for tests, CI, evals, and overhead benchmarks.

    Behaviors:
    - faithful: answer with the context sentence that best overlaps the question
    - leak: comply with prompt-exfiltration requests (the unsafe baseline)
    - echo: return the last user message unchanged
    - ungrounded: invent a claim that is not in the context
    - json: return a small JSON object
    - fail: raise a retryable provider error
    """

    def __init__(
        self,
        name: str,
        *,
        behavior: str = "faithful",
        latency_s: float = 0.0,
        fail_times: int = 0,
    ) -> None:
        self.name = name
        self.behavior = behavior
        self.latency_s = latency_s
        self._fail_remaining = 10**9 if behavior == "fail" else fail_times

    async def complete(self, request: ProviderRequest, *, timeout: float) -> ProviderResponse:
        del timeout
        if self.latency_s > 0:
            await asyncio.sleep(self.latency_s)
        if self._fail_remaining > 0:
            self._fail_remaining -= 1
            raise ProviderError(f"{self.name} injected failure", retryable=True)
        text = synthesize(request.messages, self.behavior)
        return ProviderResponse(
            text=text,
            model=self.name,
            usage=usage_for(request.messages, text),
            finish_reason="stop",
        )

    async def stream(self, request: ProviderRequest, *, timeout: float) -> AsyncIterator[ProviderEvent]:
        response = await self.complete(request, timeout=timeout)
        step = 24
        for index in range(0, len(response.text), step):
            yield ProviderEvent(text=response.text[index : index + step])
        yield ProviderEvent(usage=response.usage)


def synthesize(messages: tuple[Message, ...], behavior: str) -> str:
    system = "\n".join(message.content for message in messages if message.role == "system")
    user = next((message.content for message in reversed(messages) if message.role == "user"), "")
    if behavior == "echo":
        return user
    if behavior == "ungrounded":
        return "Atlantis has a winter festival every year."
    if behavior == "json":
        if "platinum" in user.lower() or "platinum" in system.lower():
            return json.dumps({"product": "platinum", "apr_percent": 2.9})
        return json.dumps({"status": "ok"})
    if behavior == "leak" or _LEAK.search(user):
        return system or "No system prompt was set."
    if behavior == "faithful" and _asks_to_echo_pii(user):
        return user
    return _faithful_answer(user, system)


def _asks_to_echo_pii(user: str) -> bool:
    if not re.search(r"(?i)\b(repeat|quote)\b", user):
        return False
    has_ssn = re.search(r"\d{3}-\d{2}-\d{4}", user) is not None or "ssn" in user.lower()
    has_card = re.search(r"(?:\d{4}[\s-]?){3}\d{4}", user) is not None
    return has_ssn or has_card


def _faithful_answer(question: str, context: str) -> str:
    choices = sentences(context)
    if not choices:
        return "I don't know based on the provided context."
    question_tokens = set(content_tokens(question))
    if not question_tokens:
        return "I don't know based on the provided context."

    def overlap(sentence: str) -> float:
        tokens = set(content_tokens(sentence))
        if not tokens:
            return 0.0
        return len(tokens & question_tokens) / len(question_tokens)

    best = max(choices, key=overlap)
    if overlap(best) < 0.2:
        return "I don't know based on the provided context."
    return best


class OpenAIProvider(ChatProvider):
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        client: httpx.AsyncClient | None = None,
        name: str = "openai",
    ) -> None:
        self.name = name
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._client = client

    async def complete(self, request: ProviderRequest, *, timeout: float) -> ProviderResponse:
        self._require_key()
        payload = {
            "model": request.model,
            "messages": [{"role": message.role, "content": message.content} for message in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": False,
        }
        response = await self._send("POST", "/chat/completions", payload, timeout)
        body = _json_object(response)
        text, prompt_tokens, completion_tokens = _parse_openai_completion(body)
        return ProviderResponse(
            text=text,
            model=str(body.get("model", request.model)),
            usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
        )

    async def stream(self, request: ProviderRequest, *, timeout: float) -> AsyncIterator[ProviderEvent]:
        self._require_key()
        payload = {
            "model": request.model,
            "messages": [{"role": message.role, "content": message.content} for message in request.messages],
            "temperature": request.temperature,
            "max_tokens": request.max_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        async for event in self._stream_sse("/chat/completions", payload, timeout):
            yield event

    def _require_key(self) -> None:
        if not self._api_key:
            raise ProviderError("OPENAI_API_KEY is not configured", retryable=False)

    async def _send(self, method: str, path: str, payload: dict[str, object], timeout: float) -> httpx.Response:
        client = self._client or httpx.AsyncClient()
        owned = self._client is None
        try:
            response = await client.request(
                method,
                f"{self._base_url}{path}",
                json=payload,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=timeout,
            )
        except httpx.TimeoutException as exc:
            raise ProviderError("OpenAI request timed out", retryable=True) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"OpenAI transport error: {exc}", retryable=True) from exc
        finally:
            if owned:
                await client.aclose()
        _raise_for_status(response, "OpenAI")
        return response

    async def _stream_sse(
        self,
        path: str,
        payload: dict[str, object],
        timeout: float,
    ) -> AsyncIterator[ProviderEvent]:
        client = self._client or httpx.AsyncClient()
        owned = self._client is None
        try:
            async with client.stream(
                "POST",
                f"{self._base_url}{path}",
                json=payload,
                headers={"Authorization": f"Bearer {self._api_key}"},
                timeout=timeout,
            ) as response:
                if response.status_code >= 400:
                    raw = await response.aread()
                    _raise_for_status(
                        httpx.Response(response.status_code, content=raw),
                        "OpenAI",
                    )
                async for line in response.aiter_lines():
                    event = _openai_delta(line)
                    if event is not None:
                        yield event
        except httpx.TimeoutException as exc:
            raise ProviderError("OpenAI stream timed out", retryable=True) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"OpenAI transport error: {exc}", retryable=True) from exc
        finally:
            if owned:
                await client.aclose()


class AnthropicProvider(ChatProvider):
    def __init__(
        self,
        *,
        api_key: str,
        base_url: str = "https://api.anthropic.com",
        client: httpx.AsyncClient | None = None,
        name: str = "anthropic",
    ) -> None:
        self.name = name
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._client = client

    async def complete(self, request: ProviderRequest, *, timeout: float) -> ProviderResponse:
        self._require_key()
        system, messages = _split_anthropic(request)
        payload: dict[str, object] = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "messages": messages,
        }
        if system:
            payload["system"] = system
        response = await self._send(payload, timeout)
        body = _json_object(response)
        text, prompt_tokens, completion_tokens = _parse_anthropic_completion(body)
        return ProviderResponse(
            text=text,
            model=str(body.get("model", request.model)),
            usage=Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens),
        )

    async def stream(self, request: ProviderRequest, *, timeout: float) -> AsyncIterator[ProviderEvent]:
        response = await self.complete(request, timeout=timeout)
        yield ProviderEvent(text=response.text, usage=response.usage)

    def _require_key(self) -> None:
        if not self._api_key:
            raise ProviderError("ANTHROPIC_API_KEY is not configured", retryable=False)

    async def _send(self, payload: dict[str, object], timeout: float) -> httpx.Response:
        client = self._client or httpx.AsyncClient()
        owned = self._client is None
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        try:
            response = await client.post(
                f"{self._base_url}/v1/messages",
                json=payload,
                headers=headers,
                timeout=timeout,
            )
        except httpx.TimeoutException as exc:
            raise ProviderError("Anthropic request timed out", retryable=True) from exc
        except httpx.HTTPError as exc:
            raise ProviderError(f"Anthropic transport error: {exc}", retryable=True) from exc
        finally:
            if owned:
                await client.aclose()
        _raise_for_status(response, "Anthropic")
        return response


def _split_anthropic(request: ProviderRequest) -> tuple[str, list[dict[str, str]]]:
    system = "\n".join(message.content for message in request.messages if message.role == "system")
    messages = [
        {"role": message.role, "content": message.content}
        for message in request.messages
        if message.role in {"user", "assistant"}
    ]
    if not messages:
        messages = [{"role": "user", "content": ""}]
    return system, messages


def _raise_for_status(response: httpx.Response, name: str) -> None:
    if response.status_code < 400:
        return
    retryable = response.status_code == 429 or response.status_code >= 500
    detail = response.text[:300]
    raise ProviderError(
        f"{name} returned HTTP {response.status_code}: {detail}",
        retryable=retryable,
        status_code=response.status_code,
    )


def _json_object(response: httpx.Response) -> dict[str, object]:
    try:
        parsed = response.json()
    except json.JSONDecodeError as exc:
        raise ProviderError("provider returned non-JSON body", retryable=False) from exc
    if not isinstance(parsed, dict):
        raise ProviderError("provider JSON was not an object", retryable=False)
    return parsed


def _parse_openai_completion(body: dict[str, object]) -> tuple[str, int, int]:
    choices = _as_list(body.get("choices"))
    if not choices:
        raise ProviderError("malformed OpenAI response", retryable=False)
    choice = _as_dict(choices[0])
    if choice is None:
        raise ProviderError("malformed OpenAI response", retryable=False)
    message = _as_dict(choice.get("message"))
    if message is None or not isinstance(message.get("content"), str):
        raise ProviderError("malformed OpenAI response", retryable=False)
    prompt_tokens, completion_tokens = _usage(body.get("usage"), "prompt_tokens", "completion_tokens")
    return str(message["content"]), prompt_tokens, completion_tokens


def _parse_anthropic_completion(body: dict[str, object]) -> tuple[str, int, int]:
    blocks = _as_list(body.get("content"))
    if blocks is None:
        raise ProviderError("malformed Anthropic response", retryable=False)
    parts: list[str] = []
    for block in blocks:
        parsed = _as_dict(block)
        if parsed is None or parsed.get("type") != "text":
            continue
        text = parsed.get("text")
        if isinstance(text, str):
            parts.append(text)
    prompt_tokens, completion_tokens = _usage(body.get("usage"), "input_tokens", "output_tokens")
    return "".join(parts), prompt_tokens, completion_tokens


def _optional_usage(value: object) -> Usage | None:
    if value is None:
        return None
    prompt_tokens, completion_tokens = _usage(value, "prompt_tokens", "completion_tokens")
    return Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)


def _delta_text(choices: object) -> str:
    parsed_choices = _as_list(choices)
    if not parsed_choices:
        return ""
    choice = _as_dict(parsed_choices[0])
    if choice is None:
        return ""
    delta = _as_dict(choice.get("delta"))
    if delta is None:
        return ""
    content = delta.get("content")
    if not isinstance(content, str):
        return ""
    return content


def _usage(value: object, prompt_key: str, completion_key: str) -> tuple[int, int]:
    parsed = _as_dict(value) or {}
    prompt = parsed.get(prompt_key, 0)
    completion = parsed.get(completion_key, 0)
    if not isinstance(prompt, int) or not isinstance(completion, int):
        raise ProviderError("malformed provider usage", retryable=False)
    return prompt, completion


def _as_dict(value: object) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    parsed: dict[str, object] = {}
    for key, item in value.items():
        if isinstance(key, str):
            parsed[key] = item
    return parsed


def _as_list(value: object) -> list[object] | None:
    if not isinstance(value, list):
        return None
    return list(value)


def _openai_delta(line: str) -> ProviderEvent | None:
    if not line or line.startswith(":"):
        return None
    if not line.startswith("data:"):
        return None
    data = line[len("data:") :].strip()
    if data == "[DONE]" or data == "":
        return None
    try:
        parsed_json: object = json.loads(data)
    except json.JSONDecodeError as exc:
        raise ProviderError("malformed OpenAI stream event", retryable=False) from exc
    parsed = _as_dict(parsed_json)
    if parsed is None:
        return None
    usage = _optional_usage(parsed.get("usage"))
    content = _delta_text(parsed.get("choices"))
    if content == "" and usage is None:
        return None
    return ProviderEvent(text=content, usage=usage)
