"""OpenAI-compatible request and response models."""

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from llm_guard.types import Message, ResponseFormat


class TextPart(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["text"]
    text: str


class ChatMessageIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    role: Literal["system", "user", "assistant"]
    content: str | list[TextPart]
    name: str | None = None

    def plain(self) -> str:
        if isinstance(self.content, str):
            return self.content
        return "".join(part.text for part in self.content)


class JsonSchemaBody(BaseModel):
    model_config = ConfigDict(extra="allow", populate_by_name=True)

    name: str | None = None
    schema_payload: dict[str, Any] | None = Field(default=None, alias="schema")


class ResponseFormatIn(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["text", "json_object", "json_schema"] = "text"
    json_schema: JsonSchemaBody | None = None

    def to_domain(self) -> ResponseFormat:
        if self.type != "json_schema" or self.json_schema is None:
            return ResponseFormat(type=self.type)
        return ResponseFormat(type="json_schema", json_schema=self.json_schema.schema_payload)


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="ignore")

    include_usage: bool = False


class LLMGuardOptions(BaseModel):
    model_config = ConfigDict(extra="ignore")

    cache: bool | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    model: str
    messages: list[ChatMessageIn] = Field(min_length=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=4096)
    stream: bool = False
    n: int | None = 1
    tools: list[dict[str, Any]] | None = None
    response_format: ResponseFormatIn | None = None
    stream_options: StreamOptions | None = None
    llm_guard: LLMGuardOptions | None = None

    def to_messages(self) -> tuple[Message, ...]:
        return tuple(Message(role=message.role, content=message.plain()) for message in self.messages)


class ChatMessageOut(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class ChoiceOut(BaseModel):
    index: int
    message: ChatMessageOut
    finish_reason: str


class UsageOut(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class TraceOut(BaseModel):
    stage: str
    direction: str
    decision: str
    reasons: list[str]
    score: float | None = None
    latency_ms: float


class DiagnosticsOut(BaseModel):
    cache: str
    decision: str
    cost_usd: float
    guardrails: list[TraceOut]


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChoiceOut]
    usage: UsageOut
    llm_guard: DiagnosticsOut | None = None


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    owned_by: str = "llm-guard"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]
