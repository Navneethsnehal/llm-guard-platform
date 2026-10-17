"""FastAPI application factory."""

from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response, StreamingResponse

from llm_guard import __version__
from llm_guard.api.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessageOut,
    ChoiceOut,
    DiagnosticsOut,
    ModelCard,
    ModelList,
    TraceOut,
    UsageOut,
)
from llm_guard.auth import Tenant
from llm_guard.config import Settings, load_settings
from llm_guard.errors import GatewayError, GuardrailBlocked, InvalidRequestError, RateLimitError
from llm_guard.gateway import result_headers
from llm_guard.observability import request_id_var
from llm_guard.runtime import Runtime, build_runtime
from llm_guard.types import CompletionRequest, CompletionResult

CallNext = Callable[[Request], Awaitable[Response]]


def create_app(settings: Settings | None = None, runtime: Runtime | None = None) -> FastAPI:
    resolved = settings if settings is not None else load_settings()
    app_runtime = runtime if runtime is not None else build_runtime(resolved)

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        yield
        await app_runtime.aclose()

    app = FastAPI(
        title="llm-guard-platform",
        version=__version__,
        summary="OpenAI-compatible gateway with guardrails, routing, and audit.",
        lifespan=lifespan,
    )
    app.state.runtime = app_runtime

    @app.middleware("http")
    async def attach_request_id(request: Request, call_next: CallNext) -> Response:
        incoming = request.headers.get("x-request-id")
        request_id = incoming if incoming else f"req_{uuid4().hex[:24]}"
        token = request_id_var.set(request_id)
        try:
            response = await call_next(request)
        finally:
            request_id_var.reset(token)
        response.headers["x-request-id"] = request_id
        return response

    @app.exception_handler(GatewayError)
    async def on_gateway_error(_request: Request, exc: GatewayError) -> JSONResponse:
        error: dict[str, object] = {
            "message": exc.message,
            "type": exc.error_type,
            "code": exc.code,
        }
        if isinstance(exc, GuardrailBlocked):
            error["details"] = {"stage": exc.stage, "reasons": exc.reasons}
        headers: dict[str, str] = {}
        if isinstance(exc, RateLimitError):
            headers["retry-after"] = f"{exc.retry_after:.3f}"
        return JSONResponse(status_code=exc.status_code, content={"error": error}, headers=headers)

    @app.exception_handler(RequestValidationError)
    async def on_validation(_request: Request, exc: RequestValidationError) -> JSONResponse:
        message = "Invalid request"
        problems = exc.errors()
        if problems:
            first = problems[0]
            location = ".".join(str(part) for part in first.get("loc", []))
            message = f"{location}: {first.get('msg', 'invalid')}"
        return JSONResponse(
            status_code=400,
            content={
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                    "code": "invalid_request",
                }
            },
        )

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {
            "status": "ok",
            "version": __version__,
            "cache": app_runtime.settings.cache.backend,
            "models": len(app_runtime.router.models),
        }

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(
            content=app_runtime.metrics.render(),
            media_type="text/plain; version=0.0.4",
        )

    @app.get("/v1/models")
    async def models(request: Request) -> ModelList:
        _authenticate(request, app_runtime)
        return ModelList(data=[ModelCard(id=model_id) for model_id in app_runtime.router.known_ids()])

    @app.post("/v1/chat/completions")
    async def chat_completions(payload: ChatCompletionRequest, request: Request) -> Response:
        _reject_unsupported(payload)
        tenant = _authenticate(request, app_runtime)
        completion = _to_completion(payload, app_runtime.settings)
        if payload.stream:
            headers, body = await app_runtime.gateway.stream(tenant, completion)
            return StreamingResponse(body, media_type="text/event-stream", headers=headers)
        result = await app_runtime.gateway.execute(tenant, completion)
        exposed = app_runtime.settings.server.expose_diagnostics
        return JSONResponse(
            content=_response_body(result, exposed),
            headers=result_headers(result),
        )

    @app.get("/admin/snapshot")
    async def snapshot(request: Request, limit: int = 50) -> dict[str, object]:
        _authenticate_admin(request, app_runtime)
        return app_runtime.snapshot(audit_limit=max(0, min(limit, 500)))

    @app.get("/admin/audit")
    async def audit(request: Request, limit: int = 100) -> dict[str, object]:
        _authenticate_admin(request, app_runtime)
        bounded = max(0, min(limit, 1000))
        return {"events": [event.as_dict() for event in app_runtime.audit.recent(bounded)]}

    return app


def _authenticate(request: Request, runtime: Runtime) -> Tenant:
    return runtime.tenants.authenticate(_bearer(request.headers.get("authorization")))


def _authenticate_admin(request: Request, runtime: Runtime) -> None:
    runtime.tenants.authenticate_admin(
        _bearer(request.headers.get("authorization")),
        runtime.settings.server.admin_api_key,
    )


def _bearer(header: str | None) -> str | None:
    if header is None:
        return None
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or token.strip() == "":
        return None
    return token.strip()


def _reject_unsupported(payload: ChatCompletionRequest) -> None:
    if payload.n not in (None, 1):
        raise InvalidRequestError("Only n=1 is supported")
    if payload.tools:
        raise InvalidRequestError("Tool calls are not supported")


def _to_completion(payload: ChatCompletionRequest, settings: Settings) -> CompletionRequest:
    response_format = None if payload.response_format is None else payload.response_format.to_domain()
    use_cache = True
    if payload.llm_guard is not None and payload.llm_guard.cache is not None:
        use_cache = payload.llm_guard.cache
    include_usage = bool(payload.stream_options and payload.stream_options.include_usage)
    return CompletionRequest(
        model=payload.model,
        messages=payload.to_messages(),
        temperature=payload.temperature,
        max_tokens=payload.max_tokens or settings.server.reservation_output_tokens,
        stream=payload.stream,
        response_format=response_format,
        include_usage=include_usage,
        use_cache=use_cache,
    )


def _response_body(result: CompletionResult, expose: bool) -> dict[str, object]:
    diagnostics = None
    if expose:
        diagnostics = DiagnosticsOut(
            cache=result.cache,
            decision=result.decision,
            cost_usd=result.cost_usd,
            guardrails=[
                TraceOut(
                    stage=item.stage,
                    direction=item.direction,
                    decision=item.decision,
                    reasons=list(item.reasons),
                    score=item.score,
                    latency_ms=item.latency_ms,
                )
                for item in result.traces
            ],
        )
    model = ChatCompletionResponse(
        id=result.id,
        created=result.created,
        model=result.model,
        choices=[ChoiceOut(index=0, message=ChatMessageOut(content=result.text), finish_reason="stop")],
        usage=UsageOut(
            prompt_tokens=result.usage.prompt_tokens,
            completion_tokens=result.usage.completion_tokens,
            total_tokens=result.usage.total_tokens,
        ),
        llm_guard=diagnostics,
    )
    return model.model_dump(exclude_none=True)
