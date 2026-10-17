"""Domain errors mapped to OpenAI-style HTTP responses by the API layer."""


class GatewayError(Exception):
    """Base class for errors that should be returned to the caller."""

    status_code: int = 500
    error_type: str = "internal_error"
    code: str = "internal_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class AuthenticationError(GatewayError):
    status_code = 401
    error_type = "authentication_error"
    code = "invalid_api_key"


class RateLimitError(GatewayError):
    status_code = 429
    error_type = "rate_limit_exceeded"
    code = "rate_limit"

    def __init__(self, message: str, retry_after: float) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class BudgetExceededError(GatewayError):
    status_code = 402
    error_type = "budget_exceeded"
    code = "budget_exceeded"


class GuardrailBlocked(GatewayError):
    status_code = 400
    error_type = "content_filter"
    code = "guardrail_blocked"

    def __init__(self, stage: str, reasons: list[str]) -> None:
        super().__init__(f"Blocked by guardrail stage '{stage}'")
        self.stage = stage
        self.reasons = list(reasons)


class UnknownModelError(GatewayError):
    status_code = 404
    error_type = "invalid_request_error"
    code = "model_not_found"


class UpstreamError(GatewayError):
    status_code = 503
    error_type = "upstream_error"
    code = "upstream_unavailable"


class InvalidRequestError(GatewayError):
    status_code = 400
    error_type = "invalid_request_error"
    code = "invalid_request"


class PayloadTooLargeError(GatewayError):
    status_code = 413
    error_type = "invalid_request_error"
    code = "payload_too_large"


class ProviderError(Exception):
    """Failure from an upstream model provider."""

    def __init__(self, message: str, *, retryable: bool, status_code: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.retryable = retryable
        self.status_code = status_code
