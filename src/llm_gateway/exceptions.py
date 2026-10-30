class LLMGatewayError(Exception):
    """Base class for exceptions in the LLM Gateway."""


class ProviderRateLimitError(LLMGatewayError):
    """Exception raised when a provider's rate limit is exceeded."""

    def __init__(
        self,
        message: str = "Provider request rate limited.",
        retry_after: float | None = None,
    ):
        super().__init__(message)
        self.retry_after = retry_after


class ProviderTimeoutError(LLMGatewayError):
    """Exception raised when a provider request times out."""
