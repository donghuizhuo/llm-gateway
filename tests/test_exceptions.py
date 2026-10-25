import pytest

from llm_gateway.exceptions import (
    LLMGatewayError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)


def test_timeout_caught_by_gateway_error():
    with pytest.raises(LLMGatewayError):
        raise ProviderTimeoutError("Provider request timed out.")


def test_caught_by_gateway_error():
    with pytest.raises(LLMGatewayError):
        raise ProviderRateLimitError("Provider rate limit exceeded.")


def test_inheritance():
    assert issubclass(ProviderRateLimitError, LLMGatewayError)
    assert issubclass(ProviderTimeoutError, LLMGatewayError)


def test_retry_after_attribute():
    error_with_retry = ProviderRateLimitError("Rate limited", retry_after=5.0)
    assert str(error_with_retry) == "Rate limited"
    assert error_with_retry.retry_after == 5.0

    error_without_retry = ProviderRateLimitError("Rate limited.")
    assert str(error_without_retry) == "Rate limited."
    assert error_without_retry.retry_after is None
