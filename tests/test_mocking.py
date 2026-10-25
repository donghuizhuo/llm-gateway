from unittest.mock import Mock

import pytest

from llm_gateway.exceptions import ProviderRateLimitError


def test_provider_rate_limit_error():
    provider = Mock()

    provider.generate.side_effect = [
        ProviderRateLimitError("Provider rate limit exceeded."),
        ProviderRateLimitError("Provider rate limit exceeded."),
        "Success",
    ]

    with pytest.raises(ProviderRateLimitError):
        provider.generate("What is Python?")

    with pytest.raises(ProviderRateLimitError):
        provider.generate("What is Python?")

    result = provider.generate("What is Python?")

    assert result == "Success"  # This should succeed
    assert provider.generate.call_count == 3
