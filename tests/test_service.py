from unittest.mock import Mock, patch

import pytest

from llm_gateway.exceptions import ProviderRateLimitError, ProviderTimeoutError
from llm_gateway.service import can_retry_before_deadline, generate_with_retry


def test_success_after_retries():
    provider = Mock()

    provider.generate.side_effect = [
        ProviderRateLimitError("Provider rate limit exceeded."),
        ProviderRateLimitError("Provider rate limit exceeded."),
        "Success",
    ]

    with patch("llm_gateway.service.time.sleep") as mock_sleep:
        result = generate_with_retry(provider, "What is Python?", max_attempts=3)

    assert result == "Success"  # This should succeed
    assert provider.generate.call_count == 3
    assert mock_sleep.call_count == 2


def test_retry_exhaustion():
    provider = Mock()

    provider.generate.side_effect = [
        ProviderRateLimitError("Provider rate limit exceeded."),
        ProviderRateLimitError("Provider rate limit exceeded."),
        ProviderRateLimitError("Provider rate limit exceeded."),
    ]

    with (
        pytest.raises(ProviderRateLimitError),
        patch("llm_gateway.service.time.sleep") as mock_sleep,
    ):
        generate_with_retry(provider, "What is Python?")

    assert provider.generate.call_count == 3
    assert mock_sleep.call_count == 2


def test_timeout_not_retried():
    provider = Mock()

    provider.generate.side_effect = [
        ProviderTimeoutError("Provider did not respond in time."),
    ]

    with pytest.raises(ProviderTimeoutError):
        generate_with_retry(provider, "What is Python?")

    assert provider.generate.call_count == 1


def test_invalid_max_attempts():
    provider = Mock()

    with pytest.raises(ValueError):
        generate_with_retry(provider, "What is Python?", max_attempts=0)

    assert provider.generate.call_count == 0


def test_success_without_retries():
    provider = Mock()

    provider.generate.return_value = "Success"

    result = generate_with_retry(provider, "What is Python?", max_attempts=3)

    assert result == "Success"
    assert provider.generate.call_count == 1


def test_retry_delay_calculation():
    from llm_gateway.service import calculate_retry_delay

    # Test with retry_number = 0
    delay = calculate_retry_delay(retry_number=0, base_delay=1.0, max_delay=30.0)
    assert 0 <= delay <= 1.0

    # Test with retry_number = 1
    delay = calculate_retry_delay(1, base_delay=1.0, max_delay=30.0)
    assert 0 <= delay <= 2.0

    # Test with retry_number = 2
    delay = calculate_retry_delay(retry_number=2, base_delay=1.0, max_delay=30.0)
    assert 0 <= delay <= 4.0

    # Test with retry_number = 5 (should hit max_delay)
    delay = calculate_retry_delay(retry_number=5, base_delay=1.0, max_delay=30.0)
    assert 0 <= delay <= 30.0


def test_retry_delay_with_mock():
    from llm_gateway.service import calculate_retry_delay

    with patch("llm_gateway.service.random.uniform", return_value=2.5) as mock_uniform:
        delay = calculate_retry_delay(retry_number=2, base_delay=1.0, max_delay=30.0)

    assert delay == 2.5  # Should return the mocked value
    mock_uniform.assert_called_once_with(
        0, 4.0
    )  # Check the upper bound for retry_number=2


def test_retry_max_delay_with_mock():
    from llm_gateway.service import calculate_retry_delay

    with patch("llm_gateway.service.random.uniform", return_value=25.0) as mock_uniform:
        delay = calculate_retry_delay(retry_number=10, base_delay=1.0, max_delay=30.0)

    assert delay == 25.0  # Should return the mocked value
    mock_uniform.assert_called_once_with(
        0, 30.0
    )  # Check the upper bound for retry_number=10


def test_retry_waits_between_attempts():
    provider = Mock()

    # Simulate two rate limit errors followed by a success
    provider.generate.side_effect = [
        ProviderRateLimitError("Provider rate limit exceeded."),
        ProviderRateLimitError("Provider rate limit exceeded."),
        "Success",
    ]

    with (
        patch(
            "llm_gateway.service.calculate_retry_delay", side_effect=[0.5, 1.2]
        ) as mock_delay,
        patch("llm_gateway.service.time.sleep") as mock_sleep,
    ):
        result = generate_with_retry(provider, "What is Python?", max_attempts=3)

    assert result == "Success"
    assert provider.generate.call_count == 3
    assert mock_delay.call_count == 2
    assert mock_sleep.call_count == 2
    assert mock_sleep.call_args_list[0].args == (0.5,)
    assert mock_sleep.call_args_list[1].args == (1.2,)


def test_no_sleep_after_retry_exhaustion():
    provider = Mock()

    # Simulate three rate limit errors
    provider.generate.side_effect = ProviderRateLimitError(
        "Provider rate limit exceeded."
    )

    with (
        pytest.raises(ProviderRateLimitError),
        patch("llm_gateway.service.time.sleep") as mock_sleep,
    ):
        generate_with_retry(provider, "What is Python?", max_attempts=3)

    assert provider.generate.call_count == 3
    assert mock_sleep.call_count == 2


def test_retry_respects_retry_after():
    provider = Mock()

    # Simulate two rate limit errors with retry_after, followed by a success
    provider.generate.side_effect = [
        ProviderRateLimitError("Provider rate limit exceeded.", retry_after=10.0),
        "Success",
    ]

    with (
        patch(
            "llm_gateway.service.calculate_retry_delay", return_value=2.5
        ) as mock_delay,
        patch("llm_gateway.service.time.sleep") as mock_sleep,
    ):
        result = generate_with_retry(provider, "What is Python?", max_attempts=3)

    assert result == "Success"
    assert provider.generate.call_count == 2
    assert mock_delay.call_count == 1
    mock_sleep.assert_called_once_with(
        10.0
    )  # Should wait for retry_after, not the calculated delay


def test_retry_uses_longer_local_backoff():
    provider = Mock()

    # Simulate two rate limit errors with retry_after, followed by a success
    provider.generate.side_effect = [
        ProviderRateLimitError("Provider rate limit exceeded.", retry_after=2.0),
        "Success",
    ]

    with (
        patch(
            "llm_gateway.service.calculate_retry_delay", return_value=5.0
        ) as mock_delay,
        patch("llm_gateway.service.time.sleep") as mock_sleep,
    ):
        result = generate_with_retry(provider, "What is Python?", max_attempts=3)

    assert result == "Success"
    assert provider.generate.call_count == 2
    assert mock_delay.call_count == 1
    mock_sleep.assert_called_once_with(
        5.0
    )  # Should wait for the longer local backoff, not retry_after


def test_can_retry_before_deadline():

    with patch("llm_gateway.service.time.monotonic", return_value=100.0):
        assert (
            can_retry_before_deadline(delay=5.0, deadline=115.0) is True
        )  # 100 + 5 < 115
        assert (
            can_retry_before_deadline(delay=15.0, deadline=115.0) is False
        )  # 100 + 15 >= 115
        assert (
            can_retry_before_deadline(
                delay=2.0, deadline=110.0, estimated_call_duration=5.0
            )
            is True
        )  # 100 + 2 + 5 < 110
        assert (
            can_retry_before_deadline(
                delay=8.0, deadline=110.0, estimated_call_duration=5.0
            )
            is False
        )  # 100 + 8 + 5 >= 110

    with patch("llm_gateway.service.time.monotonic", return_value=120.0):
        assert (
            can_retry_before_deadline(delay=1.0, deadline=115.0) is False
        )  # 120 + 1 >= 115


def test_can_retry_before_deadline_invalid_delay():
    with pytest.raises(ValueError):
        can_retry_before_deadline(delay=-1.0, deadline=100.0)


def test_can_retry_before_deadline_invalid_estimated_call_duration():
    with pytest.raises(ValueError):
        can_retry_before_deadline(
            delay=5.0, deadline=100.0, estimated_call_duration=-1.0
        )


def test_retry_stops_when_deadline_insufficient():
    provider = Mock()

    # Simulate a rate limit error with retry_after
    provider.generate.side_effect = [
        ProviderRateLimitError("Provider rate limit exceeded.", retry_after=10.0),
        "Success",
    ]

    with (
        patch("llm_gateway.service.time.monotonic", return_value=100.0),
        patch("llm_gateway.service.calculate_retry_delay", return_value=2.0),
        patch("llm_gateway.service.time.sleep") as mock_sleep,
        pytest.raises(ProviderRateLimitError),
    ):
        generate_with_retry(provider, "What is Python?", max_attempts=3, deadline=105.0)

    assert provider.generate.call_count == 1  # Should not retry due to deadline
    mock_sleep.assert_not_called()  # No sleep should occur since it won't retry


def test_retry_succeeds_within_deadline():
    provider = Mock()

    # Simulate a rate limit error with retry_after
    provider.generate.side_effect = [
        ProviderRateLimitError("Provider rate limit exceeded.", retry_after=10.0),
        "Success",
    ]

    with (
        patch("llm_gateway.service.time.monotonic", return_value=100.0),
        patch("llm_gateway.service.calculate_retry_delay", return_value=2.0),
        patch("llm_gateway.service.time.sleep") as mock_sleep,
    ):
        result = generate_with_retry(
            provider, "What is Python?", max_attempts=3, deadline=120.0
        )

    assert provider.generate.call_count == 2
    assert result == "Success"
    mock_sleep.assert_called_once_with(10.0)


def test_retry_invalid_estimated_call_duration():
    provider = Mock()

    with pytest.raises(ValueError):
        generate_with_retry(
            provider, "What is Python?", max_attempts=3, estimated_call_duration=-1.0
        )
