import asyncio
from unittest.mock import AsyncMock, call, patch

import pytest

from llm_gateway.async_service import generate_async, generate_with_retry_async
from llm_gateway.exceptions import ProviderRateLimitError


@pytest.mark.asyncio
async def test_generate_async_success():
    provider = AsyncMock()
    provider.generate.return_value = "Success"

    result = await generate_async(provider, "What is Python?")

    assert result == "Success"
    provider.generate.assert_awaited_once_with("What is Python?")


@pytest.mark.asyncio
async def test_generate_async_rate_limit():
    provider = AsyncMock()
    provider.generate.side_effect = ProviderRateLimitError(
        "Provider rate limit exceeded."
    )

    with pytest.raises(ProviderRateLimitError) as exc_info:
        await generate_async(provider, "What is Python?")

    assert str(exc_info.value) == "Provider rate limit exceeded."
    provider.generate.assert_awaited_once_with("What is Python?")


@pytest.mark.asyncio
async def test_generate_async_success_after_retry():
    provider = AsyncMock()
    provider.generate.side_effect = [
        ProviderRateLimitError("Provider rate limit exceeded."),
        "Success",
    ]

    with (
        patch(
            "llm_gateway.async_service.calculate_retry_delay", return_value=2.0
        ) as mock_backoff,
        patch(
            "llm_gateway.async_service.asyncio.sleep", new_callable=AsyncMock
        ) as mock_sleep,
    ):
        result = await generate_with_retry_async(
            provider, "What is Python?", max_attempts=2
        )

    assert result == "Success"
    assert provider.generate.await_count == 2
    mock_sleep.assert_awaited_once_with(2.0)
    mock_backoff.assert_called_once_with(
        0
    )  # First attempt is 0, second is 1, so backoff for first retry is 0


@pytest.mark.asyncio
async def test_generate_async_retry_exhaustion():
    provider = AsyncMock()
    provider.generate.side_effect = ProviderRateLimitError(
        "Provider rate limit exceeded."
    )

    with (
        pytest.raises(ProviderRateLimitError),
        patch(
            "llm_gateway.async_service.asyncio.sleep", new_callable=AsyncMock
        ) as mock_sleep,
        patch(
            "llm_gateway.async_service.calculate_retry_delay", return_value=1.0
        ) as mock_backoff,
    ):
        await generate_with_retry_async(provider, "What is Python?", max_attempts=3)

    assert provider.generate.await_count == 3
    assert mock_sleep.await_count == 2

    assert mock_backoff.call_args_list == [
        call(0),
        call(1),
    ]
    assert mock_sleep.await_args_list == [
        call(1.0),
        call(1.0),
    ]


@pytest.mark.asyncio
async def test_generate_async_cancelled_during_backoff():
    provider = AsyncMock()
    provider.generate.side_effect = ProviderRateLimitError(
        "Provider rate limit exceeded."
    )

    with (
        patch(
            "llm_gateway.async_service.calculate_retry_delay", return_value=5.0
        ) as mock_backoff,
        patch(
            "llm_gateway.async_service.asyncio.sleep", new_callable=AsyncMock
        ) as mock_sleep,
        pytest.raises(asyncio.CancelledError),
    ):
        mock_sleep.side_effect = asyncio.CancelledError()
        await generate_with_retry_async(provider, "What is Python?", max_attempts=3)

    assert provider.generate.await_count == 1
    mock_backoff.assert_called_once_with(0)
    mock_sleep.assert_awaited_once_with(5.0)


@pytest.mark.asyncio
async def test_generate_async_retry_respects_retry_after():
    provider = AsyncMock()
    provider.generate.side_effect = [
        ProviderRateLimitError("Provider rate limit exceeded.", retry_after=10.0),
        "Success",
    ]

    with (
        patch(
            "llm_gateway.async_service.calculate_retry_delay", return_value=2.0
        ) as mock_backoff,
        patch(
            "llm_gateway.async_service.asyncio.sleep", new_callable=AsyncMock
        ) as mock_sleep,
    ):
        result = await generate_with_retry_async(
            provider, "What is Python?", max_attempts=3
        )

    assert result == "Success"
    assert provider.generate.await_count == 2
    mock_backoff.assert_called_once_with(0)
    mock_sleep.assert_awaited_once_with(10.0)


@pytest.mark.asyncio
async def test_generate_async_stops_when_deadline_exceeded():
    provider = AsyncMock()
    provider.generate.side_effect = ProviderRateLimitError(
        message="Provider rate limit exceeded.", retry_after=10.0
    )

    with (
        pytest.raises(ProviderRateLimitError) as exc_info,
        patch(
            "llm_gateway.async_service.calculate_retry_delay", return_value=2.0
        ) as mock_backoff,
        patch(
            "llm_gateway.async_service.asyncio.sleep", new_callable=AsyncMock
        ) as mock_sleep,
        patch(
            "llm_gateway.service.time.monotonic", return_value=100.0
        ) as mock_monotonic,
    ):
        await generate_with_retry_async(
            provider,
            "What is Python?",
            max_attempts=3,
            deadline=115.0,
            estimated_call_duration=8.0,
        )

    assert exc_info.value.retry_after == 10.0
    assert provider.generate.await_count == 1
    mock_backoff.assert_called_once_with(0)
    mock_sleep.assert_not_awaited()
    mock_monotonic.assert_called_once_with()


@pytest.mark.asyncio
async def test_generate_async_succeeds_within_deadline():
    provider = AsyncMock()
    provider.generate.side_effect = [
        ProviderRateLimitError("Provider rate limit exceeded.", retry_after=2.0),
        "Success",
    ]

    with (
        patch(
            "llm_gateway.async_service.calculate_retry_delay", return_value=1.0
        ) as mock_backoff,
        patch(
            "llm_gateway.async_service.asyncio.sleep", new_callable=AsyncMock
        ) as mock_sleep,
        patch(
            "llm_gateway.service.time.monotonic", return_value=100.0
        ) as mock_monotonic,
    ):
        result = await generate_with_retry_async(
            provider,
            "What is Python?",
            max_attempts=3,
            deadline=110.0,
            estimated_call_duration=5.0,
        )

    assert result == "Success"
    assert provider.generate.await_count == 2
    mock_backoff.assert_called_once_with(0)
    mock_sleep.assert_awaited_once_with(2.0)
    mock_monotonic.assert_called_once_with()
