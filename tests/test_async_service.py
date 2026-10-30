import asyncio
from unittest.mock import AsyncMock, call, patch

import pytest

from llm_gateway.async_service import (
    generate_async,
    generate_with_retry_async,
    generate_with_timeout,
)
from llm_gateway.exceptions import ProviderRateLimitError, ProviderTimeoutError


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
        deadline = asyncio.get_running_loop().time() + 30.0
        result = await generate_with_retry_async(
            provider, "What is Python?", deadline=deadline, max_attempts=3
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
    ):
        deadline = asyncio.get_running_loop().time() + 5.0
        await generate_with_retry_async(
            provider,
            question="What is Python?",
            max_attempts=3,
            deadline=deadline,
            attempt_timeout=8.0,
        )

    assert exc_info.value.retry_after == 10.0
    provider.generate.assert_awaited_once_with("What is Python?")
    mock_backoff.assert_called_once_with(0)
    mock_sleep.assert_not_awaited()


@pytest.mark.asyncio
async def test_generate_async_succeeds_within_deadline():
    provider = AsyncMock()
    provider.generate.side_effect = [
        ProviderRateLimitError("Provider rate limit exceeded.", retry_after=2.0),
        "Success",
    ]

    loop = asyncio.get_running_loop()

    with (
        patch(
            "llm_gateway.async_service.calculate_retry_delay", return_value=1.0
        ) as mock_backoff,
        patch(
            "llm_gateway.async_service.asyncio.sleep", new_callable=AsyncMock
        ) as mock_sleep,
        patch.object(loop, "time", return_value=100.0),
    ):
        result = await generate_with_retry_async(
            provider,
            "What is Python?",
            max_attempts=3,
            deadline=110.0,
            attempt_timeout=5.0,
        )

    assert result == "Success"
    assert provider.generate.await_count == 2
    mock_backoff.assert_called_once_with(0)
    mock_sleep.assert_awaited_once_with(2.0)


@pytest.mark.asyncio
async def test_generate_with_timeout_success():
    provider = AsyncMock()
    provider.generate.return_value = "Success"

    result = await generate_with_timeout(
        provider, question="What is Python?", timeout_seconds=30
    )
    assert result == "Success"
    provider.generate.assert_awaited_once_with("What is Python?")


@pytest.mark.asyncio
@pytest.mark.parametrize("timeout_seconds", [0, -1])
async def test_generate_with_timeout_invalid_timeout(timeout_seconds):
    provider = AsyncMock()

    with pytest.raises(ValueError):
        await generate_with_timeout(
            provider, question="What is Python?", timeout_seconds=timeout_seconds
        )


@pytest.mark.asyncio
async def test_generate_with_timeout_expired():
    class SlowProvider:
        async def generate(self, question: str) -> str:
            await asyncio.sleep(1)
            return "Too late"

    provider = SlowProvider()
    with pytest.raises(ProviderTimeoutError) as exc_info:
        await generate_with_timeout(
            provider, question="What is Python?", timeout_seconds=0.1
        )

    assert isinstance(exc_info.value.__cause__, TimeoutError)


@pytest.mark.asyncio
async def test_generate_with_retry_async_timeout_then_success():
    class MockProvider:
        def __init__(self):
            self.call_count = 0

        async def generate(self, question: str) -> str:
            self.call_count += 1

            if self.call_count == 1:
                await asyncio.sleep(1)
                return "Too late"

            return "Success"

    provider = MockProvider()
    result = await generate_with_retry_async(
        provider, question="What is Python?", max_attempts=2, attempt_timeout=0.01
    )

    assert result == "Success"
    assert provider.call_count == 2


@pytest.mark.asyncio
async def test_generate_with_retry_async_expired_deadline():
    provider = AsyncMock()

    deadline = asyncio.get_running_loop().time() - 1

    with pytest.raises(ProviderTimeoutError):
        await generate_with_retry_async(
            provider, question="What is Python?", max_attempts=3, deadline=deadline
        )

    provider.generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_generate_with_retry_async_deadline_during_call():
    class MockProvider:
        def __init__(self):
            self.call_count = 0

        async def generate(self, question: str) -> str:
            self.call_count += 1
            await asyncio.sleep(1)
            return "Success"

    provider = MockProvider()
    deadline = asyncio.get_running_loop().time() + 0.05
    with pytest.raises(ProviderTimeoutError):
        await generate_with_retry_async(
            provider,
            question="What is Python?",
            max_attempts=3,
            deadline=deadline,
            attempt_timeout=3.0,
        )

    assert provider.call_count == 1


@pytest.mark.asyncio
async def test_generate_with_retry_async_external_cancellation():
    class MockProvider:
        def __init__(self):
            self.started = asyncio.Event()
            self.call_count = 0

        async def generate(self, question: str) -> str:
            self.call_count += 1
            self.started.set()

            await asyncio.sleep(10)
            return "Success"

    provider = MockProvider()
    task = asyncio.create_task(
        generate_with_retry_async(
            provider, question="What is Python?", max_attempts=3, attempt_timeout=5.0
        )
    )
    await provider.started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    assert provider.call_count == 1
