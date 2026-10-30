import asyncio

from llm_gateway.exceptions import ProviderRateLimitError, ProviderTimeoutError
from llm_gateway.service import calculate_retry_delay


async def generate_async(provider, question: str) -> str:
    """
    Asynchronously generate a response from the provider.

    Args:
        provider: The LLM provider instance.
        question (str): The question to ask the model.
    Returns:
        str: The generated response.
    """
    return await provider.generate(question)


async def generate_with_retry_async(
    provider,
    question: str,
    max_attempts: int = 3,
    deadline: float | None = None,
    attempt_timeout: float = 3.0,
) -> str:
    """
    Asynchronously generate a response from the provider.

    Args:
        provider: The LLM provider instance.
        question (str): The question to ask the model.
        max_attempts (int): The maximum number of attempts to make.
        deadline (float | None): Absolute deadline in the event loop's monotonic clock.
        attempt_timeout (float): The maximum duration of each provider attempt, in seconds

    Returns:
        str: The generated response.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1.")

    if attempt_timeout <= 0:
        raise ValueError("attempt_timeout must be greater than 0.")

    if deadline is None:
        deadline = asyncio.get_running_loop().time() + 10.0

    for attempt in range(max_attempts):
        remaining = deadline - asyncio.get_running_loop().time()

        if remaining <= 0:
            raise ProviderTimeoutError("Overall deadline exceeded")

        try:
            timeout_seconds = min(attempt_timeout, remaining)
            return await generate_with_timeout(
                provider, question, timeout_seconds=timeout_seconds
            )
        except (ProviderRateLimitError, ProviderTimeoutError) as error:
            if attempt == max_attempts - 1:
                raise

            delay = calculate_retry_delay(attempt)
            if (
                isinstance(error, ProviderRateLimitError)
                and error.retry_after is not None
            ):
                delay = max(delay, error.retry_after)

            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= delay:
                raise

            await asyncio.sleep(delay)


async def generate_with_timeout(
    provider,
    question: str,
    timeout_seconds: float,
) -> str:
    """Generate a response from the provider, raising if it takes too long."""
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than 0.")

    try:
        async with asyncio.timeout(timeout_seconds):
            return await provider.generate(question)
    except TimeoutError as error:
        raise ProviderTimeoutError("Provider request timed out.") from error
