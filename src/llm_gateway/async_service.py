import asyncio

from llm_gateway.exceptions import ProviderRateLimitError
from llm_gateway.service import calculate_retry_delay, can_retry_before_deadline


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
    estimated_call_duration: float = 0.0,
) -> str:
    """
    Asynchronously generate a response from the provider.

    Args:
        provider: The LLM provider instance.
        question (str): The question to ask the model.
        max_attempts (int): The maximum number of attempts to make.
        deadline (float | None): The absolute deadline expressed using time.monotonic(), in seconds.
        estimated_call_duration (float): The estimated duration of the API call in seconds.

    Returns:
        str: The generated response.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1.")

    if estimated_call_duration < 0:
        raise ValueError("estimated_call_duration must be non-negative.")

    for attempt in range(max_attempts):
        try:
            return await provider.generate(question)
        except ProviderRateLimitError as error:
            if attempt == max_attempts - 1:
                raise

            delay = calculate_retry_delay(attempt)
            if error.retry_after is not None:
                delay = max(delay, error.retry_after)

            if deadline is not None and not can_retry_before_deadline(
                delay, deadline, estimated_call_duration
            ):
                raise

            await asyncio.sleep(delay)
