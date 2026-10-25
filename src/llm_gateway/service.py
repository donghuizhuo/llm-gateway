import random
import time

from llm_gateway.exceptions import ProviderRateLimitError


def generate_with_retry(
    provider,
    question: str,
    max_attempts: int = 3,
    deadline: float | None = None,
    estimated_call_duration: float = 0.0,
) -> str:
    """
    Attempt to generate a response from the provider with retry logic.

    Args:
        provider: The LLM provider instance.
        question (str): The question to ask the model.
        max_attempts (int): Maximum number of provider calls, including the initial attempt.
        deadline (float | None): The absolute deadline expressed using time.monotonic(), in seconds.
        estimated_call_duration (float): The estimated duration of the API call in seconds.

    Returns:
        str: The generated response from the provider.

    Raises:
        ProviderRateLimitError: If the provider's rate limit is exceeded after retries.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1.")

    if estimated_call_duration < 0:
        raise ValueError("estimated_call_duration must be non-negative.")

    for attempt in range(max_attempts):
        try:
            return provider.generate(question)
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

            time.sleep(delay)  # Wait before retrying


def calculate_retry_delay(
    retry_number: int, base_delay: float = 1.0, max_delay: float = 30.0
) -> float:
    """
    Calculate the delay before the next retry attempt using exponential backoff.

    Args:
        retry_number (int): The current retry number (0-indexed).
        base_delay (float): The base delay in seconds.
        max_delay (float): The maximum delay in seconds.

    Returns:
        float: The calculated delay in seconds.
    """
    if retry_number < 0:
        raise ValueError("retry_number must be non-negative.")

    if base_delay <= 0:
        raise ValueError("base_delay must be positive.")

    if max_delay <= 0:
        raise ValueError("max_delay must be positive.")

    upper_bound = min(base_delay * (2**retry_number), max_delay)
    return random.uniform(0, upper_bound)


def can_retry_before_deadline(
    delay: float, deadline: float, estimated_call_duration: float = 0.0
) -> bool:
    """
    Determine if a retry can be attempted before the given deadline.

    Args:
        delay (float): The delay for the next retry attempt.
        deadline (float): The absolute deadline expressed using time.monotonic(), in seconds.
        estimated_call_duration (float): The estimated duration of the API call in seconds.

    Returns:
        bool: True if a retry can be attempted before the deadline, False otherwise.
    """
    if delay < 0:
        raise ValueError("delay must be non-negative.")

    if estimated_call_duration < 0:
        raise ValueError("estimated_call_duration must be non-negative.")

    return time.monotonic() + delay + estimated_call_duration < deadline
