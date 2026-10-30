import json

import httpx
import pytest

from llm_gateway.async_service import generate_with_retry_async
from llm_gateway.exceptions import ProviderRateLimitError, ProviderTimeoutError
from llm_gateway.http_provider import HTTPProvider, parse_retry_after


@pytest.mark.parametrize(
    "header_value, expected",
    [
        ("5", 5.0),
        ("0", 0.0),
        ("0.5", 0.5),
        ("-5", None),
        ("invalid", None),
        ("nan", None),
        ("inf", None),
        (None, None),
    ],
)
def test_parse_retry_after(header_value, expected):
    response = httpx.Response(
        status_code=429,
        headers={"Retry-After": header_value} if header_value is not None else {},
        json={"message": "Request rate limited"},
    )

    retry_after = parse_retry_after(response)
    assert retry_after == expected


@pytest.mark.asyncio
async def test_http_provider_success():
    def mock_handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url) == "https://example.com/generate"
        assert json.loads(request.content) == {"question": "What is Python?"}
        assert request.headers["Content-Type"] == "application/json"

        return httpx.Response(
            status_code=200, json={"answer": "Python is a programming language."}
        )

    transport = httpx.MockTransport(mock_handler)

    async with httpx.AsyncClient(transport=transport) as client:
        provider = HTTPProvider(client=client, url="https://example.com/generate")

        answer = await provider.generate("What is Python?")

    assert answer == "Python is a programming language."


@pytest.mark.asyncio
async def test_http_provider_rate_limit():
    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=429,
            headers={"Retry-After": "5"},
            json={"message": "Rate limit exceeded"},
        )

    transport = httpx.MockTransport(mock_handler)

    async with httpx.AsyncClient(transport=transport) as client:
        provider = HTTPProvider(client=client, url="https://example.com/generate")

        with pytest.raises(ProviderRateLimitError) as exc_info:
            await provider.generate(question="What is Python?")

        assert exc_info.value.retry_after == 5.0
        assert isinstance(exc_info.value.__cause__, httpx.HTTPStatusError)


@pytest.mark.asyncio
async def test_http_provider_timeout():
    def mock_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Read timed out", request=request)

    transport = httpx.MockTransport(mock_handler)

    async with httpx.AsyncClient(transport=transport) as client:
        provider = HTTPProvider(
            client=client,
            url="https://example.com/generate",
        )

        with pytest.raises(ProviderTimeoutError) as exc:
            await provider.generate(question="What is Python?")

        assert isinstance(exc.value.__cause__, httpx.ReadTimeout)


@pytest.mark.asyncio
async def test_http_provider_with_retry():
    attempts = 0

    def mock_handler(requst: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1

        if attempts == 1:
            return httpx.Response(status_code=429, headers={"Retry-After": "0.01"})

        return httpx.Response(status_code=200, json={"answer": "Success"})

    transport = httpx.MockTransport(mock_handler)

    async with httpx.AsyncClient(transport=transport) as client:
        provider = HTTPProvider(client=client, url="https://example.com/generate")

        answer = await generate_with_retry_async(
            provider=provider,
            question="What is Python?",
            max_attempts=3,
            attempt_timeout=3,
        )

    assert answer == "Success"
    assert attempts == 2
