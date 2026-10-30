import httpx
import pytest


@pytest.mark.asyncio
async def test_httpx_mock_transport_success():
    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200, json={"answer": "Python is a programming language"}
        )

    transport = httpx.MockTransport(mock_handler)

    async with httpx.AsyncClient(transport=transport) as client:
        response = await client.get("https://example.com")

    assert response.status_code == 200
    assert response.json() == {"answer": "Python is a programming language"}


@pytest.mark.asyncio
async def test_httpx_mock_transport_rate_limit():
    def mock_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=429,
            headers={"Retry-After": "5"},
            json={"message": "Request rate limited"},
        )

    transport = httpx.MockTransport(mock_handler)

    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        async with httpx.AsyncClient(transport=transport) as client:
            response = await client.get("https://example.com")
            response.raise_for_status()

    assert exc_info.value.response.status_code == 429
    assert exc_info.value.response.headers["Retry-After"] == "5"
