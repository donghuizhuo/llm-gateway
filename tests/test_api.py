from unittest.mock import AsyncMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from llm_gateway.api import app, get_http_client, get_provider
from llm_gateway.exceptions import ProviderRateLimitError


def test_rate_limit_endpoint():
    client = TestClient(app)
    response = client.get("/test-rate-limit")

    assert response.status_code == 429
    assert response.json()["error"]["code"] == "UPSTREAM_RATE_LIMITED"
    assert response.json()["error"]["message"] == "Provider rate limit exceeded."


def test_timeout_endpoint():
    client = TestClient(app)
    response = client.get("/test-timeout")

    assert response.status_code == 504
    assert response.json()["error"]["code"] == "UPSTREAM_TIMEOUT"
    assert (
        response.json()["error"]["message"]
        == "The LLM provider did not respond in time."
    )


def test_ask_success():
    mock_provider = AsyncMock()
    mock_provider.generate.return_value = "Hello from the mock LLM!"
    app.dependency_overrides[get_provider] = lambda: mock_provider

    client = TestClient(app)
    try:
        response = client.post("/ask", json={"question": "What is Python?"})

        assert response.status_code == 200
        assert response.json() == {"answer": "Hello from the mock LLM!"}
        mock_provider.generate.assert_awaited_once_with("What is Python?")
    finally:
        app.dependency_overrides.pop(get_provider, None)


def test_ask_rate_limit():
    mock_provider = AsyncMock()
    mock_provider.generate.side_effect = ProviderRateLimitError(
        "Provider rate limit exceeded."
    )
    app.dependency_overrides[get_provider] = lambda: mock_provider

    try:
        client = TestClient(app)

        with patch("llm_gateway.async_service.asyncio.sleep", new_callable=AsyncMock):
            response = client.post("/ask", json={"question": "What is Python?"})

            assert response.status_code == 429
            assert response.json() == {
                "error": {
                    "code": "UPSTREAM_RATE_LIMITED",
                    "message": "Provider rate limit exceeded.",
                }
            }
            assert mock_provider.generate.await_count == 3
    finally:
        app.dependency_overrides.pop(get_provider, None)


def test_http_client_lifecycle():
    with TestClient(app) as client:
        response = client.get("/client-status")
        assert response.status_code == 200
        assert response.json().get("is_closed") is False

        shared_client = app.state.http_client
        assert shared_client.is_closed is False

    assert shared_client.is_closed is True


def test_http_client_identity():
    with TestClient(app) as client:
        response1 = client.get("/client-status")
        response2 = client.get("/client-status")
        assert response1.status_code == 200
        assert response2.status_code == 200

        data1 = response1.json()
        data2 = response2.json()

        assert data1["is_closed"] is False
        assert data2["is_closed"] is False
        assert data1["client_id"] == data2["client_id"]

        shared_client = app.state.http_client
        assert data1["client_id"] == id(shared_client)

    assert shared_client.is_closed is True


@pytest.mark.asyncio
async def test_ask_http_integration():
    def mock_handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "POST"
        assert str(request.url) == "https://example.com/generate"

        return httpx.Response(
            status_code = 200,
            json={"answer": "Hello from the upstream LLM!"}
        )

    transport = httpx.MockTransport(mock_handler)

    async with httpx.AsyncClient(transport=transport) as mock_client:
        app.dependency_overrides[get_http_client] = lambda: mock_client

        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://testserver",
            ) as test_client:
                response = await test_client.post("/ask", json={"question": "What is Python?"})

                assert response.status_code == 200
                assert response.json() == {"answer": "Hello from the upstream LLM!"}
        finally:
            app.dependency_overrides.pop(get_http_client, None)

@pytest.mark.asyncio
async def test_ask_http_retry_integration():
    attempts = 0

    def mock_handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts

        assert request.method == "POST"
        assert str(request.url) == "https://example.com/generate"

        attempts += 1
        if attempts <= 2:
            return httpx.Response(
                status_code = 429,
                headers={"Retry-After": "0.01"}
            )
        
        return httpx.Response(
            status_code = 200,
            json={"answer": "Success after retries"}
        )

    transport = httpx.MockTransport(mock_handler)

    async with httpx.AsyncClient(transport=transport) as mock_client:
        app.dependency_overrides[get_http_client] = lambda: mock_client

        try:
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app),
                base_url="http://testserver",
            ) as test_client:
                response = await test_client.post("/ask", json={"question": "What is Python?"})

                assert response.status_code == 200
                assert response.json() == {"answer": "Success after retries"}
        finally:
            app.dependency_overrides.pop(get_http_client, None)

    assert attempts == 3