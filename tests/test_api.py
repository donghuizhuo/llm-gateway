from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from llm_gateway.api import app, get_provider
from llm_gateway.exceptions import ProviderRateLimitError

client = TestClient(app)


def test_rate_limit_endpoint():
    response = client.get("/test-rate-limit")

    assert response.status_code == 429
    assert response.json()["error"]["code"] == "UPSTREAM_RATE_LIMITED"
    assert (
        response.json()["error"]["message"]
        == "Provider rate limit exceeded."
    )


def test_timeout_endpoint():
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

        print(response.status_code)
        print(response.json())

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

            print(response.json())

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
