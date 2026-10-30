from contextlib import asynccontextmanager
from typing import Annotated

import httpx
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from llm_gateway.async_service import generate_with_retry_async
from llm_gateway.exceptions import ProviderRateLimitError, ProviderTimeoutError
from llm_gateway.http_provider import HTTPProvider
from llm_gateway.models import AskRequest


@asynccontextmanager
async def lifespan(app: FastAPI):
    async with httpx.AsyncClient() as client:
        app.state.http_client = client
        yield


app = FastAPI(
    title="LLM Gateway",
    version="0.1.0",
    lifespan=lifespan,
    description="A gateway for LLM providers.",
)


def get_http_client(request: Request) -> httpx.AsyncClient:
    return request.app.state.http_client


@app.get("/client-status")
async def client_status(client: Annotated[httpx.AsyncClient, Depends(get_http_client)]):
    return {"is_closed": client.is_closed, "client_id": id(client)}


@app.get("/test-rate-limit", tags=["Test"])
async def test_rate_limit():
    raise ProviderRateLimitError("Provider rate limit exceeded.")


@app.get("/test-timeout", tags=["Test"])
async def test_timeout():
    raise ProviderTimeoutError("Provider request timed out.")


@app.exception_handler(ProviderRateLimitError)
async def provider_rate_limit_exception_handler(
    request: Request, exc: ProviderRateLimitError
) -> JSONResponse:
    return JSONResponse(
        status_code=429,
        content={
            "error": {
                "code": "UPSTREAM_RATE_LIMITED",
                "message": str(exc) or "The LLM provider is temporarily rate limited.",
            }
        },
    )


@app.exception_handler(ProviderTimeoutError)
async def provider_timeout_exception_handler(
    request: Request, exc: ProviderTimeoutError
) -> JSONResponse:
    return JSONResponse(
        status_code=504,
        content={
            "error": {
                "code": "UPSTREAM_TIMEOUT",
                "message": "The LLM provider did not respond in time.",
            }
        },
    )


class FakeProvider:
    async def generate(self, question: str) -> str:
        return f"Fake answer to: {question}"


def get_provider(
        client: Annotated[httpx.AsyncClient, Depends(get_http_client)]
    ) -> HTTPProvider:
    return HTTPProvider(
        client=client,
        url="https://example.com/generate"
    )


@app.post("/ask")
async def ask(
    request: AskRequest,
    provider: Annotated[HTTPProvider, Depends(get_provider)]
):
    answer = await generate_with_retry_async(
        provider=provider,
        question=request.question,
        max_attempts=3,
        attempt_timeout=3
    )
    return {"answer": answer}


@app.get("/query")
async def query():
    async with httpx.AsyncClient() as client:
        response = await client.get("https://example.com")
        return {"result": response.text}
