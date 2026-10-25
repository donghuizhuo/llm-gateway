from typing import Annotated

from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from llm_gateway.async_service import generate_with_retry_async
from llm_gateway.exceptions import ProviderRateLimitError, ProviderTimeoutError
from llm_gateway.models import AskRequest

app = FastAPI(
    title="LLM Gateway", version="0.1.0", description="A gateway for LLM providers."
)


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


def get_provider():
    return FakeProvider()


@app.post("/ask")
async def ask(
    question: AskRequest, provider: Annotated[FakeProvider, Depends(get_provider)]
):
    result = await generate_with_retry_async(provider, question.question)
    return {"answer": result}
