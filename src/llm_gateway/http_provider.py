import math

import httpx

from llm_gateway.exceptions import ProviderRateLimitError, ProviderTimeoutError


def parse_retry_after(response: httpx.Response) -> float | None:
    header = response.headers.get("Retry-After")

    if header is None:
        return None

    try:
        retry_after = float(header)
        if not math.isfinite(retry_after) or retry_after < 0:
            retry_after = None
    except ValueError:
        retry_after = None

    return retry_after


class HTTPProvider:
    def __init__(self, client: httpx.AsyncClient, url: str):
        self.client = client
        self.url = url

    async def generate(self, question: str) -> str:
        try:
            response = await self.client.post(url=self.url, json={"question": question})
            response.raise_for_status()
            data = response.json()
            return data["answer"]
        except httpx.HTTPStatusError as exc:
            if exc.response.status_code == 429:
                retry_after = parse_retry_after(exc.response)

                raise ProviderRateLimitError(
                    message="Request is rate limited", retry_after=retry_after
                ) from exc
            raise
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError("Upstream LLM provider timed out") from exc
