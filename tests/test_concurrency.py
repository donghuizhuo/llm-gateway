import asyncio
from unittest.mock import AsyncMock

import pytest


class MockProvider:
    def __init__(self, name: str, delay: float):
        self.name = name
        self.delay = delay

    async def generate(self, question: str) -> str:
        await asyncio.sleep(self.delay)
        return f"{self.name}: {question}"


async def query_all_providers(providers, question: str) -> list[str]:
    coroutines = [provider.generate(question) for provider in providers]

    results = await asyncio.gather(*coroutines)

    return results


async def query_all_providers_with_partial_results(
    providers, question: str
) -> list[str]:
    coroutines = [provider.generate(question) for provider in providers]
    return_values = await asyncio.gather(*coroutines, return_exceptions=True)
    return [r for r in return_values if not isinstance(r, BaseException)]


async def query_all_providers_bounded(
    providers, question: str, max_concurrency: int = 20
) -> list[str]:
    if max_concurrency <= 0:
        raise ValueError("max_concurrency must be greater than 0.")

    semaphore = asyncio.Semaphore(max_concurrency)

    async def query_one_provider(provider):
        async with semaphore:
            return await provider.generate(question)

    coroutines = [query_one_provider(provider) for provider in providers]

    result = await asyncio.gather(*coroutines, return_exceptions=True)
    return result


@pytest.mark.asyncio
async def test_query_all_providers():
    providers = [
        MockProvider("OpenAI", 0.1),
        MockProvider("Anthropic", 0.2),
        MockProvider("Gemini", 0.3),
    ]

    loop = asyncio.get_running_loop()
    start = loop.time()

    results = await query_all_providers(providers=providers, question="What is Python?")

    elapsed = loop.time() - start
    assert results[0] == "OpenAI: What is Python?"
    assert results[1] == "Anthropic: What is Python?"
    assert results[2] == "Gemini: What is Python?"
    assert elapsed < 0.4


@pytest.mark.asyncio
async def test_query_all_providers_with_partial_results():
    class ProviderA:
        async def generate(self, question: str) -> str:
            await asyncio.sleep(0.1)
            return "A"

    class ProviderB:
        async def generate(self, question: str) -> str:
            await asyncio.sleep(0.2)
            raise ConnectionError("Provider B failed")

    class ProviderC:
        async def generate(self, question: str) -> str:
            await asyncio.sleep(0.3)
            return "C"

    results = await query_all_providers_with_partial_results(
        providers=[ProviderA(), ProviderB(), ProviderC()], question="What is Python?"
    )

    assert results == ["A", "C"]


@pytest.mark.asyncio
async def test_query_all_providers_bounded():
    active_calls = 0
    max_active_calls = 0

    class MockProvider:
        async def generate(self, question: str) -> str:
            nonlocal active_calls, max_active_calls

            active_calls += 1
            max_active_calls = max(max_active_calls, active_calls)

            try:
                await asyncio.sleep(0.1)
                return "Success"
            finally:
                active_calls -= 1

    providers = [
        MockProvider(),
        MockProvider(),
        MockProvider(),
        MockProvider(),
        MockProvider(),
    ]
    result = await query_all_providers_bounded(
        providers=providers, question="What is Python?", max_concurrency=2
    )

    assert result == ["Success", "Success", "Success", "Success", "Success"]
    assert active_calls == 0
    assert max_active_calls == 2


@pytest.mark.parametrize("max_concurrency", [0, -1])
@pytest.mark.asyncio
async def test_query_all_providers_bounded_invalid_concurrency(max_concurrency):
    providers = [AsyncMock()]
    with pytest.raises(ValueError):
        await query_all_providers_bounded(
            providers, question="What is Python?", max_concurrency=max_concurrency
        )

    providers[0].generate.assert_not_awaited()
