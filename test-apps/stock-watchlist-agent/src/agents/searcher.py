from __future__ import annotations

from openai import AsyncOpenAI

_client = AsyncOpenAI()

SEARCH_INSTRUCTIONS = (
    "Search the web for the requested information and return a concise, factual "
    "summary of your findings. Include specific numbers, dates, and sources "
    "where possible. Do not editorialize."
)


async def search(query: str, **kwargs) -> str:
    """Run a web search via OpenAI Responses API and return a summary."""
    response = await _client.responses.create(
        model="gpt-4o-mini",
        instructions=SEARCH_INSTRUCTIONS,
        input=query,
        tools=[{"type": "web_search"}],
    )
    return response.output_text
