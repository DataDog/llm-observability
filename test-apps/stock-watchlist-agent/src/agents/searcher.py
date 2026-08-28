from __future__ import annotations

import logging

from openai import AsyncOpenAI

log = logging.getLogger(__name__)

_client = AsyncOpenAI()

SEARCH_INSTRUCTIONS = (
    "Search the web for the requested information and return a concise, factual "
    "summary of your findings. Include specific numbers, dates, and sources "
    "where possible. Do not editorialize."
)


async def search(query: str, **kwargs) -> str:
    """Run a web search via OpenAI Responses API and return a summary."""
    log.info("web_search: %s", query if len(query) <= 90 else query[:87] + "...")
    response = await _client.responses.create(
        model="gpt-4o-mini",
        instructions=SEARCH_INSTRUCTIONS,
        input=query,
        tools=[{"type": "web_search"}],
    )
    text = response.output_text
    log.debug("web_search: returned %d chars", len(text))
    return text
