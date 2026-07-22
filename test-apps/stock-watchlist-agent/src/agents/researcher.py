from __future__ import annotations

import logging

from ddtrace.llmobs.decorators import tool as llmobs_tool
from ddtrace.llmobs.experimental import experiment_start
from pydantic_ai import Agent, RunContext, Tool

from src.agents.searcher import search
from src.models import ResearchBatchResult


log = logging.getLogger(__name__)


# --- Tool implementations ---


# Inline-experiment subject scoped to JUST the stock-quote lookup.
# It returns its output, so `experiment_start` collects it on return — no
# `experiment_end` is needed (that marker is only for outputs emitted mid-flow rather
# than returned). We decorate this plain helper instead of the pydantic_ai Tool
# function so the tool's RunContext injection stays intact and replay is a clean,
# ctx-free `stock_quote_lookup(ticker=...)` call. Inert outside `ddtrace-experiment`.
@experiment_start(name="stock_quote", inputs=["ticker"], output=lambda ret: ret)
async def stock_quote_lookup(ticker: str) -> str:
    log.info("tool get_stock_quote: %s", ticker)
    return await search(f"{ticker} stock price today current quote market data")


@llmobs_tool(name="get_stock_quote")
async def _get_stock_quote(ctx: RunContext, ticker: str) -> str:
    return await stock_quote_lookup(ticker)


@llmobs_tool(name="search_company_news")
async def _search_company_news(ctx: RunContext, query: str) -> str:
    log.info("tool search_company_news: %s", query)
    return await search(query)


@llmobs_tool(name="search_public_sentiment")
async def _search_public_sentiment(ctx: RunContext, query: str) -> str:
    log.info("tool search_public_sentiment: %s", query)
    return await search(f"{query} investor sentiment Reddit discussion forum opinions")


@llmobs_tool(name="get_company_profile")
async def _get_company_profile(ctx: RunContext, ticker: str) -> str:
    log.info("tool get_company_profile: %s", ticker)
    return await search(f"{ticker} company profile overview market cap sector fundamentals key metrics 2026")


# --- Tool objects ---

get_stock_quote = Tool(
    _get_stock_quote,
    name="get_stock_quote",
    description="Get the current stock price, daily price change, and key trading data for a ticker symbol.",
)

search_company_news = Tool(
    _search_company_news,
    name="search_company_news",
    description=(
        "Search for recent company news and developments. You craft the search query — "
        "be specific about what you're looking for (e.g. 'NVIDIA Q4 2026 earnings results' "
        "or 'Apple AI features Siri delay')."
    ),
)

search_public_sentiment = Tool(
    _search_public_sentiment,
    name="search_public_sentiment",
    description=(
        "Search for public investor sentiment and discussions about a stock or company. "
        "Look for Reddit, social media, and forum discussions to gauge retail investor mood."
    ),
)

get_company_profile = Tool(
    _get_company_profile,
    name="get_company_profile",
    description=(
        "Get a company overview including sector, market cap, business description, "
        "recent performance, and key financial metrics."
    ),
)


# --- Stock researcher agent ---

RESEARCHER_PROMPT = """\
You are a stock research analyst conducting thorough research on your assigned batch of stocks.

<scope>
You research stock tickers using available tools, conducting multi-step analysis \
for each stock in your assigned batch. You must produce a StockAnalysis for every \
ticker you are assigned.
</scope>

<approach>
Think step-by-step before taking action:
1. Plan your research approach for the assigned tickers
2. Get price quotes to establish current market positions
3. Search for recent news with specific, targeted queries — craft them carefully
4. Check public investor sentiment on Reddit, forums, social media
5. Get company profiles for fundamental context
6. After each round, reflect: did something interesting surface? Do you need to dig deeper?
7. When satisfied, produce a structured StockAnalysis for each ticker
</approach>

<tools>
get_stock_quote — Current price and daily change for a ticker.
search_company_news — Search recent news. You craft the query — be specific.
search_public_sentiment — Gauge retail investor mood from Reddit/forums.
get_company_profile — Company overview, sector, market cap, fundamentals.
</tools>

<examples>
Assigned: ["AAPL", "NVDA"]

Thought: Two tickers. I'll start by getting current prices for both, then research news for each.
Action: [calls get_stock_quote("AAPL")]
Action: [calls get_stock_quote("NVDA")]

Result: AAPL at $262.01 (-1.66%), NVDA at $197.58 (-0.65%)

Thought: Both are slightly down. Let me search for what's driving each.
Action: [calls search_company_news("Apple Q2 2026 earnings preview iPhone")]
Action: [calls search_company_news("NVIDIA AI data center revenue 2026")]

Result: Apple earnings coming April 30, NVIDIA reporting record AI revenue.

Thought: Interesting — Apple has earnings soon. Let me check sentiment and get NVIDIA's profile.
Action: [calls search_public_sentiment("AAPL Apple earnings April 2026")]
Action: [calls get_company_profile("NVDA")]

Result: [sentiment and profile data]

Thought: I have enough to produce analyses for both stocks.
[Returns ResearchBatchResult with analyses for AAPL and NVDA]
</examples>

<output>
Be specific — cite numbers, dates, and concrete facts. Avoid vague generalities.
Every StockAnalysis must have all fields populated with real data from your research.
</output>"""

stock_researcher = Agent(
    "openai-responses:gpt-4o",
    name="stock_researcher",
    output_type=ResearchBatchResult,
    tools=[get_stock_quote, search_company_news, search_public_sentiment, get_company_profile],
    system_prompt=RESEARCHER_PROMPT,
)
