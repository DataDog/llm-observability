from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ddtrace import tracer
from ddtrace.llmobs import LLMObs
from ddtrace.llmobs.decorators import agent as llmobs_agent
from ddtrace.llmobs.decorators import tool as llmobs_tool
from pydantic_ai import Agent, RunContext, Tool

from src.agents.researcher import stock_researcher
from src.models import PortfolioBriefing


# --- Delegation tool ---


@llmobs_tool(name="delegate_research")
async def _delegate_research(ctx: RunContext, tickers: list[str]) -> str:
    """Delegate a batch of stock tickers to a research agent."""
    result = await stock_researcher.run(
        f"Research these stocks: {', '.join(tickers)}. "
        f"Get current prices, search for recent news and developments, "
        f"check public investor sentiment, and gather company fundamentals. "
        f"Dig deeper if something interesting surfaces. "
        f"Produce a StockAnalysis for each ticker.",
        usage=ctx.usage,
    )
    return result.output.model_dump_json()


delegate_research = Tool(
    _delegate_research,
    name="delegate_research",
    description=(
        "Delegate a batch of stock tickers to a specialized research agent. "
        "The agent conducts multi-step research (price, news, sentiment, fundamentals) "
        "for every ticker in the batch and returns structured StockAnalysis results. "
        "Group related stocks together (same sector, similar themes) for focused research. "
        "You may call this multiple times with different batches — they run in parallel."
    ),
)


# --- Orchestrator agent ---

ORCHESTRATOR_PROMPT = """\
<scope>
You are a senior portfolio analyst who plans research strategy, delegates work to \
specialized research agents, and synthesizes results into a portfolio briefing.
</scope>

<approach>
Think step-by-step before taking action:
1. Review the tickers provided and plan how to batch them efficiently
2. Group related stocks together (same sector, similar themes) for focused research
3. Delegate each batch to a research agent via delegate_research
4. Review the results returned from each batch and identify cross-cutting themes
5. Synthesize everything into a comprehensive PortfolioBriefing
</approach>

<tools>
delegate_research — Delegate a batch of stock tickers to a research agent. \
The agent conducts multi-step research (price, news, sentiment, fundamentals) \
for every ticker in the batch. Group related stocks together for efficiency. \
You may call this multiple times with different batches — they run in parallel.
</tools>

<examples>
User: "Analyze AAPL, GOOGL, NVDA, MSFT, TSLA, AMZN"

Thought: 6 tickers across tech. I should batch by similarity:
- AAPL, MSFT, GOOGL, AMZN: Big tech / cloud platforms
- NVDA, TSLA: Semiconductors + EVs — smaller batch, different growth drivers

Action: [calls delegate_research(tickers=["AAPL", "MSFT", "GOOGL", "AMZN"])]
Action: [calls delegate_research(tickers=["NVDA", "TSLA"])]

Result: Research results for both batches returned.

Thought: I now have analyses for all 6 stocks. Key cross-cutting themes:
- Big tech investing heavily in AI infrastructure
- Semiconductor demand remains strong driven by AI
- Mixed sentiment on EV market but Tesla innovating

[Synthesizes into PortfolioBriefing]

---

User: "Analyze AAPL, NVDA"

Thought: Just 2 tickers, both tech but different subsectors. A single batch is most efficient.

Action: [calls delegate_research(tickers=["AAPL", "NVDA"])]

Result: Research for both stocks returned.

Thought: Apple focused on consumer tech, NVIDIA on AI infrastructure. \
Different growth drivers but both benefit from AI tailwinds. Now I'll synthesize.

[Produces PortfolioBriefing]
</examples>

<output>
Be concise and actionable. Cite specific numbers and dates from the research.
Focus on cross-cutting themes and what matters most to an investor reviewing their watchlist.
</output>"""

orchestrator = Agent(
    "openai-responses:gpt-4o",
    name="orchestrator",
    output_type=PortfolioBriefing,
    tools=[delegate_research],
    system_prompt=ORCHESTRATOR_PROMPT,
)


@llmobs_agent(name="analyze_portfolio")
async def analyze_portfolio(tickers: list[str]) -> tuple[PortfolioBriefing, dict[str, Any] | None]:
    """Analyze multiple stocks via delegated research agents and synthesize a briefing.

    Returns (briefing, span_context) where span_context is the root agent span's
    span_id/trace_id for attaching evaluations, or None if LLMObs is disabled.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    prompt = f"Analyze these stock tickers: {', '.join(tickers)}. Current time: {now}"

    # Capture the root span (this analyze_portfolio agent span) before entering
    # the orchestrator — all nested pydantic-ai and @llmobs_tool spans will be
    # children of this root, giving us one trace per conversation.
    span_context = None
    span = None
    try:
        span = tracer.current_span()
        if span:
            span_context = LLMObs.export_span(span=span)
    except Exception:
        pass

    async with orchestrator.iter(prompt) as agent_run:
        async for _node in agent_run:
            pass

    briefing = agent_run.result.output
    # Record this run's (input, output) as a self-contained "capture" case on the root span, so the
    # trace is self-describing for replay (annotate = production capture, seeded from prod instead of
    # a local baseline file). We stamp the EXTRACTED output (the briefing), not the native span
    # output — which is the (briefing, span_context) tuple. `replay_input` duplicates the clean arg
    # @llmobs_agent already captures, but keeping the whole case in metadata makes it uniform to read.
    try:
        if span is not None:
            LLMObs.annotate(span=span, metadata={
                "replay_input": {"tickers": tickers},
                "replay_output": briefing.model_dump(),
            })
    except Exception:
        pass

    return briefing, span_context
