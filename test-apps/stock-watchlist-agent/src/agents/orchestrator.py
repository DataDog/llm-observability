from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from ddtrace import tracer
from ddtrace.llmobs import LLMObs
from ddtrace.llmobs.decorators import tool as llmobs_tool
from ddtrace.llmobs.experimental import experiment_start
from pydantic_ai import Agent, RunContext, Tool

from src.agents.researcher import stock_researcher
from src.models import PortfolioBriefing


log = logging.getLogger(__name__)


# --- Delegation tool ---


@llmobs_tool(name="delegate_research")
async def _delegate_research(ctx: RunContext, tickers: list[str]) -> str:
    """Delegate a batch of stock tickers to a research agent."""
    log.info("delegate_research: dispatching research batch for %d ticker(s): %s", len(tickers), ", ".join(tickers))
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


def _portfolio_evaluators() -> list[Any]:
    """Lazily import the evaluators defined in ``src.evals``.

    Deferred (a zero-arg thunk, like ``fixtures``) so the LLM judges — which carry
    provider config — are constructed only when an activated runner resolves them
    (`run --evaluate` or `run --publish`), never in a production import of this module.
    """
    from src.evals import portfolio_evaluators

    return portfolio_evaluators()


# Mark this input->output boundary as an inline-experiment subject. This is a pure
# no-op in normal execution (prod or local) — it only activates under the
# `ddtrace-experiment run` command, which records (tickers -> briefing) baselines and
# re-runs the current code against them. `trace_link` points the case at this call's real
# orchestrator span (returned as the second element below) so a published run links each
# case to its actual trace instead of wrapping it in a synthetic span. `evaluators`
# attaches richer checks (completeness + two LLM judges) that score each case alongside
# the structural comparator — locally via `run --evaluate` and as eval metrics on
# `run --publish`.
@experiment_start(
    name="portfolio",
    inputs=["tickers"],
    output=lambda ret: ret[0].model_dump(),
    trace_link=lambda ret: ret[1],
    evaluators=_portfolio_evaluators,
)
async def analyze_portfolio(tickers: list[str]) -> tuple[PortfolioBriefing, dict[str, Any] | None]:
    """Analyze multiple stocks via delegated research agents and synthesize a briefing.

    Returns (briefing, span_context) where span_context is the orchestrator agent's
    span_id/trace_id for attaching evaluations, or None if LLMObs is disabled.
    """
    log.info("analyze_portfolio: start — %d ticker(s): %s", len(tickers), ", ".join(tickers))
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    prompt = f"Analyze these stock tickers: {', '.join(tickers)}. Current time: {now}"

    # Use orchestrator.iter() so we can capture the auto-instrumented agent span.
    # LLMObs.export_span() can't find it via its own context provider in async code,
    # but tracer.current_span() CAN — the span is activated with activate=True in
    # the ddtrace integration. We pass it explicitly to export_span(span=...).
    span_context = None
    async with orchestrator.iter(prompt) as agent_run:
        try:
            span = tracer.current_span()
            if span:
                span_context = LLMObs.export_span(span=span)
        except Exception:
            pass

        async for _node in agent_run:
            pass

    briefing = agent_run.result.output
    log.info(
        "analyze_portfolio: done — %d analysis(es), %d highlight(s)",
        len(briefing.analyses),
        len(briefing.highlights),
    )
    return briefing, span_context
