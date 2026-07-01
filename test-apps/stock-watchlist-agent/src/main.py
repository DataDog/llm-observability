from __future__ import annotations

import argparse
import asyncio
import os
import sys

from ddtrace.llmobs import LLMObs

from src.agents.orchestrator import analyze_portfolio
from src.evals import run_evaluations


def print_briefing(briefing) -> None:
    """Pretty-print a PortfolioBriefing to stdout."""
    print("\n" + "=" * 60)
    print("  STOCK WATCHLIST BRIEFING")
    print("=" * 60)
    print(f"\nGenerated: {briefing.generated_at}")
    print(f"\n{'─' * 60}")
    print("MARKET OVERVIEW")
    print(f"{'─' * 60}")
    print(briefing.market_overview)

    for analysis in briefing.analyses:
        print(f"\n{'─' * 60}")
        print(f"  {analysis.ticker} ({analysis.company_name})")
        print(f"  {analysis.current_price}  ({analysis.price_change})")
        print(f"  Sentiment: {analysis.sentiment.upper()}")
        print(f"{'─' * 60}")
        print(f"\n{analysis.summary}\n")
        print("Key Factors:")
        for factor in analysis.key_factors:
            print(f"  * {factor}")
        print("\nRecent News:")
        for news in analysis.recent_news:
            print(f"  - {news}")
        print("\nPublic Sentiment:")
        print(f"  {analysis.public_sentiment_summary}")

    print(f"\n{'─' * 60}")
    print("HIGHLIGHTS")
    print(f"{'─' * 60}")
    for highlight in briefing.highlights:
        print(f"  >> {highlight}")

    print("\n" + "=" * 60 + "\n")


async def main(tickers: list[str]) -> None:
    llmobs_enabled = bool(os.environ.get("DD_API_KEY"))
    if llmobs_enabled:
        LLMObs.enable(
            ml_app=os.environ.get("DD_LLMOBS_ML_APP", "stock-watchlist-agent"),
            agentless_enabled=True,
        )

    print(f"Analyzing {len(tickers)} ticker(s): {', '.join(tickers)}")
    print("Running parallel analysis with web search...\n")

    briefing, span_context = await analyze_portfolio(tickers)
    print_briefing(briefing)

    if llmobs_enabled and span_context:
        print("Running evaluations...")
        run_evaluations(briefing, tickers, span_context)
        print("Evaluations submitted to LLM Observability.\n")


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="Stock Watchlist Analyst - AI-powered parallel stock research"
    )
    parser.add_argument(
        "tickers",
        nargs="+",
        type=str,
        help="Stock ticker symbols to analyze (e.g., AAPL GOOGL MSFT)",
    )
    args = parser.parse_args()

    tickers = [t.upper() for t in args.tickers]

    if not tickers:
        print("Error: provide at least one ticker symbol", file=sys.stderr)
        sys.exit(1)

    asyncio.run(main(tickers))


if __name__ == "__main__":
    cli()
