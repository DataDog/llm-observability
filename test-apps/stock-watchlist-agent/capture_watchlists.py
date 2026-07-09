"""Capture driver for the inline experiment.

The experiment boundary itself is marked **in the application source** — see the
``@experiment_start`` decorator on ``analyze_portfolio`` in
``src/agents/orchestrator.py``. That decorator is a no-op during normal runs and only
activates under the ``ddtrace-experiment`` command.

This module is just a harness: it imports the real (decorated) ``analyze_portfolio``
and exercises it over a representative set of watchlists. Each call becomes one
captured case (one ``PortfolioBriefing``).

    # record a baseline of the current behavior (one case per watchlist)
    ddtrace-experiment capture capture_watchlists:generate_traffic

    # ...with traces sent to LLM Obs, each case linked to its real orchestrator span
    ddtrace-experiment capture capture_watchlists:generate_traffic --trace --ml-app stock-watchlist-agent

    # publish the capture as the baseline experiment (real run -> real cost) + a dataset:
    ddtrace-experiment capture capture_watchlists --publish --project stock-watchlist-agent

    WATCHLISTS="NVDA AMD|AAPL JPM XOM"   # optional override: '|'-separated lists
"""
import os

from src.agents.orchestrator import analyze_portfolio


DEFAULT_WATCHLISTS = [
    ["NVDA"],                          # single mega-cap
    ["TSLA"],                          # single mega-cap, different sector (auto/EV)
    ["NVDA", "AMD"],                   # same-sector pair (semis) -> one research batch
    ["V", "MA"],                       # same-sector pair (payments)
    ["AAPL", "JPM", "XOM"],            # cross-sector trio -> multiple themed batches
    ["AAPL", "MSFT", "GOOGL", "AMZN", "META"],  # big-tech basket (5) -> larger fan-out
    ["SPY"],                           # broad-market ETF (non-equity ticker)
    ["XLE"],                           # sector ETF (energy)
]


def _watchlists() -> list[list[str]]:
    env = os.environ.get("WATCHLISTS")
    return [s.split() for s in env.split("|")] if env else DEFAULT_WATCHLISTS


# `--publish` reads these: SUBJECT names the experiment; INPUTS are the boundary's kwargs
# (analyze_portfolio(tickers=...)), one dataset record each.
SUBJECT = "portfolio"
INPUTS = [{"tickers": w} for w in _watchlists()]


async def generate_traffic():  # local capture (no --publish): exercise the boundary directly
    for tickers in _watchlists():
        await analyze_portfolio(tickers)
