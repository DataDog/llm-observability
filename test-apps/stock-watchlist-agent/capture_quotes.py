"""Capture driver scoped to the stock-quote tool.

The experiment boundary is the ``@experiment_start(name="stock_quote")`` on
``stock_quote_lookup`` in ``src/agents/researcher.py``. This harness records one
``(ticker -> quote)`` case per ticker by calling that lookup **directly** — no agent,
no orchestrator — so the experiment stays limited to just the quote function.

    # capture a baseline of the quote tool (one case per ticker)
    ddtrace-experiment capture capture_quotes:generate_traffic

    # replay scoped to ONLY the quote tool (importing researcher registers `stock_quote`
    # but not `portfolio`, so the orchestrator experiment is skipped)
    ddtrace-experiment replay src.agents.researcher --comparator structural

    TICKERS="NVDA AAPL SPY"   # optional override (space-separated)
"""
import os

from src.agents.researcher import stock_quote_lookup


DEFAULT_TICKERS = ["NVDA", "AAPL", "MSFT", "SPY", "XOM"]


async def generate_traffic():
    env = os.environ.get("TICKERS")
    tickers = env.split() if env else DEFAULT_TICKERS
    for ticker in tickers:
        await stock_quote_lookup(ticker)
