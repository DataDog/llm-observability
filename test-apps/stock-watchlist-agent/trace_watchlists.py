#!/usr/bin/env python3
"""trace_watchlists.py — drive the watchlist set with LLM Obs enabled to emit traces.

Reuses the same ``generate_traffic`` driver as the inline-experiment baseline
(``capture_watchlists``), but enables LLM Observability first, so every ``analyze_portfolio``
call becomes an agent trace in Datadog. Use it to populate LLM Obs with traces for the
exact watchlists your experiment captures/replays over.

Prereqs (real, paid OpenAI calls; ~30s-2min per watchlist):
    Set OPENAI_API_KEY (required), and DD_API_KEY + DD_SITE to emit traces, in .env
    (see .env.example) — loaded automatically via sitecustomize.py.

Usage:
    python trace_watchlists.py                         # default 8 watchlists
    WATCHLISTS="NVDA" python trace_watchlists.py        # one watchlist (cheap)
    WATCHLISTS="NVDA AMD|AAPL JPM XOM" python trace_watchlists.py   # '|'-separated lists
    DD_LLMOBS_ML_APP=stock-watchlist-agent-dev python trace_watchlists.py
"""
from __future__ import annotations

import asyncio
import os
import sys

from ddtrace.llmobs import LLMObs

from capture_watchlists import generate_traffic


def main() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set — the agent cannot run.", file=sys.stderr)
        sys.exit(1)

    ml_app = os.environ.get("DD_LLMOBS_ML_APP", "stock-watchlist-agent")
    if os.environ.get("DD_API_KEY"):
        # Same enablement as src/main.py — agentless so no local agent is required.
        LLMObs.enable(ml_app=ml_app, agentless_enabled=True)
        print(f"LLM Obs enabled (ml_app: {ml_app}) — traces will be sent to Datadog.")
    else:
        print("DD_API_KEY not set — running WITHOUT tracing (no traces will be sent).")

    try:
        asyncio.run(generate_traffic())
    finally:
        # Ensure spans are flushed before the process exits.
        try:
            LLMObs.flush()
        except Exception:
            pass


if __name__ == "__main__":
    main()
