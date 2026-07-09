"""Single-case capture driver — the smallest end-to-end inline-experiment example.

The experiment boundary is marked in the app source (``@experiment_start`` on
``analyze_portfolio`` in ``src/agents/orchestrator.py``). This harness exercises it
**exactly once**, so ``capture`` records a single baseline case — then ``replay --publish``
turns that one case into a Dataset + Experiment in LLM Observability.

    # 1a. local capture ONE case -> .llmobs_experiments.json (offline; no dataset/cost)
    ddtrace-experiment capture capture_one:generate_traffic

    # 1b. OR publish the capture as the baseline experiment (real run -> real cost) + dataset
    ddtrace-experiment capture capture_one --publish --project stock-watchlist-agent

    # 2. replay the current code against it (local, offline): match / changed
    ddtrace-experiment replay src.agents.orchestrator

    # 3. publish the current run + compare view vs the baseline (needs DD_API_KEY)
    ddtrace-experiment replay src.agents.orchestrator --publish --project stock-watchlist-agent
"""
from src.agents.orchestrator import analyze_portfolio


# `--publish` reads these: SUBJECT names the experiment; INPUTS is the single boundary call.
SUBJECT = "portfolio"
INPUTS = [{"tickers": ["NVDA"]}]


async def generate_traffic():  # local capture: one watchlist -> one (tickers -> briefing) case
    await analyze_portfolio(["NVDA"])
