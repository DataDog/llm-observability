# Testing the `capture --publish` (baseline-with-cost) changes

How to exercise the inline-experiments **capture-publish** flow end to end on this app:
`capture --publish` publishes the capture as the **baseline experiment** (a real run of the
boundary, so it carries real spans + token/dollar cost) plus a Dataset; then
`replay --publish` runs the current code as the **current** experiment over the same dataset
and links a **compare view**.

Feature branches: ddtrace `feat/llmobs-inline-experiments-capture-publish` (PR #18625) and this
app's `stock-watchlist-agent-capture-publish`.

## 1. Put both repos on the capture-publish branches

```bash
git -C ~/dd/dd-trace-py-inline-experiments checkout feat/llmobs-inline-experiments-capture-publish
cd ~/dd/llm-observability/test-apps/stock-watchlist-agent
git checkout stock-watchlist-agent-capture-publish
```
The app's `.venv` editable-installs `ddtrace` from that worktree, so it picks up the CLI live —
no reinstall/overlay needed.

## 2. Config via `.env` (no exports — the command auto-loads `.env`)

```bash
cat > .env <<'ENV'
OPENAI_API_KEY=sk-...
DD_API_KEY=...
DD_SITE=datad0g.com
DD_LLMOBS_ML_APP=stock-watchlist-agent
ENV
echo ".env" >> .gitignore     # never commit secrets
```

## 3. Sanity check — subject registered

```bash
PYTHONPATH=. .venv/bin/python -m ddtrace.commands.ddtrace_experiment list src.agents.orchestrator
# -> portfolio
```

## 4. Capture → publish the baseline (real run → real cost + dataset)

Single case keeps it cheap; use `capture_watchlists` for the full 8.

```bash
PYTHONPATH=. .venv/bin/python -m ddtrace.commands.ddtrace_experiment \
    capture capture_one --publish --project stock-watchlist-agent
```
Verify in LLM Obs: the `inline-portfolio-baseline` experiment has **token/dollar cost** (it ran
the real agent), and the `inline-experiment-portfolio` dataset's records have `expected_output`
populated.

## 5. Make a change, then replay → publish the current run + compare

```bash
# edit src/agents/orchestrator.py:  gpt-4o -> gpt-4o-mini
PYTHONPATH=. .venv/bin/python -m ddtrace.commands.ddtrace_experiment \
    replay src.agents.orchestrator --publish --project stock-watchlist-agent
```
Open the printed **compare** link → baseline vs current over the same dataset, with **cost on
both** and the eval-metric deltas (completeness / sentiment_consistency / factual_grounding).

## 6. (Optional) offline local loop — no backend

```bash
PYTHONPATH=. .venv/bin/python -m ddtrace.commands.ddtrace_experiment \
    capture capture_one:generate_traffic

PYTHONPATH=. .venv/bin/python -m ddtrace.commands.ddtrace_experiment \
    replay src.agents.orchestrator --evaluate
```

## What to watch for (backend-coupled — validate on a live run)

1. `capture --publish` baseline shows **real cost** (the agent runs through the engine).
2. `replay --publish` **reuses the same dataset** via `pull_dataset` — check the capture and
   replay dataset ids match (it silently falls back to creating one from the local JSON if the
   pull fails).
3. `expected_output` **backfill** — the current run's `regression_match` compares against the
   baseline, not an empty value.
4. The **compare** link opens the intended baseline-vs-current view.

## Notes

- Without `--publish`, `capture` stays **local** (a `.llmobs_experiments.json` baseline only —
  no dataset, cost, or compare view).
- `capture --publish` reads `SUBJECT` and `INPUTS` from the driver module (see
  `capture_watchlists.py` / `capture_quotes.py` / `capture_one.py`).
- `capture --publish` supports single-function subjects (both subjects here are).
- Cost/time knobs: `capture_one` = 1 case; `WATCHLISTS="NVDA"` narrows `capture_watchlists`;
  `TICKERS="NVDA AAPL"` narrows `capture_quotes`.
