# Testing the inline-experiments `run` flow

How to exercise inline experiments end to end on this app. There is **one verb, `run`**.
It is **offline by default** (records a baseline, then re-runs the current code against it with
a CI-friendly exit code — no backend). Add **`--publish`** to send it to LLM Obs Experiments:
the first publish is the frozen **baseline** experiment (a real run, so it carries spans + cost
+ eval metrics) over an auto-managed dataset; every later publish is compared against it in a
**compare view**. The dataset is refreshed to exactly each run's inputs.

Feature branches: ddtrace `feat/llmobs-inline-experiments-run` and this app's
`stock-watchlist-agent-run`.

## 1. Put both repos on the `run` branches

```bash
git -C ~/dd/dd-trace-py-inline-experiments checkout feat/llmobs-inline-experiments-run
cd ~/dd/llm-observability/test-apps/stock-watchlist-agent
git checkout stock-watchlist-agent-run
```
The app's `.venv` editable-installs `ddtrace` from that worktree, so it picks up the CLI live —
no reinstall/overlay needed. Keep the worktree on this branch while testing.

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

## 4. Offline loop — no backend, fast, CI-friendly exit code

```bash
# first run: records a baseline of the current behavior (one case)
PYTHONPATH=. .venv/bin/python -m ddtrace.commands.ddtrace_experiment run capture_one:generate_traffic

# edit src/agents/orchestrator.py (e.g. gpt-4o -> gpt-4o-mini), then re-run to compare:
PYTHONPATH=. .venv/bin/python -m ddtrace.commands.ddtrace_experiment run src.agents.orchestrator --evaluate
# per-case MATCH/CHANGED + evaluator verdicts; exit 1 if anything regressed.
```

## 5. Publish — the first run is the frozen baseline (real cost)

Single case keeps it cheap; use `capture_watchlists` for the full 8.

```bash
PYTHONPATH=. .venv/bin/python -m ddtrace.commands.ddtrace_experiment \
    run capture_one --publish --project stock-watchlist-agent
```
This is the **baseline** — it prints the dataset (`inline-experiment-portfolio`, synced to this
run's inputs) and a "run again to compare" prompt. Verify in LLM Obs that the experiment carries
**token/dollar cost** (it ran the real agent) and the subject's eval metrics (completeness /
sentiment_consistency / factual_grounding).

## 6. Change the code, publish again → compare view

```bash
# edit src/agents/orchestrator.py:  gpt-4o -> gpt-4o-mini
PYTHONPATH=. .venv/bin/python -m ddtrace.commands.ddtrace_experiment \
    run src.agents.orchestrator --publish --project stock-watchlist-agent
```
Open the printed **compare** link → the frozen baseline vs this run over the same dataset, with
**cost on both** and the eval-metric deltas. Publish again after each change; every run is
compared against the same frozen baseline.

## What to watch for (backend-coupled — validate on a live run)

1. The first `run --publish` experiment shows **real cost** (the agent runs through the engine).
2. The dataset `inline-experiment-portfolio` holds **exactly this run's inputs** (diff-synced):
   change the inputs and confirm removed cases disappear and new ones are added, while unchanged
   cases keep aligning in the compare view.
3. The **baseline stays frozen** across several `run --publish` iterations (same baseline id in
   the compare link); `--record` forces a fresh baseline.
4. The **compare** link opens the intended baseline-vs-current view.

## Notes

- Without `--publish`, `run` stays **local** (a `.llmobs_experiments.json` baseline only — no
  dataset, cost, or compare view) and sets a CI-friendly exit code.
- `run --publish` reads this run's inputs from the module's `INPUTS` (edit it to change the test
  set), falling back to a prior offline baseline. `SUBJECT` names the subject when a module
  registers more than one (see `capture_watchlists.py` / `capture_quotes.py` / `capture_one.py`).
- Both subjects here are single-function (return-based), so no `experiment_end` is needed.
- Cost/time knobs: `capture_one` = 1 case; `WATCHLISTS="NVDA"` narrows `capture_watchlists`;
  `TICKERS="NVDA AAPL"` narrows `capture_quotes`.

## Demo knobs

- **Quiet the logs.** The app logs at INFO by default (`analyze_portfolio: start`, tool calls,
  etc.). For a clean demo, set the level via env — no code change:
  ```bash
  export LOG_LEVEL=WARNING     # or ERROR; put it in .env to make it automatic
  ```
- **Full comparison to a file.** The terminal report is a compact, truncated summary. To read
  the complete per-case diff (full input / recorded / new + every evaluator verdict + reasoning),
  add `--report`:
  ```bash
  exp run src.agents.orchestrator --evaluate --report        # -> .llmobs_experiments.report.json
  jq '.portfolio.cases[0]' .llmobs_experiments.report.json   # inspect one case, untruncated
  ```
