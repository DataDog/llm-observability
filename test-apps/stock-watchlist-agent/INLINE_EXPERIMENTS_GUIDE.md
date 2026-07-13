# Inline Experiments on the Stock Watchlist Agent — quickstart

A hands-on runbook for **inline experiments** ("a unit test for LLM apps") on a real agent.
The `stock-watchlist-agent` batches tickers, delegates to researcher agents (OpenAI web
search), and synthesizes a `PortfolioBriefing`. The boundary under test is marked in the app
source (`src/agents/orchestrator.py`, `name="portfolio"`):

```python
@experiment_start(name="portfolio", inputs=["tickers"],
                  output=lambda ret: ret[0].model_dump(),   # compare the briefing dict
                  trace_link=lambda ret: ret[1],            # published runs link each case to its span
                  evaluators=_portfolio_evaluators)         # richer checks (Step 3)
async def analyze_portfolio(tickers): ...
```

It's a pure no-op in normal execution — it only activates under `ddtrace-experiment run`.

---

## Setup (one time)

```bash
cd test-apps/stock-watchlist-agent
python -m venv .venv && .venv/bin/pip install -e .

# credentials (or put them in .env — copy .env.example; it's loaded automatically)
export OPENAI_API_KEY=sk-...                    # required — the agent makes real web-search calls
export DD_API_KEY=...  DD_SITE=datadoghq.com    # only for --publish

# convenience + keep cost down while learning (1 watchlist instead of the default 8)
exp() { PYTHONPATH=. .venv/bin/ddtrace-experiment "$@"; }
export WATCHLISTS="NVDA"
```

> **Requires a `ddtrace` version that includes inline experiments** — the `ddtrace-experiment`
> CLI plus `ddtrace.llmobs.experimental` (`experiment_start` / `experiment_end` / `comparison`).
> `run` executes the **real, paid** agent (minutes per case) — `WATCHLISTS="NVDA"` keeps it to one.

---

## The loop — one verb, `run`

```bash
exp run capture_watchlists:generate_traffic     # 1. baseline current behavior -> .llmobs_experiments.json
exp list capture_watchlists                     #    confirm the boundary registered -> portfolio
# 2. change a model / prompt / tool / schema
exp run src.agents.orchestrator                 # 3. structural check (default) -> match / changed
exp run src.agents.orchestrator --evaluate      # 3b. + completeness / sentiment / grounding
exp run src.agents.orchestrator --publish --project stock-watchlist-agent   # 4. UI + eval metrics + compare view
```

`run` is offline until you add `--publish`. The first `--publish` is the frozen baseline
experiment (real cost); each later `--publish` is compared against it via a printed compare link.

**Reading the output.** Each case shows a `run` status (`OK` = produced an output to judge;
else `ERROR` / `NO_END`) and one verdict line per evaluator. The structural check is just the
default evaluator, printed as `regression_match` with a **`match`** / **`changed`** assessment:

```
  run   input                 recorded             new
  OK    {"tickers":["NVDA"]}  recent_news present  recent_news missing
      regression_match      changed
  OK=1  CHANGED=1
# exit code 1
```

`run` exits non-zero on `changed`, `ERROR`/`NO_END`, or (with `--evaluate`) any evaluator
that fails/errors — so it gates CI.

---

## Pick what "changed" means

The default evaluator is a comparison; `--comparator` / `--ignore` configure it (all report
`match`/`changed`):

| Question                    | Flag                     | Catches                                                   | Tolerates                               |
| --------------------------- | ------------------------ | --------------------------------------------------------- | --------------------------------------- |
| Shape/coverage changed?     | `structural` _(default)_ | dropped/added/renamed fields, fewer tickers, type changes | reworded text, new timestamps           |
| Anything but volatile keys? | `--ignore generated_at`  | value changes outside the ignored keys                    | the listed keys                         |
| Byte-identical?             | `exact`                  | everything                                                | nothing (always `changed` for LLM text) |

---

## Go beyond structure: evaluators

`structural` can't see a flipped `sentiment` or a weakly-grounded analysis. The boundary
attaches evaluators (in `src/evals.py`, wired via a lazy thunk so credentialed judges aren't
built in a prod import). `--evaluate` scores them locally (opt-in — the judges make paid
calls); `--publish` always runs them as one eval metric each.

| Evaluator               | Catches                                                                                              | Needs            |
| ----------------------- | ---------------------------------------------------------------------------------------------------- | ---------------- |
| `completeness`          | a requested ticker missing from `analyses`                                                           | offline          |
| `sentiment_consistency` | a `sentiment` label contradicting its narrative (incl. the bullish→bearish flip `structural` misses) | `OPENAI_API_KEY` |
| `factual_grounding`     | analyses drifting toward vague, ungrounded prose                                                     | `OPENAI_API_KEY` |

```bash
exp run src.agents.orchestrator --evaluate
#   OK    {"tickers":["NVDA"]}  ...
#       regression_match        match      # structure preserved...
#       sentiment_consistency   fail       # ...but the label contradicts the narrative
#   OK=1  EVAL_FAIL=1
```

You can also drop a built-in comparison into `evaluators=` via `comparison(...)` — so "a
comparison" and "an evaluator" are the same surface.

---

## Common changes (what to expect)

| You change…                                            | Expect                                                                                        |
| ------------------------------------------------------ | --------------------------------------------------------------------------------------------- |
| Orchestrator model (`gpt-4o → gpt-4o-mini`)            | all `match` → ship the saving; any `changed` → the mini model dropped coverage, inspect       |
| Prompt wording                                         | `match` (structure preserved); `changed` if it stops emitting a field like `highlights`       |
| Add a `risk_rating` field (intentional)                | `changed` everywhere — expected; re-run with `--record` to reset the baseline                 |
| Researcher stops populating `recent_news` (regression) | `changed` on affected cases — caught before it ships                                          |
| Model flips a `sentiment` value                        | `regression_match: match` (shape intact) but `sentiment_consistency: fail` under `--evaluate` |

---

## CI

```yaml
# pre-merge job (needs a committed baseline + OPENAI_API_KEY secret)
- run: |
    PYTHONPATH=. python -m ddtrace.commands.ddtrace_experiment \
      run src.agents.orchestrator --evaluate
```

Drop `--evaluate` for a fast, offline, structure-only gate.

---

## Notes

- **Re-baseline after intended changes** — an accepted change _should_ report `changed`; re-run with `--record`.
- **Cost** — every `run` executes the real agent; `--evaluate` adds judge calls. Keep `WATCHLISTS` small.
- **`--publish` writes to the backend** (Dataset + Experiment; needs `DD_API_KEY`); the first publish is the frozen baseline and carries real cost. The LLM judges also call OpenAI.
- **Full-trace export on `--publish`** requires a recent `ddtrace` (older versions can fail to serialize some provider-specific span metadata).
- **Serializable inputs only** (`tickers`); live infra goes through `inputs=[...]` + a `fixtures=` provider
