# Stock Watchlist Agent

Multi-agent stock research tool built with [Pydantic AI](https://ai.pydantic.dev/) and OpenAI's Responses API. An example app for exploring Datadog LLM Observability instrumentation.

## Architecture

```
@llmobs_agent("orchestrator")        ← evals attach here
└── orchestrator (Pydantic AI Agent, ReAct loop)
    ├── delegate_research (tool, batched tickers)
    │   └── stock_researcher (Pydantic AI Agent, ReAct loop)
    │       ├── get_stock_quote (tool) → OpenAI web search
    │       ├── search_company_news (tool) → OpenAI web search
    │       ├── search_public_sentiment (tool) → OpenAI web search
    │       └── get_company_profile (tool) → OpenAI web search
    └── delegate_research (tool, second batch — parallel)
        └── stock_researcher ...
```

The orchestrator plans how to batch tickers by sector/theme, delegates batches to researcher agents, and synthesizes a portfolio briefing. Each researcher runs a multi-step ReAct loop with 4 research tools backed by OpenAI web search. Post-run evaluations (completeness, sentiment consistency, factual grounding) are submitted to LLM Observability.

## Development

```bash
cd stock-watchlist-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Requires Python 3.11+.

## Credentials

Copy `.env.example` to `.env` and fill in real values — `.env` is gitignored and is
loaded automatically for every entrypoint (`src.main`, `capture_*.py`,
`trace_scenarios.py`, `ddtrace-experiment`) via `sitecustomize.py`.

```bash
cp .env.example .env
# edit .env: set OPENAI_API_KEY, and DD_API_KEY/DD_SITE if you want LLM Obs traces
```

## Running

```bash
python -m src.main AAPL GOOGL NVDA
```

## Running with Datadog LLM Observability

Set `DD_API_KEY` and `DD_SITE` in `.env` (see above), then run as usual:

```bash
python -m src.main AAPL GOOGL NVDA
```

Traces appear in **Datadog > LLM Observability** under the `stock-watchlist-agent` app.

To override the app name, set `DD_LLMOBS_ML_APP` in `.env`.

## Evaluations

When LLMObs is enabled, three evaluations run after each analysis and are submitted to the orchestrator span:

| Eval | Type | Description |
|------|------|-------------|
| `completeness` | boolean | All requested tickers present in output |
| `sentiment_consistency` | boolean (LLM judge) | Sentiment labels match analysis narratives |
| `factual_grounding` | score 1-5 (LLM judge) | Analyses cite specific numbers, dates, events |

## Project Structure

```
src/
├── main.py                 # CLI entry point, LLMObs init, eval runner
├── models.py               # Pydantic models (StockAnalysis, PortfolioBriefing)
├── evals.py                # Evaluators (BaseEvaluator, LLMJudge)
└── agents/
    ├── orchestrator.py     # ReAct orchestrator, delegation tool, @llmobs_agent
    ├── researcher.py       # Per-batch research agent with 4 tools
    └── searcher.py         # OpenAI Responses API web search helper
```
