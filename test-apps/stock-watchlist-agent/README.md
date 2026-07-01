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

## Running

```bash
export OPENAI_API_KEY="sk-..."
python -m src.main AAPL GOOGL NVDA
```

## Running with Datadog LLM Observability

```bash
export OPENAI_API_KEY="sk-..."
export DD_API_KEY="<your-datadog-api-key>"
export DD_SITE="datadoghq.com"  # your Datadog site

python -m src.main AAPL GOOGL NVDA
```

Traces appear in **Datadog > LLM Observability** under the `stock-watchlist-agent` app.

To override the app name:

```bash
export DD_LLMOBS_ML_APP="my-custom-name"
```

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
