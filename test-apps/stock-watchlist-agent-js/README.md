# Stock Watchlist Agent JS

JavaScript translation of `test-apps/stock-watchlist-agent`, built with OpenAI's Responses API and instrumented with `dd-trace-js` LLM Observability.

## Architecture

```
llmobs.trace(kind="agent", name="analyze_portfolio")        ← evals attach here
└── orchestrator (OpenAI Responses ReAct loop)
    ├── delegate_research (tool, batched tickers)
    │   └── stock_researcher (OpenAI Responses ReAct loop)
    │       ├── get_stock_quote (tool) → OpenAI web search
    │       ├── search_company_news (tool) → OpenAI web search
    │       ├── search_public_sentiment (tool) → OpenAI web search
    │       └── get_company_profile (tool) → OpenAI web search
    └── delegate_research (tool, second batch — parallel)
        └── stock_researcher ...
```

The orchestrator plans how to batch tickers by sector/theme, delegates batches to researcher agents, and synthesizes a portfolio briefing. Each researcher runs a multi-step ReAct loop with four research tools backed by OpenAI web search. Post-run evaluations (completeness, sentiment consistency, factual grounding) are submitted to LLM Observability.

## Development

```bash
cd test-apps/stock-watchlist-agent-js
npm install
cp .env.example .env
# edit .env with OPENAI_API_KEY and optional Datadog settings
```

Requires Node.js 20+.

If you see `Error: Cannot find module 'dd-trace'`, `Cannot find module 'openai'`, or `Cannot find module 'dotenv'`, run `npm install` from this directory.

To test a local `dd-trace-js` checkout instead of the published package:

```bash
npm install /path/to/dd-trace-js/packages/dd-trace
```

## Running

```bash
# Either export OPENAI_API_KEY or set it in .env
npm start -- AAPL GOOGL NVDA
```

## Running with Datadog LLM Observability

```bash
# Either export these variables or set them in .env
export OPENAI_API_KEY="sk-..."
export DD_API_KEY="<your-datadog-api-key>"
export DD_SITE="datadoghq.com"  # your Datadog site

npm start -- AAPL GOOGL NVDA
```

Example `.env`:

```dotenv
OPENAI_API_KEY=sk-...
DD_API_KEY=<your-datadog-api-key>
DD_SITE=datadoghq.com
DD_LLMOBS_ML_APP=stock-watchlist-agent-js
DD_LLMOBS_AGENTLESS_ENABLED=true
```

For Datadog internal/staging credentials, you can populate Datadog keys with:

```bash
dd-auth --output --domain dd.datad0g.com >> .env
```

Then confirm `.env` still contains:

```dotenv
DD_LLMOBS_ML_APP=stock-watchlist-agent-js
DD_LLMOBS_AGENTLESS_ENABLED=true
```

Traces appear in **Datadog > LLM Observability** under the `stock-watchlist-agent-js` app.

To override the app name:

```bash
export DD_LLMOBS_ML_APP="my-custom-name"
```

The app loads environment variables from `.env` before initializing `dd-trace`, initializes `dd-trace` before loading `openai`, enables `DD_LLMOBS_ENABLED` automatically when `DD_API_KEY` is present, and sets `DD_LLMOBS_AGENTLESS_ENABLED=1` unless you override it. It also sets standalone LLMObs mode (`DD_APM_TRACING_ENABLED=false` by default) so the CLI does not require a local Datadog Agent on `127.0.0.1:8126`.

## Evaluations

When LLMObs is enabled, three evaluations run after each analysis and are submitted to the root `analyze_portfolio` agent span:

| Eval | Type | Description |
|------|------|-------------|
| `completeness` | boolean | All requested tickers present in output |
| `sentiment_consistency` | boolean (LLM judge) | Sentiment labels match analysis narratives |
| `factual_grounding` | score 1-5 (LLM judge) | Analyses cite specific numbers, dates, events |

## Project Structure

```
src/
├── main.js                    # CLI entry point, eval runner
├── observability.js           # .env loading, dd-trace-js initialization, LLMObs helpers
├── models.js                  # JSON schemas + runtime validation
├── evals.js                   # Evaluators + LLMObs submitEvaluation calls
└── agents/
    ├── orchestrator.js        # ReAct orchestrator, delegation tool, agent span
    ├── researcher.js          # Per-batch research agent with 4 tools
    ├── responses-agent.js     # Generic Responses API function-calling loop
    └── searcher.js            # OpenAI Responses API web search helper
```
