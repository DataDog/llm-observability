# Stock Watchlist Agent JS

JavaScript translation of `test-apps/stock-watchlist-agent`, built with OpenAI's Responses API and instrumented with `dd-trace-js` LLM Observability.

## Architecture

```
llmobs.trace(kind="agent", name="analyze_portfolio")        ← evals attach here
├── resolve_tickers_from_images (workflow)                 ← only when image inputs are given
│   └── identify_ticker (llm, one per image) → OpenAI vision call, image_parts annotated
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

Inputs can be ticker symbols, images, or a mix of both. When images are provided, a vision step runs first inside the root `analyze_portfolio` span: one LLM call per image identifies the public company shown and returns its ticker symbol, and the resolved tickers are merged with any tickers passed directly. Images that cannot be matched to a public company come back as `UNKNOWN` and are skipped. Image resolution and research therefore share a single trace.

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

# a) All three inputs are ticker symbols
npm start -- AAPL GOOGL NVDA

# b) First two inputs are images, third is a ticker symbol
npm start -- logos/apple.png logos/google.png NVDA

# Image URLs work too, and --image forces an argument to be read as an image
npm start -- https://example.com/nvidia-logo.jpg --image logos/apple.png
```

The `logos/` directory holds small sample wordmark images for Apple, Google, and NVIDIA so the image path can be run without supplying your own files.

Arguments ending in `.png`, `.jpg`, `.jpeg`, `.gif`, or `.webp`, plus any `http(s)://` or `data:image/...` argument, are treated as images automatically. Use `--image <path|url>` to force an argument to be read as an image. Every image is read into memory as base64 (URLs are downloaded first) so the same bytes can be sent to OpenAI and attached to the trace. Set `OPENAI_VISION_MODEL` to override the model used for image-to-ticker translation (defaults to `OPENAI_MODEL`, then `gpt-5.4-nano`).

### Images on spans

`identify_ticker` is annotated as an `llm`-kind span whose user message carries `imageParts: [{ mimeType, content }]`, which the SDK emits as `image_parts: [{ mime_type, content }]`.

To see an image input in Datadog, open the trace in **LLM Observability > Traces** and select the span named **`identify_ticker`**.

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
logos/                         # Small sample images for the image-input path
src/
├── main.js                    # CLI entry point, eval runner
├── observability.js           # .env loading, dd-trace-js initialization, LLMObs helpers
├── models.js                  # JSON schemas + runtime validation
├── evals.js                   # Evaluators + LLMObs submitEvaluation calls
└── agents/
    ├── orchestrator.js        # ReAct orchestrator, delegation tool, agent span
    ├── researcher.js          # Per-batch research agent with 4 tools
    ├── responses-agent.js     # Generic Responses API function-calling loop
    ├── searcher.js            # OpenAI Responses API web search helper
    └── vision.js              # Image → ticker symbol translation
```
