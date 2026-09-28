# Stock Watchlist Agent — Eve

An [Eve](https://www.npmjs.com/package/eve)-based stock-watchlist agent for exercising Datadog LLM Observability and Datadog Experiments. It mirrors the Python and JavaScript stock-watchlist apps in this repository while using Eve's schema-v4 instrumentation provider layout.

## Architecture

```text
Eve agent
├── runtime instrumentation
│   ├── schema-v4 invoke_agent / agent.step / chat spans
│   └── Datadog OTLP trace export
├── analyze_watchlist tool
│   ├── orchestrator Responses agent
│   ├── delegated stock research
│   └── structured PortfolioBriefing validation
└── Eve eval runner
    ├── behavior, surface, live, and judge evals
    └── optional Datadog Experiment reporter
```

The top-level agent exposes these stock-research tools:

- `analyze_watchlist`
- `delegate_research`
- `get_stock_quote`
- `search_company_news`
- `search_public_sentiment`
- `get_company_profile`

## Setup

Requires Node.js 24.x.

```bash
cd test-apps/stock-watchlist-agent-eve
npm install
cp .env.example .env.local
```

Set an OpenAI key for the stock-research tools:

```bash
OPENAI_API_KEY="sk-..."
```

For the Eve model through Vercel AI Gateway, either set `AI_GATEWAY_API_KEY` or run:

```bash
npm exec -- eve link
```

Never commit `.env.local`.

## Run locally

```bash
npm run dev
```

Example request:

```text
Analyze this watchlist: AAPL, NVDA, MSFT. Skip evaluations and keep the answer concise.
```

## Datadog runtime telemetry

Enable Datadog export in `.env.local`:

```bash
DD_API_KEY="<datadog-api-key>"
DD_SITE="datadoghq.com"
DD_LLMOBS_ML_APP="stock-watchlist-agent-eve"
DD_LLMOBS_ENABLED=true
DD_LLMOBS_AGENTLESS_ENABLED=true
```

Eve automatically discovers the instrumentation provider layout:

- `agent/instrumentation/otel.ts` owns process-wide resource and content-capture policy.
- `agent/instrumentation/datadog.ts` exports Eve's schema-v4 spans directly to Datadog's OTLP trace intake.
- `agent/lib/stock-watchlist/observability.ts` retains application-authored `dd-trace` LLMObs spans.

Privacy-safe defaults:

```text
EVE_OTEL_RECORD_INPUTS=false
EVE_OTEL_RECORD_OUTPUTS=false
EVE_OTEL_DEBUG_PAYLOADS=false
DD_APM_TRACING_ENABLED=false
```

Only enable input/output capture after confirming that the destination and retention path are approved for the data. Debug payloads are opt-in and contain structural IDs, span names, safe attributes, and exporter result codes—not API headers, prompt/response bodies, tool arguments/results, or event attribute values.

Override `OTEL_EXPORTER_OTLP_TRACES_ENDPOINT` to use a local collector or another approved OTLP destination.

## Datadog Experiment reporting

`evals/evals.config.ts` attaches Eve's Datadog reporter when both `DD_API_KEY` and `DD_APP_KEY` are present:

```bash
DD_API_KEY="<datadog-api-key>"
DD_APP_KEY="<datadog-application-key>"
DD_SITE="datadoghq.com"
npm run eval
```

Force-disable reporting while retaining local eval output:

```bash
DD_EVE_EVAL_EXPERIMENTS_ENABLED=false npm run eval
```

The reporter records approved eval inputs, outputs, expected outputs, and assertion scores. The runtime spans remain independently indexed in LLM Observability.

## Evals

Run deterministic and judge-backed coverage:

```bash
npm run eval
```

Live OpenAI-backed stock evals are skipped by default. Enable them explicitly:

```bash
RUN_LIVE_STOCK_EVALS=true npm exec -- eve eval --strict evals/live
```

Coverage includes:

| Eval | Purpose |
| --- | --- |
| `evals/behavior/greeting-no-tools.eval.ts` | Greetings should not trigger stock tools. |
| `evals/behavior/missing-ticker-clarifies.eval.ts` | Missing ticker requests should be clarified. |
| `evals/surface/tools-discovered.eval.ts` | Eve should expose all authored tools. |
| `evals/live/quote-tool.eval.ts` | A quote request should call `get_stock_quote`. |
| `evals/live/watchlist-tool.eval.ts` | A watchlist request should call `analyze_watchlist`. |
| `evals/judge/onboarding-quality.eval.ts` | An LLM judge scores onboarding guidance. |
| `evals/judge/stock-tool-routing.eval.ts` | Focused stock research should use quote, profile, news, and sentiment tools. |

## Validation

```bash
npm run typecheck
npm run build
npm exec -- eve info
```

## Project structure

```text
agent/
├── agent.ts
├── channels/eve.ts
├── instructions.md
├── instrumentation/
│   ├── datadog.ts
│   └── otel.ts
├── lib/stock-watchlist/
└── tools/
evals/
├── behavior/
├── judge/
├── live/
├── surface/
└── evals.config.ts
```
