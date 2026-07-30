# Node.js LLM Observability Experiments examples

Runnable examples for the Node.js `dd-trace-js` LLMObs experiments API. These pair with the `dd-trace-js` dataset and experiment-tracing SDK PRs and mirror the Python notebooks with live Datadog backend validation.

Every Node.js experiments SDK feature should have a runnable example in this directory when it is added.

## Setup

```sh
cd /Users/mehul.sonowal/dd/llm-observability/experiments/nodejs
cp .env.example .env
# Fill in DD_API_KEY, DD_APP_KEY, and OPENAI_API_KEY.
# DD_APPLICATION_KEY also works if DD_APP_KEY is not set.
npm install

npm run validate:all
```

The scripts load `.env` from this directory automatically. To keep credentials in a different centralized file, set:

```sh
EXPERIMENTS_ENV_FILE=/path/to/experiments.env npm run validate:all
```

Shell environment variables win over values in `.env`.

For production validation, keep `DD_SITE=datadoghq.com`; generated UI links use `https://app.datadoghq.com/...`.

## Running the examples

Run one example at a time while developing:

```sh
# 00: Dataset create -> push -> pull, explicit version pull, and incremental record pushes.
npm run dataset
# Equivalent direct command:
node examples/00-dataset-operations.js
```

```sh
# 01: Basic experiment with live OpenAI calls, nested spans, row evaluators, and a summary evaluator.
npm run basic
# Equivalent direct command:
node examples/01-basic-experiment.js
```

```sh
# 02: Error handling, retries, evaluator failures, summary metrics, and live OpenAI calls on successful rows.
npm run errors
# Equivalent direct command:
node examples/02-error-retry-summary.js
```

```sh
# 03: Stock watchlist workflow with multiple OpenAI calls per experiment row.
npm run stock-watchlist
# Equivalent direct command:
node examples/03-stock-watchlist-experiment.js
```

Run only the experiment trace validation sequence:

```sh
npm run validate:experiments
```

Run the full validation sequence:

```sh
npm run validate:all
```

Run against production with `dd-auth` credentials:

```sh
dd-auth --domain dd.datadoghq.com -- env DD_SITE=datadoghq.com npm run validate:experiments
```

The dataset script exits non-zero if local result shape checks fail. It validates:

- `tracer.llmobs.experiments.createDataset(name, { description, records })`
- `dataset.push()`
- incremental `dataset.addRecord(...)` plus follow-up/no-op pushes
- `tracer.llmobs.experiments.pullDataset(name, { expectedRecordCount })`
- version-pinned pulls with `pullDataset(name, { version })` when the backend returns a version
- custom record IDs returned by dataset push/pull and used for experiment row tags

The experiment scripts exit non-zero if local result shape checks fail. They validate:

- `tracer.llmobs.experiments.experiment(...)`
- top-level `tracer.llmobs.experiment(...)`
- experiment row spans and returned row `spanId` / `traceId`
- custom and backend-generated dataset record IDs on experiment rows
- nested provider LLM spans inside experiment rows
- named function-array evaluators
- object-map evaluators
- summary evaluators and summary metrics
- task/evaluator retries through `run({ maxRetries, retryDelay })`
- captured row task errors
- `run({ throwOnErrors: true })` for task errors that should be captured and bubbled to callers
- nested workflow/task/LLM span traces in the basic experiment
- multiple provider calls in a single row with the stock watchlist workflow

The examples use the official OpenAI Node.js SDK, flush and wait briefly for LLMObs span delivery, then print URLs plus row span/trace IDs for UI validation of row spans, nested OpenAI LLM spans, evaluator metrics, and summary metrics. The basic example should show each row trace as `experiment row → capital_answer_workflow → build_capital_prompt / openai.generate_capital / normalize_capital_answer`. The stock watchlist example should show each row trace as `experiment row → stock_watchlist_workflow → stock_researcher → quote/news/sentiment/ticker_synthesis + portfolio_synthesis`.
