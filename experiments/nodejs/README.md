# Node.js LLM Observability Experiments examples

Runnable smoke examples for the Node.js `dd-trace-js` LLMObs experiments API. These pair with the `dd-trace-js` dataset and experiment-tracing SDK PRs and mirror the Python notebooks with live Datadog backend validation.

Every Node.js experiments SDK feature should have a runnable example in this directory when it is added.

## Setup

```sh
cd /Users/mehul.sonowal/dd/llm-observability/experiments/nodejs
cp .env.example .env
# Fill in DD_API_KEY, DD_APP_KEY, and OPENAI_API_KEY.
# DD_APPLICATION_KEY also works if DD_APP_KEY is not set.

npm run validate:p0
```

The scripts load `.env` from this directory automatically. To keep credentials in a different centralized file, set:

```sh
EXPERIMENTS_ENV_FILE=/path/to/experiments.env npm run validate:p0
```

Shell environment variables win over values in `.env`.

For staging validation, keep `DD_SITE=datad0g.com`; generated UI links use `https://dd.datad0g.com/...`.

## Running the examples

Run one example at a time while developing:

```sh
# 00: Dataset create -> push -> pull, explicit version pull, CSV dataset creation, and custom CSV record IDs.
npm run dataset
# Equivalent direct command:
node examples/00-dataset-operations.js
```

```sh
# 01: Basic experiment with live OpenAI calls, row evaluators, and a summary evaluator.
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
# 03: Multispan experiment task. Each experiment row trace contains nested workflow/task/LLM spans.
npm run multispan
# Equivalent direct command:
node examples/03-multispan-experiment.js
```

```sh
# 04: Stock watchlist workflow with multiple OpenAI calls per experiment row.
npm run stock-watchlist
# Equivalent direct command:
node examples/04-stock-watchlist-experiment.js
```

Run only the experiment trace validation sequence:

```sh
npm run validate:experiments
```

Run the full P0 validation sequence:

```sh
npm run validate:p0
```

Run against staging with `dd-auth` credentials:

```sh
dd-auth --domain dd.datad0g.com -- env DD_SITE=datad0g.com npm run validate:experiments
```

The dataset script exits non-zero if local result shape checks fail. It validates:

- `tracer.llmobs.createDataset(name, { description, records })`
- `dataset.push()`
- `tracer.llmobs.pullDataset(name, { expectedRecordCount })`
- version-pinned pulls with `pullDataset(name, { version })` when the backend returns a version
- `tracer.llmobs.createDatasetFromCsv(csvPath, name, options)`
- selected CSV input, expected output, metadata, and custom record ID columns

The experiment scripts exit non-zero if local result shape checks fail. They validate:

- `tracer.llmobs.experiments.experiment(...)`
- top-level `tracer.llmobs.experiment(...)` and `asyncExperiment(...)`
- experiment row spans and returned row `spanId` / `traceId`
- nested provider LLM spans inside experiment rows
- named function-array evaluators
- object-map evaluators
- summary evaluators and summary metrics
- task/evaluator retries through `run({ maxRetries, retryDelay })`
- captured row task errors
- `run({ raiseErrors: true })`
- nested workflow/task/LLM span traces for multispan tasks
- multiple provider calls in a single row with the stock watchlist workflow

The examples flush and wait briefly for LLMObs span delivery, then print URLs plus row span/trace IDs for UI validation of row spans, nested OpenAI LLM spans, evaluator metrics, and summary metrics. The multispan example should show each row trace as `experiment row → capital_answer_workflow → build_capital_prompt / openai.generate_capital_multispan / normalize_capital_answer`. The stock watchlist example should show each row trace as `experiment row → stock_watchlist_workflow → stock_researcher → quote/news/sentiment/ticker_synthesis + portfolio_synthesis`.
