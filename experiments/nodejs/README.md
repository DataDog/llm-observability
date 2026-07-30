# Node.js LLM Observability Experiments dataset example

Runnable example for the Node.js `dd-trace-js` LLMObs experiments dataset API. This pairs with the `dd-trace-js` dataset SDK PR and mirrors the Python dataset notebook flow: create a local dataset, `push()` it, then `pullDataset(...)` from Datadog for read-after-write validation.

Every Node.js experiments SDK feature should have a runnable example in this directory when it is added.

## Setup

```sh
cd /Users/mehul.sonowal/dd/llm-observability/experiments/nodejs
cp .env.example .env
# Fill in DD_API_KEY and DD_APP_KEY.
# DD_APPLICATION_KEY also works if DD_APP_KEY is not set.

npm run validate:dataset
```

The script loads `.env` from this directory automatically. To keep credentials in a different centralized file, set:

```sh
EXPERIMENTS_ENV_FILE=/path/to/experiments.env npm run validate:dataset
```

Shell environment variables win over values in `.env`.

For production validation, keep `DD_SITE=datadoghq.com`; generated UI links use `https://app.datadoghq.com/...`.

## Running the example

```sh
# 00: Dataset create -> push -> pull, explicit version pull, and incremental record pushes.
npm run dataset
# Equivalent direct command:
node examples/00-dataset-operations.js
```

Run against production with `dd-auth` credentials:

```sh
dd-auth --domain dd.datadoghq.com -- env DD_SITE=datadoghq.com npm run validate:dataset
```

The dataset script exits non-zero if local result shape checks fail. It validates:

- `tracer.llmobs.experiments.createDataset(name, { description, records })`
- `dataset.push()`
- incremental `dataset.addRecord(...)` plus follow-up/no-op pushes
- `tracer.llmobs.experiments.pullDataset(name, { expectedRecordCount })`
- version-pinned pulls with `pullDataset(name, { version })` when the backend returns a version
- custom record IDs returned by dataset push/pull and used for experiment row tags
