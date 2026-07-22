# Node.js LLM Observability Experiments dataset example

Runnable smoke example for the Node.js `dd-trace-js` LLMObs experiments dataset API. This pairs with the `dd-trace-js` dataset SDK PR and mirrors the Python dataset notebook flow: create a local dataset, `push()` it, then `pullDataset(...)` from Datadog for read-after-write validation.

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

For staging validation, keep `DD_SITE=datad0g.com`; generated UI links use `https://dd.datad0g.com/...`. To validate local `dd-trace-js` changes without publishing/installing the package, set:

```sh
DD_TRACE_JS_PATH=/Users/mehul.sonowal/go/src/github.com/DataDog/dd-trace-js
```

If `DD_TRACE_JS_PATH` is unset, the script uses `require('dd-trace')` from normal Node resolution.

For detailed instructions on testing unpublished local `dd-trace-js` changes, see [LOCAL_DD_TRACE_JS.md](./LOCAL_DD_TRACE_JS.md).

## Running the example

```sh
# 00: Dataset create -> push -> pull, explicit version pull, CSV dataset creation, and custom CSV record IDs.
npm run dataset
# Equivalent direct command:
node examples/00-dataset-operations.js
```

Run against staging with `dd-auth` credentials and the local `dd-trace-js` checkout:

```sh
DD_TRACE_JS_PATH=/Users/mehul.sonowal/go/src/github.com/DataDog/dd-trace-js \
  dd-auth --domain dd.datad0g.com -- env DD_SITE=datad0g.com npm run validate:dataset
```

The dataset script exits non-zero if local result shape checks fail. It validates:

- `tracer.llmobs.createDataset(name, { description, records })`
- `dataset.push()`
- `tracer.llmobs.pullDataset(name, { expectedRecordCount })`
- version-pinned pulls with `pullDataset(name, { version })` when the backend returns a version
- `tracer.llmobs.createDatasetFromCsv(csvPath, name, options)`
- selected CSV input, expected output, metadata, and custom record ID columns
