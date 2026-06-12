# Datadog LLM Experiments — Node.js / TypeScript SDK

> **Status: preview (v0.1).** The API may change. Not yet published to npm — install from source.

A thin workflow layer over the Datadog LLM Observability API for running LLM
experiments from Node.js — six small types covering datasets, tasks, evaluators,
and experiment runs.

## Requirements

- Node.js 20+ (uses the built-in `fetch` and `node:crypto`)
- A Datadog API key and Application key
- [`@datadog/datadog-api-client`](https://github.com/DataDog/datadog-api-client-typescript) `^1.58.0` (the only runtime dependency — installed automatically)

## Install (from source, v0.1)

```bash
cd experiments/node-sdk
npm install
npm run build      # emits dist/
```

Then depend on it via a local path (`"@datadog/llm-experiments": "file:../node-sdk"`)
or `npm link`.

## Quick start

```ts
import { Experiment, ExperimentsClient } from "@datadog/llm-experiments";

const client = new ExperimentsClient({
  apiKey: process.env.DD_API_KEY!,
  applicationKey: process.env.DD_APP_KEY!,
  site: "datadoghq.com",          // optional; defaults to datadoghq.com
  projectName: "my-project",
});

const dataset = client
  .createDataset("greetings", "simple casing dataset")
  .addRecord("hello", "HELLO", { source: "demo" })
  .addRecord("world", "WORLD");

const result = await Experiment.builder<string, string>(client)
  .name("uppercase")
  .dataset(dataset)
  .task((input) => input.toUpperCase())
  .evaluator("matches_expected", (_input, output, expected) => output === expected) // boolean
  .evaluator("length", (_input, output) => String(output).length)                   // score
  .tags({ team: "ml-obs" })
  .build()
  .run();

console.log(dataset.url());        // https://app.datadoghq.com/llm/datasets/<id>
console.log(result.url);           // https://app.datadoghq.com/llm/experiments/<id>
console.log(result.rows.length);
```

The dataset is created and pushed automatically on the first `run()`. Call
`await dataset.push()` if you want to push eagerly.

## Concepts

| Type | Purpose |
|---|---|
| `ExperimentsClient` | Auth + site + project; factory for datasets and the API escape hatch. |
| `Dataset` | Record buffer; auto-created and pushed on first experiment run. `dataset.url()` links to its UI page once pushed. |
| `DatasetRecord` | Immutable `{ input, expectedOutput?, metadata? }`. |
| `Task<I, O>` | Your callable: `(input, config) => output`. Sync or async. |
| `Evaluator<V>` | Your callable: `(input, output, expectedOutput) => value`. Sync or async. |
| `Experiment<I, O>` | Builder + `run()` orchestration. |
| `ExperimentResult` | What `run()` returns — `rows`, `experimentId`, `url`. |

### Evaluator return types drive the metric type

| Return value | Metric |
|---|---|
| `boolean` | boolean metric |
| `number` | score metric |
| anything else | categorical metric (stringified) |

### Eventually-consistent reads (`pullDataset`)

LLM Obs reads lag writes slightly, so a `pullDataset` issued right after a push
can briefly see the dataset missing or only some records. `pullDataset` polls
with exponential backoff (250ms → 500ms → 1s → …, capped at a 30s total by
default) until the dataset is visible. Pass `expectedRecordCount` to also wait
until that many records are readable — confirming the push fully landed — and
`maxWaitMs` to tune the ceiling:

```ts
const pulled = await client.pullDataset(name, {
  expectedRecordCount: dataset.records().length,
  maxWaitMs: 30_000, // default
});
```

If the dataset never appears (or the expected records never arrive) within the
budget, it throws a clear error.

### Error handling

- A thrown error (or rejected promise) in a **task** is captured on that row's
  span (`status: "error"`, `meta.error`); the experiment continues.
- A thrown error in an **evaluator** is captured as an `error` on that metric;
  other evaluators and rows continue.

## Limitations (v0.1)

- **Sequential execution.** Rows run one at a time; parallelism is a v0.2 add.
- **No tracer dependency.** Spans are emitted via the events endpoint, not
  `dd-trace`. This is intentional.
- **Transport.** The SDK calls the generated `@datadog/datadog-api-client`
  (`LLMObservabilityApi`) for project, dataset and experiment creation and for
  the dataset/record list reads. Three endpoints the generated client cannot
  perform correctly are hand-rolled over `fetch`:
  - **W1** — records POST must send `type: "datasets"` (generated model still emits `"records"`).
  - **W2** — events POST must send `type: "experiments"` (generated model still emits `"events"`).
  - experiment **status PATCH** — the generated update model exposes only `name`/`description`, not `status`.

  The SDK also applies **W5** (relaxes the generated deserializer's `required`-field
  enforcement) and **W6/W7** (a fresh `Configuration` with an explicit
  `baseServer`, so `DD_SITE` is never read eagerly and every site is allowed).
- **Staging.** `site: "datad0g.com"` works out of the box (W6/W7).

## Examples

| File | What it shows |
|---|---|
| [`examples/topicRelevance.ts`](examples/topicRelevance.ts) | The canonical example: mock task + boolean/numeric/categorical evaluators over 4 records. |
| [`examples/minimalExperiment.ts`](examples/minimalExperiment.ts) | Smallest end-to-end experiment. |
| [`examples/datasetOperations.ts`](examples/datasetOperations.ts) | Create + push + pull a dataset by name. |

```bash
DD_API_KEY=... DD_APPLICATION_KEY=... npm run runTopicRelevance
```

The examples write to the `node-sdk-bootstrap` and `node-sdk-minimal` projects.

## Develop & test

```bash
npm test      # unit tests (node:test) — mock fetch, no network
npm run build # type-check + emit dist/
```

The unit tests assert the exact wire format (span/metric JSON, `type`
discriminators, auto-tags, metadata propagation, metric typing, error capture)
against a mocked `fetch`, so they run offline and pin the wire format.
