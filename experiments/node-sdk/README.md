# Datadog LLM Experiments — Node.js / TypeScript SDK

> **Status: preview (v0.1).** The API may change. Not yet published to npm — install from source.

A thin workflow layer over the Datadog LLM Observability API for running LLM
experiments from Node.js. This SDK is **interface-compatible** with
[`datadog-llm-experiments-java`](../java-sdk): the same six concepts, the same
operations, and the same wire format. A user who knows one SDK can use the other
without learning anything new.

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

### Error handling

- A thrown error (or rejected promise) in a **task** is captured on that row's
  span (`status: "error"`, `meta.error`); the experiment continues.
- A thrown error in an **evaluator** is captured as an `error` on that metric;
  other evaluators and rows continue.

## API parity with the Java SDK

The two SDKs map one-to-one. Node uses object-argument constructors and
`async/await` where Java uses builders and blocking calls; everything else —
type names, methods, wire format, auto-tags, metadata propagation — is identical.

| Java | Node |
|---|---|
| `ExperimentsClient.builder()....build()` | `new ExperimentsClient({...})` *or* `ExperimentsClient.builder()....build()` |
| `client.createDataset(name, desc)` | `client.createDataset(name, desc)` |
| `client.pullDataset(name)` | `await client.pullDataset(name)` |
| `dataset.addRecord(input, expected, meta)` | `dataset.addRecord(input, expected, meta)` |
| `Experiment.builder(client)....build()` | `Experiment.builder(client)....build()` |
| `experiment.run()` | `await experiment.run()` |
| `result.url()` | `result.url` |
| `client.api()` (generated `LlmObservabilityApi`) | `client.api()` (generated `LLMObservabilityApi` from `@datadog/datadog-api-client`) |

## Limitations (v0.1)

- **Sequential execution.** Rows run one at a time; parallelism is a v0.2 add.
- **No tracer dependency.** Spans are emitted via the events endpoint, not
  `dd-trace`. This is intentional per the polyglot spec.
- **Transport.** Like the Java SDK, the Node SDK calls the generated
  `@datadog/datadog-api-client` (`LLMObservabilityApi`) for project, dataset and
  experiment creation and for the dataset list/record-list reads. The three
  endpoints with active spec-drift workarounds are hand-rolled over `fetch`,
  exactly as Java hand-rolls them via `DirectPost`:
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

These are direct ports of the Java SDK examples — same dataset records,
evaluators, config and tags — so the Node and Java runs produce equivalent
artifacts. They write to Node-specific projects (`node-sdk-bootstrap`,
`node-sdk-minimal`; the Java examples use `java-sdk-bootstrap` /
`java-sdk-minimal`), keeping each SDK's artifacts in its own project for
side-by-side comparison.

## Develop & test

```bash
npm test      # unit tests (node:test) — mock fetch, no network
npm run build # type-check + emit dist/
```

The unit tests assert the exact wire format (span/metric JSON, `type`
discriminators, auto-tags, metadata propagation, metric typing, error capture)
against a mocked `fetch`, so they run offline and pin parity with the spec.
