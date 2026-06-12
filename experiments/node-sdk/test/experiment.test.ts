import assert from "node:assert/strict";
import { afterEach, beforeEach, test } from "node:test";
import { Experiment, ExperimentsClient } from "../src/index.js";
import { MockFetch, defaultRoutes } from "./mockFetch.js";

let mock: MockFetch;

function newClient(): ExperimentsClient {
  return new ExperimentsClient({
    apiKey: "key",
    applicationKey: "app",
    site: "datad0g.com", // staging — must route without special handling (W6/W7)
    projectName: "p",
    httpApi: mock.httpLibrary,
  });
}

beforeEach(() => {
  mock = new MockFetch(defaultRoutes());
  mock.install();
});
afterEach(() => mock.restore());

function buildExperiment(client: ExperimentsClient) {
  const ds = client.createDataset("d");
  ds.addRecord("apple banana", "fruit", { row: 0 });
  ds.addRecord("car truck", "vehicle", { row: 1 });
  ds.addRecord("x", "y", { row: 2 });
  ds.addRecord("z", "w", { row: 3 });

  return Experiment.builder<string, string>(client)
    .name("topic-relevance")
    .description("test exp")
    .dataset(ds)
    .task((input) => `echo:${input}`)
    .evaluator("nonempty", (_i, o) => typeof o === "string" && o.length > 0) // boolean
    .evaluator("length", (_i, o) => String(o).length) // number -> score
    .evaluator("label", (_i, o) => (String(o).includes("apple") ? "match" : "miss")) // categorical
    .tags({ env: "test" })
    .config({ temperature: 0 })
    .build();
}

test("run() returns a result with one row per record and a dashboard url", async () => {
  const client = newClient();
  const result = await buildExperiment(client).run();

  assert.equal(result.experimentId, "exp-1");
  assert.equal(result.rows.length, 4);
  assert.equal(result.url, "https://app.datad0g.com/llm/experiments/exp-1");
  assert.equal(result.rows[0].output, "echo:apple banana");
  assert.equal(result.rows[0].isError, false);
  assert.equal(result.rows[0].evaluations.nonempty, true);
  assert.equal(result.rows[0].evaluations.length, "echo:apple banana".length);
  assert.equal(result.rows[0].evaluations.label, "match");
});

test("experiment create body carries project_id, dataset_id, config and ensure_unique", async () => {
  await buildExperiment(newClient()).run();
  const createReq = mock
    .matching("POST /api/v2/llm-obs/v1/experiments")
    .filter((r) => !r.path.includes("/events"))[0];
  assert.ok(createReq);
  const attrs = createReq.body.data.attributes;
  assert.equal(createReq.body.data.type, "experiments");
  assert.equal(attrs.name, "topic-relevance");
  assert.equal(attrs.project_id, "proj-1");
  assert.equal(attrs.dataset_id, "ds-1");
  assert.equal(attrs.ensure_unique, true);
  assert.deepEqual(attrs.config, { temperature: 0 });
});

test("events POST uses type 'experiments' (W2) with spans and metrics", async () => {
  await buildExperiment(newClient()).run();
  const eventsReq = mock.matching("POST /api/v2/llm-obs/v1/experiments/exp-1/events")[0];
  assert.ok(eventsReq);
  assert.equal(eventsReq.body.data.type, "experiments");
  const { spans, metrics } = eventsReq.body.data.attributes;
  assert.equal(spans.length, 4);
  // 4 rows * 3 evaluators
  assert.equal(metrics.length, 12);
});

test("span wire format — ids, status, meta and metadata (W4)", async () => {
  await buildExperiment(newClient()).run();
  const span = mock.matching("POST /api/v2/llm-obs/v1/experiments/exp-1/events")[0].body.data
    .attributes.spans[0];

  assert.match(span.span_id, /^[0-9a-f]{32}$/);
  assert.match(span.trace_id, /^[0-9a-f]{32}$/);
  assert.equal(span.project_id, "proj-1");
  assert.equal(span.dataset_id, "ds-1");
  assert.equal(span.name, "topic-relevance");
  assert.equal(span.status, "ok");
  assert.equal(typeof span.start_ns, "number");
  assert.equal(typeof span.duration, "number");
  assert.equal(span.meta.input, "apple banana");
  assert.equal(span.meta.output, "echo:apple banana");
  assert.equal(span.meta.expected_output, "fruit");
  // metadata is raw / unwrapped and lives only under meta.metadata
  assert.deepEqual(span.meta.metadata, { row: 0 });
});

test("auto tags are present and win over user tags; metadata never appears in tags", async () => {
  const client = newClient();
  const ds = client.createDataset("d");
  ds.addRecord("a", "b", { secret: "should-not-be-a-tag" });
  const exp = Experiment.builder(client)
    .name("e")
    .dataset(ds)
    .task((i) => i)
    .evaluator("ok", () => true)
    .tags({ team: "core", experiment_id: "user-attempt-to-override" })
    .build();
  await exp.run();

  const span = mock.matching("POST /api/v2/llm-obs/v1/experiments/exp-1/events")[0].body.data
    .attributes.spans[0];
  const tags: string[] = span.tags;
  assert.ok(tags.includes("experiment_id:exp-1"), "auto experiment_id wins");
  assert.ok(!tags.includes("experiment_id:user-attempt-to-override"));
  assert.ok(tags.includes("dataset_id:ds-1"));
  assert.ok(tags.some((t) => t.startsWith("dataset_record_id:")));
  assert.ok(tags.includes("team:core"));
  assert.ok(!tags.some((t) => t.includes("secret")), "metadata is not emitted as a tag");
});

test("metric typing: boolean -> boolean_value, number -> score_value, other -> categorical_value", async () => {
  await buildExperiment(newClient()).run();
  const metrics: any[] = mock.matching("POST /api/v2/llm-obs/v1/experiments/exp-1/events")[0].body
    .data.attributes.metrics;

  const byLabel = (label: string) => metrics.find((m) => m.label === label && m.span_id);
  const nonempty = byLabel("nonempty");
  const length = byLabel("length");
  const label = byLabel("label");

  assert.equal(nonempty.metric_type, "boolean");
  assert.equal(nonempty.boolean_value, true);
  assert.equal(length.metric_type, "score");
  assert.equal(typeof length.score_value, "number");
  assert.equal(label.metric_type, "categorical");
  assert.equal(label.categorical_value, "match");

  // every metric carries the experiment_id auto tag
  for (const m of metrics) {
    assert.ok((m.tags as string[]).includes("experiment_id:exp-1"));
  }
});

test("status lifecycle: completed on success", async () => {
  await buildExperiment(newClient()).run();
  const patch = mock.matching("PATCH /api/v2/llm-obs/v1/experiments/exp-1")[0];
  assert.ok(patch);
  assert.equal(patch.body.data.type, "experiments");
  assert.equal(patch.body.data.attributes.status, "completed");
});

test("a task exception is captured per-row without aborting the experiment", async () => {
  const client = newClient();
  const ds = client.createDataset("d");
  ds.addRecord("good");
  ds.addRecord("bad");
  ds.addRecord("good2");

  const exp = Experiment.builder<string, string>(client)
    .name("e")
    .dataset(ds)
    .task((input) => {
      if (input === "bad") throw new Error("boom");
      return `ok:${input}`;
    })
    .evaluator("len", (_i, o) => String(o ?? "").length)
    .build();

  const result = await exp.run();
  assert.equal(result.rows.length, 3, "all rows present despite the failure");
  assert.equal(result.rows[1].isError, true);
  assert.equal(result.rows[1].errorMessage, "boom");

  const spans: any[] = mock.matching("POST /api/v2/llm-obs/v1/experiments/exp-1/events")[0].body
    .data.attributes.spans;
  assert.equal(spans[1].status, "error");
  assert.equal(spans[1].meta.error.message, "boom");
  // experiment still completes
  const patch = mock.matching("PATCH /api/v2/llm-obs/v1/experiments/exp-1")[0];
  assert.equal(patch.body.data.attributes.status, "completed");
});

test("an evaluator exception is captured per-evaluation without aborting", async () => {
  const client = newClient();
  const ds = client.createDataset("d");
  ds.addRecord("a");

  const exp = Experiment.builder(client)
    .name("e")
    .dataset(ds)
    .task((i) => i)
    .evaluator("explodes", () => {
      throw new Error("eval-fail");
    })
    .evaluator("fine", () => true)
    .build();

  const result = await exp.run();
  assert.equal(result.rows[0].evaluationErrors.explodes, "eval-fail");
  assert.equal(result.rows[0].evaluations.fine, true);

  const metrics: any[] = mock.matching("POST /api/v2/llm-obs/v1/experiments/exp-1/events")[0].body
    .data.attributes.metrics;
  const errored = metrics.find((m) => m.label === "explodes");
  assert.equal(errored.error.message, "eval-fail");
});

test("async tasks and evaluators are awaited", async () => {
  const client = newClient();
  const ds = client.createDataset("d");
  ds.addRecord("a");
  const exp = Experiment.builder<string, string>(client)
    .name("e")
    .dataset(ds)
    .task(async (i) => {
      await Promise.resolve();
      return `async:${i}`;
    })
    .evaluator("ok", async (_i, o) => String(o).startsWith("async"))
    .build();
  const result = await exp.run();
  assert.equal(result.rows[0].output, "async:a");
  assert.equal(result.rows[0].evaluations.ok, true);
});

test("builder requires name, dataset and task", () => {
  const client = newClient();
  const ds = client.createDataset("d");
  assert.throws(() => Experiment.builder(client).dataset(ds).task((i) => i).build(), /name/);
  assert.throws(() => Experiment.builder(client).name("n").task((i) => i).build(), /dataset/);
  assert.throws(() => Experiment.builder(client).name("n").dataset(ds).build(), /task/);
});
