import assert from "node:assert/strict";
import { afterEach, beforeEach, test } from "node:test";
import { buildTopicRelevance } from "../examples/topicRelevance.js";
import { ExperimentsClient } from "../src/index.js";
import { MockFetch, defaultRoutes } from "./mockFetch.js";

/**
 * Artifact-parity check: runs the *actual* TopicRelevance example code against a
 * capturing mock transport and asserts the wire artifacts it emits (dataset
 * records, experiment, spans, metrics) match what the Java `runTopicRelevance`
 * example produces.
 *
 * Only the inherently per-run fields differ between any two runs (of either SDK)
 * and are therefore excluded from the comparison: span_id / trace_id (random),
 * start_ns / duration (timing), and the server-assigned experiment/dataset/record
 * ids. Everything an end user sees as "the experiment" — records, inputs,
 * outputs, expected outputs, metadata, tags, eval labels/types/values — is
 * deterministic and asserted exactly.
 */

let mock: MockFetch;
beforeEach(() => {
  mock = new MockFetch(defaultRoutes());
  mock.install();
});
afterEach(() => mock.restore());

test("TopicRelevance emits the same artifacts as the Java example", async () => {
  const client = new ExperimentsClient({
    apiKey: "k",
    applicationKey: "a",
    site: "datadoghq.com",
    projectName: "node-sdk-bootstrap",
    httpApi: mock.httpLibrary,
  });

  // Fixed dataset name so the run is deterministic (the example normally appends
  // Date.now(), exactly like the Java example).
  const { dataset, experiment } = buildTopicRelevance(client, "topic-relevance-demo");
  const result = await experiment.run();

  // ---- 1. Dataset records artifact (what lands in the project's dataset) ----
  const recordsReq = mock.matching("POST /api/v2/llm-obs/v1/proj-1/datasets/ds-1/records")[0];
  assert.equal(recordsReq.body.data.type, "datasets"); // W1
  assert.deepEqual(recordsReq.body.data.attributes.records, [
    {
      input: { prompt: "I love hiking in the mountains on weekends.", topics: "outdoor, travel" },
      expected_output: "true",
      metadata: { source: "synthetic", difficulty: "easy" },
    },
    {
      input: { prompt: "Explain quantum entanglement in two sentences.", topics: "outdoor, travel" },
      expected_output: "false",
      metadata: { source: "synthetic", difficulty: "easy" },
    },
    {
      input: { prompt: "Best Italian restaurants in Brooklyn?", topics: "food, nyc" },
      expected_output: "true",
      metadata: { source: "user-report", difficulty: "medium", reviewer: "alex" },
    },
    {
      input: { prompt: "How do I configure nginx for HTTPS?", topics: "food, nyc" },
      expected_output: "false",
      metadata: { source: "user-report", difficulty: "hard", reviewer: "alex" },
    },
  ]);

  // ---- 2. Experiment artifact ----
  const expReq = mock
    .matching("POST /api/v2/llm-obs/v1/experiments")
    .filter((r) => !r.path.includes("/events"))[0];
  // The generated client serializes camelCase model props to snake_case on the
  // wire (ensureUnique → ensure_unique, projectId → project_id) — same as Java's
  // generated client. The captured body is the actual wire payload.
  assert.equal(expReq.body.data.type, "experiments");
  assert.equal(expReq.body.data.attributes.name, "topic-relevance-demo");
  assert.equal(expReq.body.data.attributes.ensure_unique, true);
  assert.equal(expReq.body.data.attributes.project_id, "proj-1");
  assert.equal(expReq.body.data.attributes.dataset_id, "ds-1");
  assert.deepEqual(expReq.body.data.attributes.config, { approach: "keyword-overlap", version: "v0.1" });

  // ---- 3. Span artifacts (one per row) ----
  const eventsReq = mock.matching("POST /api/v2/llm-obs/v1/experiments/exp-1/events")[0];
  assert.equal(eventsReq.body.data.type, "experiments"); // W2
  const spans: any[] = eventsReq.body.data.attributes.spans;
  assert.equal(spans.length, 4);

  // The deterministic, user-visible part of each span (volatile ids/timing dropped).
  const normalizedSpans = spans.map((s) => ({
    name: s.name,
    status: s.status,
    meta: s.meta,
    // tags minus the volatile dataset_record_id value
    tags: (s.tags as string[]).filter((t) => !t.startsWith("dataset_record_id:")).sort(),
  }));

  // All four prompts miss their topic keywords → keywordOverlap=false →
  // response="false", confidence=0.65 (identical logic to Java).
  const expectedMeta = (input: any, expected: string, metadata: any) => ({
    input,
    output: { response: "false", confidence: 0.65 },
    expected_output: expected,
    metadata,
  });
  const baseTags = ["dataset_id:ds-1", "experiment_id:exp-1", "owner:design-partner-bootstrap", "variant:java-v0.1"];

  assert.deepEqual(normalizedSpans[0], {
    name: "topic-relevance-demo",
    status: "ok",
    meta: expectedMeta(
      { prompt: "I love hiking in the mountains on weekends.", topics: "outdoor, travel" },
      "true",
      { source: "synthetic", difficulty: "easy" },
    ),
    tags: [...baseTags].sort(),
  });
  assert.equal(normalizedSpans[2].meta.expected_output, "true");
  assert.deepEqual(normalizedSpans[2].meta.metadata, { source: "user-report", difficulty: "medium", reviewer: "alex" });

  // ---- 4. Metric artifacts (3 evaluators × 4 rows) ----
  const metrics: any[] = eventsReq.body.data.attributes.metrics;
  assert.equal(metrics.length, 12);

  // Group by span so we can check each row's three metrics.
  const bySpan = new Map<string, Record<string, any>>();
  for (const m of metrics) {
    const row = bySpan.get(m.span_id) ?? {};
    row[m.label] = m;
    bySpan.set(m.span_id, row);
  }
  const rows = [...bySpan.values()];
  assert.equal(rows.length, 4);

  // exact_match: rows 0 & 2 expected "true" → false; rows 1 & 3 expected "false" → true.
  const exactMatches = rows.map((r) => r.exact_match.boolean_value);
  assert.deepEqual(
    [...exactMatches].sort(),
    [false, false, true, true],
    "two exact_match true, two false (deterministic from the heuristic vs labels)",
  );

  for (const r of rows) {
    assert.equal(r.exact_match.metric_type, "boolean");
    assert.equal(r.confidence_score.metric_type, "score");
    assert.equal(r.confidence_score.score_value, 0.65);
    assert.equal(r.verdict_category.metric_type, "categorical");
    assert.equal(r.verdict_category.categorical_value, "off-topic");
    // every metric carries the experiment_id auto tag
    assert.ok((r.exact_match.tags as string[]).includes("experiment_id:exp-1"));
  }

  // ---- 5. Result surface ----
  assert.equal(result.rows.length, 4);
  assert.equal(result.url, "https://app.datadoghq.com/llm/experiments/exp-1");
  assert.equal(dataset.url(), "https://app.datadoghq.com/llm/datasets/ds-1");
});
