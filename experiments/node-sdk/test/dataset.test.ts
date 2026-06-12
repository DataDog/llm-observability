import assert from "node:assert/strict";
import { afterEach, beforeEach, test } from "node:test";
import { Dataset, DatasetRecord, ExperimentsClient } from "../src/index.js";
import { MockFetch, defaultRoutes } from "./mockFetch.js";

let mock: MockFetch;

function newClient(): ExperimentsClient {
  return new ExperimentsClient({
    apiKey: "key",
    applicationKey: "app",
    site: "datadoghq.com",
    projectName: "my-project",
    httpApi: mock.httpLibrary,
  });
}

beforeEach(() => {
  mock = new MockFetch(defaultRoutes());
  mock.install();
});
afterEach(() => mock.restore());

test("createDataset returns an unpushed, empty buffer", () => {
  const ds = newClient().createDataset("d", "desc");
  assert.equal(ds.name(), "d");
  assert.equal(ds.id(), null);
  assert.equal(ds.projectId(), null);
  assert.deepEqual([...ds.records()], []);
});

test("addRecord is fluent and accepts both shapes", () => {
  const ds = newClient().createDataset("d");
  const ret = ds
    .addRecord("in1", "out1", { team: "a" })
    .addRecord(new DatasetRecord("in2", "out2"));
  assert.equal(ret, ds, "addRecord returns the dataset for chaining");
  const recs = ds.records();
  assert.equal(recs.length, 2);
  assert.equal(recs[0].input, "in1");
  assert.deepEqual(recs[0].metadata, { team: "a" });
  assert.equal(recs[1].expectedOutput, "out2");
});

test("push creates the dataset and sends records with type 'datasets' (W1)", async () => {
  const ds = newClient().createDataset("d", "the desc");
  ds.addRecord("hello", "world", { file: "f.txt" });
  ds.addRecord("foo"); // no expected_output, no metadata
  await ds.push();

  assert.equal(ds.id(), "ds-1");
  assert.equal(ds.projectId(), "proj-1");

  // Dataset create body
  const createReq = mock.matching("POST /api/v2/llm-obs/v1/proj-1/datasets")
    .filter((r) => !r.path.endsWith("/records"))[0];
  assert.ok(createReq, "dataset create was called");
  assert.equal(createReq.body.data.type, "datasets");
  assert.equal(createReq.body.data.attributes.name, "d");
  assert.equal(createReq.body.data.attributes.description, "the desc");

  // Records push body
  const recReq = mock.matching("POST /api/v2/llm-obs/v1/proj-1/datasets/ds-1/records")[0];
  assert.ok(recReq, "records push was called");
  assert.equal(recReq.body.data.type, "datasets");
  const records = recReq.body.data.attributes.records;
  assert.equal(records.length, 2);
  assert.deepEqual(records[0], {
    input: "hello",
    expected_output: "world",
    metadata: { file: "f.txt" },
  });
  // Second record omits expected_output and metadata entirely.
  assert.deepEqual(records[1], { input: "foo" });

  // Record ids captured from the response, aligned by index.
  assert.deepEqual([...ds.recordIds()], ["rec-0", "rec-1"]);
});

test("dataset url() is null until pushed, then links to the dataset", async () => {
  const ds = newClient().createDataset("d");
  assert.equal(ds.url(), null);
  ds.addRecord("a");
  await ds.push();
  assert.equal(ds.url(), "https://app.datadoghq.com/llm/datasets/ds-1");
});

test("push is incremental — only new records are sent on the second push", async () => {
  const ds = newClient().createDataset("d");
  ds.addRecord("a");
  await ds.push();
  ds.addRecord("b");
  await ds.push();

  const recordPosts = mock.matching("POST /api/v2/llm-obs/v1/proj-1/datasets/ds-1/records");
  assert.equal(recordPosts.length, 2);
  assert.deepEqual(
    recordPosts[1].body.data.attributes.records.map((r: any) => r.input),
    ["b"],
    "second push only sends the newly added record",
  );
});

test("DD auth headers are attached to every request", async () => {
  const ds = newClient().createDataset("d");
  ds.addRecord("a");
  await ds.push();
  for (const req of mock.requests) {
    assert.equal(req.headers["DD-API-KEY"], "key");
    assert.equal(req.headers["DD-APPLICATION-KEY"], "app");
  }
});

test("pullDataset finds a dataset by name and loads its records", async () => {
  mock.restore();
  mock = new MockFetch([
    { match: "POST /api/v2/llm-obs/v1/projects", response: { data: { id: "proj-1" } } },
    {
      match: "GET /api/v2/llm-obs/v1/proj-1/datasets/ds-9/records",
      response: {
        // The API returns record content nested under data[].attributes (JSON:API).
        data: [
          { id: "r1", type: "datasets", attributes: { input: "i1", expected_output: "e1", metadata: { a: 1 } } },
          { id: "r2", type: "datasets", attributes: { input: "i2" } },
        ],
      },
    },
    {
      match: "GET /api/v2/llm-obs/v1/proj-1/datasets",
      response: {
        data: [
          { id: "ds-other", attributes: { name: "nope" } },
          { id: "ds-9", attributes: { name: "wanted", description: "d" } },
        ],
      },
    },
  ]);
  mock.install();

  const ds = await newClient().pullDataset("wanted");
  assert.equal(ds.id(), "ds-9");
  assert.equal(ds.projectId(), "proj-1");
  assert.deepEqual([...ds.recordIds()], ["r1", "r2"]);
  const recs = ds.records();
  assert.equal(recs.length, 2);
  assert.equal(recs[0].input, "i1");
  assert.deepEqual(recs[0].metadata, { a: 1 });
});

test("pullDataset throws a clear error when the dataset is absent", async () => {
  mock.restore();
  mock = new MockFetch([
    { match: "POST /api/v2/llm-obs/v1/projects", response: { data: { id: "proj-1" } } },
    { match: "GET /api/v2/llm-obs/v1/proj-1/datasets", response: { data: [] } },
  ]);
  mock.install();
  // maxWaitMs: 0 → a single attempt, no 30s wait, in the test.
  await assert.rejects(() => newClient().pullDataset("ghost", { maxWaitMs: 0 }), /not found/);
});

test("pullDataset waits (backoff) until all expected records are readable", async () => {
  mock.restore();
  let recordsCalls = 0;
  mock = new MockFetch([
    { match: "POST /api/v2/llm-obs/v1/projects", response: { data: { id: "proj-1" } } },
    {
      match: "GET /api/v2/llm-obs/v1/proj-1/datasets/ds-9/records",
      response: () => {
        // First read sees only 1 of 2 records (read-after-write lag); then both.
        recordsCalls += 1;
        const rec = (id: string, input: string) => ({ id, type: "datasets", attributes: { input } });
        return recordsCalls < 2
          ? { data: [rec("r1", "i1")] }
          : { data: [rec("r1", "i1"), rec("r2", "i2")] };
      },
    },
    {
      match: "GET /api/v2/llm-obs/v1/proj-1/datasets",
      response: { data: [{ id: "ds-9", attributes: { name: "wanted" } }] },
    },
  ]);
  mock.install();

  const ds = await newClient().pullDataset("wanted", {
    expectedRecordCount: 2,
    maxWaitMs: 5000, // generous ceiling; the first retry (~250ms) already satisfies it
  });
  assert.equal(ds.records().length, 2);
  assert.ok(recordsCalls >= 2, "records endpoint was polled more than once");
});

test("pullDataset throws if expected records never arrive within the budget", async () => {
  mock.restore();
  mock = new MockFetch([
    { match: "POST /api/v2/llm-obs/v1/projects", response: { data: { id: "proj-1" } } },
    {
      match: "GET /api/v2/llm-obs/v1/proj-1/datasets/ds-9/records",
      response: { data: [{ id: "r1", type: "datasets", attributes: { input: "i1" } }] }, // only ever 1 record
    },
    {
      match: "GET /api/v2/llm-obs/v1/proj-1/datasets",
      response: { data: [{ id: "ds-9", attributes: { name: "wanted" } }] },
    },
  ]);
  mock.install();
  await assert.rejects(
    () => newClient().pullDataset("wanted", { expectedRecordCount: 3, maxWaitMs: 0 }),
    /expected 3.*backend may not have finished ingesting/,
  );
});
