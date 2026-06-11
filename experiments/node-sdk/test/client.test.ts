import assert from "node:assert/strict";
import { afterEach, beforeEach, test } from "node:test";
import { ExperimentsClient } from "../src/index.js";
import { MockFetch, defaultRoutes } from "./mockFetch.js";

let mock: MockFetch;
beforeEach(() => {
  mock = new MockFetch(defaultRoutes());
  mock.install();
});
afterEach(() => mock.restore());

test("object-arg constructor and static builder() are equivalent", () => {
  const a = new ExperimentsClient({
    apiKey: "k",
    applicationKey: "a",
    site: "us5.datadoghq.com",
    projectName: "p",
  });
  const b = ExperimentsClient.builder()
    .apiKey("k")
    .applicationKey("a")
    .site("us5.datadoghq.com")
    .projectName("p")
    .build();
  assert.equal(a.projectName(), b.projectName());
  assert.equal(a.site(), b.site());
  assert.equal(a.site(), "us5.datadoghq.com");
});

test("site defaults to datadoghq.com", () => {
  const c = new ExperimentsClient({ apiKey: "k", applicationKey: "a", projectName: "p" });
  assert.equal(c.site(), "datadoghq.com");
});

test("constructor validates required fields", () => {
  assert.throws(
    () => new ExperimentsClient({ apiKey: "", applicationKey: "a", projectName: "p" }),
    /apiKey/,
  );
  assert.throws(
    () => new ExperimentsClient({ apiKey: "k", applicationKey: "", projectName: "p" }),
    /applicationKey/,
  );
  assert.throws(
    () => new ExperimentsClient({ apiKey: "k", applicationKey: "a", projectName: "" }),
    /projectName/,
  );
});

test("ensureProjectId creates-or-gets the project (via the generated client) and caches the id", async () => {
  const c = new ExperimentsClient({
    apiKey: "k",
    applicationKey: "a",
    projectName: "my-proj",
    httpApi: mock.httpLibrary,
  });
  const id1 = await c.ensureProjectId();
  const id2 = await c.ensureProjectId();
  assert.equal(id1, "proj-1");
  assert.equal(id2, "proj-1");
  // cached — only one POST despite two calls
  const posts = mock.matching("POST /api/v2/llm-obs/v1/projects");
  assert.equal(posts.length, 1);
  assert.equal(posts[0].via, "generated", "project create goes through the generated client");
  assert.equal(posts[0].body.data.type, "projects");
  assert.equal(posts[0].body.data.attributes.name, "my-proj");
  // generated client attaches the DD auth headers
  assert.equal(posts[0].headers["DD-API-KEY"], "k");
  assert.equal(posts[0].headers["DD-APPLICATION-KEY"], "a");
});

test("api() escape hatch returns the generated LLMObservabilityApi", async () => {
  const c = new ExperimentsClient({
    apiKey: "k",
    applicationKey: "a",
    projectName: "p",
    httpApi: mock.httpLibrary,
  });
  const api = c.api();
  // It's the generated client, so generated operations are available directly.
  assert.equal(typeof api.createLLMObsProject, "function");
  assert.equal(typeof api.listLLMObsExperiments, "function");
  const resp = await api.createLLMObsProject({
    body: { data: { type: "projects", attributes: { name: "p" } } },
  });
  assert.equal(resp.data?.id, "proj-1");
});
