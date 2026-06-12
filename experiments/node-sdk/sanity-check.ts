/**
 * Offline sanity check for the LLM Experiments Node SDK.
 *
 * Exercises the entire library end-to-end — client → dataset → experiment →
 * run() → result — against an in-process mock transport. No API keys and no
 * network required: it verifies the library *wiring* (generated-client calls,
 * hand-rolled W1/W2/status calls, span/metric wire format, evaluator typing,
 * error capture). Exits non-zero if any assertion fails.
 *
 *   npm run sanity
 *   # or: node --import tsx sanity-check.ts
 *
 * For a *live* check against a real Datadog org, use the examples instead
 * (npm run example:minimal), which need DD_API_KEY / DD_APP_KEY.
 */
import { client } from "@datadog/datadog-api-client";
import { Experiment, ExperimentsClient } from "./src/index.js";

// ---------------------------------------------------------------------------
// Tiny mock transport: serves both the generated client (via HttpLibrary) and
// the hand-rolled fetch calls (via globalThis.fetch). Records every request.
// ---------------------------------------------------------------------------
const requests: Array<{ method: string; path: string; via: string; body: any }> = [];

function respond(method: string, path: string, body: any): { status: number; payload: unknown } {
  const recordsPath = "/api/v2/llm-obs/v1/proj/datasets/ds/records";
  if (method === "POST" && path === "/api/v2/llm-obs/v1/projects") {
    return { status: 200, payload: { data: { id: "proj", type: "projects", attributes: { name: "p" } } } };
  }
  if (method === "POST" && path === recordsPath) {
    const sent: unknown[] = body?.data?.attributes?.records ?? [];
    return { status: 200, payload: { data: sent.map((_, i) => ({ id: `rec-${i}` })) } };
  }
  if (method === "POST" && path === "/api/v2/llm-obs/v1/proj/datasets") {
    return { status: 200, payload: { data: { id: "ds", type: "datasets", attributes: { name: "d" } } } };
  }
  if (method === "POST" && path === "/api/v2/llm-obs/v1/experiments") {
    return { status: 200, payload: { data: { id: "exp", type: "experiments", attributes: { name: "e" } } } };
  }
  if (method === "POST" && path.endsWith("/events")) return { status: 202, payload: {} };
  if (method === "PATCH" && path.startsWith("/api/v2/llm-obs/v1/experiments/")) return { status: 200, payload: { data: { id: "exp" } } };
  return { status: 404, payload: { error: `no route for ${method} ${path}` } };
}

// transport 1: hand-rolled fetch (records POST, events POST, status PATCH)
globalThis.fetch = (async (input: string, init?: RequestInit) => {
  const path = new URL(String(input)).pathname;
  const method = (init?.method ?? "GET").toUpperCase();
  const body = init?.body ? JSON.parse(init.body as string) : undefined;
  requests.push({ method, path, via: "fetch", body });
  const { status, payload } = respond(method, path, body);
  return new Response(JSON.stringify(payload), { status });
}) as unknown as typeof globalThis.fetch;

// transport 2: generated client (project/dataset/experiment create + lists)
const httpApi: client.HttpLibrary = {
  send: async (req: client.RequestContext): Promise<client.ResponseContext> => {
    const path = new URL(req.getUrl()).pathname;
    const method = String(req.getHttpMethod()).toUpperCase();
    const raw = req.getBody();
    const body = typeof raw === "string" && raw ? JSON.parse(raw) : undefined;
    requests.push({ method, path, via: "generated", body });
    const { status, payload } = respond(method, path, body);
    const json = JSON.stringify(payload);
    return new client.ResponseContext(status, {}, { text: async () => json, binary: async () => Buffer.from(json) });
  },
};

// ---------------------------------------------------------------------------
// The actual sanity run.
// ---------------------------------------------------------------------------
let failures = 0;
function check(label: string, cond: boolean, detail = ""): void {
  const mark = cond ? "✓" : "✗";
  if (!cond) failures++;
  console.log(`  ${mark} ${label}${detail ? ` — ${detail}` : ""}`);
}

async function main(): Promise<void> {
  console.log("LLM Experiments Node SDK — offline sanity check\n");

  const cdClient = new ExperimentsClient({
    apiKey: "fake-key",
    applicationKey: "fake-app-key",
    site: "datad0g.com", // staging — exercises the W6/W7 server config
    projectName: "sanity-project",
    httpApi, // inject mock transport
  });

  console.log("1. Build dataset");
  const dataset = cdClient
    .createDataset("sanity-dataset", "a tiny dataset")
    .addRecord("apple banana", "fruit", { row: 0 })
    .addRecord("car truck", "vehicle", { row: 1 })
    .addRecord("BOOM", "n/a", { row: 2 }); // this row's task will throw
  check("dataset starts unpushed", dataset.id() === null);
  check("3 records buffered", dataset.records().length === 3);

  console.log("\n2. Build + run experiment");
  const experiment = Experiment.builder<string, string>(cdClient)
    .name("sanity-experiment")
    .description("offline sanity")
    .dataset(dataset)
    .task((input) => {
      if (input === "BOOM") throw new Error("intentional task failure");
      return `echo:${input}`;
    })
    .evaluator("nonempty", (_i, o) => typeof o === "string" && o.length > 0) // boolean
    .evaluator("length", (_i, o) => String(o ?? "").length) // number → score
    .evaluator("verdict", (_i, o) => (String(o).includes("apple") ? "match" : "miss")) // categorical
    .tags({ env: "sanity" })
    .config({ temperature: 0 })
    .build();

  const result = await experiment.run();

  console.log("\n3. Result");
  console.log(`   experimentId: ${result.experimentId}`);
  console.log(`   url:          ${result.url}`);
  console.log(`   rows:         ${result.rows.length}`);
  check("experiment id resolved", result.experimentId === "exp");
  check("staging experiment url", result.url === "https://app.datad0g.com/llm/experiments/exp");
  check("dataset links to its dashboard page", dataset.url() === "https://app.datad0g.com/llm/datasets/ds");
  check("one row per record", result.rows.length === 3);
  check("row 0 output", result.rows[0].output === "echo:apple banana");
  check("row 0 boolean eval", result.rows[0].evaluations.nonempty === true);
  check("row 0 score eval", result.rows[0].evaluations.length === "echo:apple banana".length);
  check("row 0 categorical eval", result.rows[0].evaluations.verdict === "match");
  check("row 2 captured task error", result.rows[2].isError && result.rows[2].errorMessage === "intentional task failure");

  console.log("\n4. Transport split (generated client vs hand-rolled)");
  const generated = requests.filter((r) => r.via === "generated").map((r) => `${r.method} ${r.path}`);
  const handRolled = requests.filter((r) => r.via === "fetch").map((r) => `${r.method} ${r.path}`);
  console.log("   via generated client:");
  for (const r of generated) console.log(`     - ${r}`);
  console.log("   hand-rolled over fetch:");
  for (const r of handRolled) console.log(`     - ${r}`);
  check("project/dataset/experiment created via generated client", generated.length === 3);
  check("records + events + status hand-rolled", handRolled.length === 3);

  console.log("\n5. Wire-format spot checks");
  const recordsReq = requests.find((r) => r.path.endsWith("/records"));
  const eventsReq = requests.find((r) => r.path.endsWith("/events"));
  check("W1: records POST sends type 'datasets'", recordsReq?.body?.data?.type === "datasets");
  check("W2: events POST sends type 'experiments'", eventsReq?.body?.data?.type === "experiments");
  const span0 = eventsReq?.body?.data?.attributes?.spans?.[0];
  check("span has 32-hex span_id", /^[0-9a-f]{32}$/.test(span0?.span_id ?? ""));
  check("span carries auto experiment_id tag", (span0?.tags ?? []).includes("experiment_id:exp"));
  check("span metadata is raw/unwrapped", JSON.stringify(span0?.meta?.metadata) === JSON.stringify({ row: 0 }));
  const metrics = eventsReq?.body?.data?.attributes?.metrics ?? [];
  check("metrics emitted (3 evaluators × 3 rows)", metrics.length === 9);
  check("boolean metric typed", metrics.some((m: any) => m.metric_type === "boolean" && m.boolean_value === true));
  check("score metric typed", metrics.some((m: any) => m.metric_type === "score" && typeof m.score_value === "number"));
  check("categorical metric typed", metrics.some((m: any) => m.metric_type === "categorical" && m.categorical_value === "match"));

  console.log(`\n${failures === 0 ? "✅ All sanity checks passed." : `❌ ${failures} sanity check(s) failed.`}`);
  process.exit(failures === 0 ? 0 : 1);
}

main().catch((err) => {
  console.error("\n❌ Sanity check crashed:", err);
  process.exit(1);
});
