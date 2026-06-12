/**
 * DatasetOperations — dataset create/push/pull round-trip over three records.
 * Creates and pushes a dataset, then pulls the same dataset back by name and
 * verifies the round-trip. Writes to the `node-sdk-bootstrap` project.
 *
 *   DD_API_KEY=... DD_APPLICATION_KEY=... npm run runDatasetOperations
 */
import { ExperimentsClient } from "../src/index.js";

function requireEnv(name: string): string {
  const v = process.env[name];
  if (!v || v.trim() === "") {
    console.error(`Missing required env var: ${name}`);
    process.exit(1);
  }
  return v;
}

async function main(): Promise<void> {
  const client = new ExperimentsClient({
    apiKey: requireEnv("DD_API_KEY"),
    applicationKey: requireEnv("DD_APPLICATION_KEY"),
    site: process.env.DD_SITE ?? "datadoghq.com",
    projectName: "node-sdk-bootstrap",
  });

  const datasetName = `dataset-ops-demo-${Date.now()}`;

  console.log("=== Creating dataset and pushing 3 records ===");
  const dataset = client
    .createDataset(datasetName, "Demo of create/push/pull")
    .addRecord({ question: "What is 2+2?" }, "4", { source: "arithmetic", difficulty: "easy" })
    .addRecord({ question: "What is the capital of France?" }, "Paris", {
      source: "geography",
      difficulty: "medium",
    })
    .addRecord(
      { question: "Define photosynthesis." },
      "The process by which plants convert light into energy.",
      { source: "biology", difficulty: "hard" },
    );
  await dataset.push();
  console.log(`Created dataset id : ${dataset.id()}`);
  console.log(`Dataset URL        : ${dataset.url()}`);
  console.log(`Pushed records     : ${dataset.records().length}`);

  console.log("=== Pulling the same dataset back from Datadog ===");
  // Wait (exponential backoff, up to 30s) until all pushed records are readable —
  // LLM Obs reads are eventually consistent right after a write.
  const pulled = await client.pullDataset(datasetName, {
    expectedRecordCount: dataset.records().length,
  });
  console.log(`Pulled dataset id  : ${pulled.id()}`);
  console.log(`Pulled records     : ${pulled.records().length}`);
  pulled.records().forEach((rec, i) => {
    console.log(
      `  [${i}] input=${JSON.stringify(rec.input)}  expected=${JSON.stringify(rec.expectedOutput)}  metadata=${JSON.stringify(rec.metadata)}`,
    );
  });

  if (dataset.id() === pulled.id() && dataset.records().length === pulled.records().length) {
    console.log("Round-trip OK: created id matches pulled id, record counts match.");
  } else {
    console.log(
      `Mismatch: created.id=${dataset.id()} pulled.id=${pulled.id()} created.count=${dataset.records().length} pulled.count=${pulled.records().length}`,
    );
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
