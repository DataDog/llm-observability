/**
 * TopicRelevance — direct port of the Java SDK's `TopicRelevance` example.
 *
 * Identical to the Java example in every respect — same four dataset records
 * (inputs, expected outputs, metadata), same evaluators, config and tags — except
 * it writes to the Node-specific project `node-sdk-bootstrap` (the Java example
 * uses `java-sdk-bootstrap`), so the two SDKs' artifacts stay in separate projects
 * for easy side-by-side comparison.
 *
 *   DD_API_KEY=... DD_APPLICATION_KEY=... [DD_SITE=datadoghq.com] npm run runTopicRelevance
 */
import { pathToFileURL } from "node:url";
import { Dataset, Experiment, ExperimentsClient } from "../src/index.js";

function requireEnv(name: string): string {
  const v = process.env[name];
  if (!v || v.trim() === "") {
    console.error(`Missing required env var: ${name}`);
    process.exit(1);
  }
  return v;
}

/**
 * Mock "task": does the prompt contain any keyword from the (comma-separated)
 * topics list? Mirrors the Java `keywordOverlap` exactly.
 */
function keywordOverlap(prompt: string, topics: string): boolean {
  const p = prompt.toLowerCase();
  for (const topic of topics.split(",")) {
    for (const word of topic.trim().toLowerCase().split(/\s+/)) {
      if (word !== "" && p.includes(word)) return true;
    }
  }
  return false;
}

type Input = { prompt: string; topics: string };

/**
 * Build the dataset + experiment (the part shared by the runnable `main` and the
 * parity test). `datasetName` is parameterized so tests can use a stable name;
 * `main` passes a timestamped one, exactly like the Java example.
 */
export function buildTopicRelevance(
  client: ExperimentsClient,
  datasetName = `topic-relevance-demo-${Date.now()}`,
): { dataset: Dataset; experiment: Experiment<Input, Record<string, unknown>> } {
  const dataset = client
    .createDataset(datasetName, "Java SDK v0.1 smoke test")
    .addRecord(
      { prompt: "I love hiking in the mountains on weekends.", topics: "outdoor, travel" },
      "true",
      { source: "synthetic", difficulty: "easy" },
    )
    .addRecord(
      { prompt: "Explain quantum entanglement in two sentences.", topics: "outdoor, travel" },
      "false",
      { source: "synthetic", difficulty: "easy" },
    )
    .addRecord(
      { prompt: "Best Italian restaurants in Brooklyn?", topics: "food, nyc" },
      "true",
      { source: "user-report", difficulty: "medium", reviewer: "alex" },
    )
    .addRecord(
      { prompt: "How do I configure nginx for HTTPS?", topics: "food, nyc" },
      "false",
      { source: "user-report", difficulty: "hard", reviewer: "alex" },
    );

  const experiment = Experiment.builder<Input, Record<string, unknown>>(client)
    .name("topic-relevance-demo")
    .dataset(dataset)
    .task((input) => {
      const overlap = keywordOverlap(input.prompt, input.topics);
      return { response: String(overlap), confidence: overlap ? 0.85 : 0.65 };
    })
    // boolean: did the predicted response match the expected label?
    .evaluator("exact_match", (_input, output, expected) => (output as any).response === expected)
    // score: the model's confidence
    .evaluator("confidence_score", (_input, output) => Number((output as any).confidence))
    // categorical: human-readable verdict
    .evaluator("verdict_category", (_input, output) =>
      (output as any).response === "true" ? "in-topic" : "off-topic",
    )
    .config({ approach: "keyword-overlap", version: "v0.1" })
    .tags({ variant: "java-v0.1", owner: "design-partner-bootstrap" })
    .build();

  return { dataset, experiment };
}

async function main(): Promise<void> {
  const client = new ExperimentsClient({
    apiKey: requireEnv("DD_API_KEY"),
    applicationKey: requireEnv("DD_APPLICATION_KEY"),
    site: process.env.DD_SITE ?? "datadoghq.com",
    projectName: "node-sdk-bootstrap",
  });

  const { dataset, experiment } = buildTopicRelevance(client);
  const result = await experiment.run();

  console.log(`Dataset URL    : ${dataset.url()}`);
  console.log(`Experiment URL : ${result.url}`);
  console.log(`Experiment ID  : ${result.experimentId}`);
  console.log(`Rows           : ${result.rows.length}`);
  for (const row of result.rows) {
    console.log(
      `  row ${row.index} status=${row.isError ? "error" : "ok"} evals=${JSON.stringify(row.evaluations)}`,
    );
  }
}

// Run only when invoked directly (not when imported by the parity test).
if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((err) => {
    console.error(err);
    process.exit(1);
  });
}
