/**
 * MinimalExperiment — direct port of the Java SDK's `MinimalExperiment`.
 *
 * Same single record, task and evaluator as the Java example; writes to the
 * Node-specific project `node-sdk-minimal` (Java uses `java-sdk-minimal`).
 *
 *   DD_API_KEY=... DD_APPLICATION_KEY=... npm run runMinimalExperiment
 */
import { Experiment, ExperimentsClient } from "../src/index.js";

function requireEnv(name: string): string {
  const v = process.env[name];
  if (!v || v.trim() === "") {
    console.error(`Missing required env var: ${name}`);
    process.exit(1);
  }
  return v;
}

type Input = { name: string };

async function main(): Promise<void> {
  const client = new ExperimentsClient({
    apiKey: requireEnv("DD_API_KEY"),
    applicationKey: requireEnv("DD_APPLICATION_KEY"),
    site: process.env.DD_SITE ?? "datadoghq.com",
    projectName: "node-sdk-minimal",
  });

  const dataset = client
    .createDataset(`minimal-${Date.now()}`)
    .addRecord({ name: "World" }, "Hello, World");

  const result = await Experiment.builder<Input, string>(client)
    .name("minimal-experiment")
    .dataset(dataset)
    .task((input) => `Hello, ${input.name}`)
    .evaluator("exact_match", (_input, output, expected) => output === expected)
    .build()
    .run();

  console.log(`Dataset URL    : ${dataset.url()}`);
  console.log(`Experiment URL : ${result.url}`);
  console.log(`Experiment ID  : ${result.experimentId}`);
  console.log(`Rows           : ${result.rows.length}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
