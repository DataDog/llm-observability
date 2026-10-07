import { openai } from "@ai-sdk/openai";
import { defineEvalConfig } from "eve/evals";
import { Datadog } from "eve/evals/reporters";

const datadogExperimentsEnabled =
  process.env.DD_EVE_EVAL_EXPERIMENTS_ENABLED === "true" ||
  (process.env.DD_EVE_EVAL_EXPERIMENTS_ENABLED !== "false" &&
    Boolean(process.env.DD_API_KEY && process.env.DD_APP_KEY));

export default defineEvalConfig({
  judge: {
    model: openai.evaluationModel(
      process.env.OPENAI_EVAL_MODEL ?? "gpt-4o-mini",
    ),
  },
  maxConcurrency: 1,
  timeoutMs: 120_000,
  reporters: datadogExperimentsEnabled
    ? [
        Datadog({
          projectName: "stock-watchlist-agent-eve",
          experimentName: process.env.DD_EVE_EVAL_EXPERIMENT_NAME,
          mlApp: "stock-watchlist-agent-eve",
          service: "stock-watchlist-agent-eve",
          recordInputs: true,
          recordOutputs: true,
          recordExpectedOutputs: true,
          tags: { app: "stock-watchlist-agent-eve" },
        }),
      ]
    : [],
});
