import { otel } from "eve/instrumentation/otel";

function boolEnv(name: string, fallback: boolean): boolean {
  const value = process.env[name];
  if (value == null || value === "") return fallback;
  return value === "1" || value.toLowerCase() === "true";
}

export default otel({
  functionId: process.env.OTEL_SERVICE_NAME || "stock-watchlist-agent-eve",
  resource: {
    "deployment.environment": process.env.DD_ENV || process.env.NODE_ENV || "dev",
  },
  // Keep prompt and response bodies out of telemetry by default. This policy is
  // evaluated before any destination receives Eve's schema-v4 GenAI spans.
  tracePolicy: () => ({
    emit: true,
    recordInputs: boolEnv("EVE_OTEL_RECORD_INPUTS", false),
    recordOutputs: boolEnv("EVE_OTEL_RECORD_OUTPUTS", false),
  }),
});
