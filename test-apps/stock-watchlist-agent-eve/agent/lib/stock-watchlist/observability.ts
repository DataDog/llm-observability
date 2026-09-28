import { createRequire } from "node:module";
import path from "node:path";
import dotenv from "dotenv";

// eve loads .env/.env.local from the app root before CLI commands. This extra
// load keeps the library usable from tests and direct scripts without copying
// secrets into source-controlled files.
dotenv.config({ path: path.resolve(process.cwd(), ".env") });
dotenv.config({ path: path.resolve(process.cwd(), ".env.local") });

const require = createRequire(import.meta.url);

export const mlApp =
  process.env.DD_LLMOBS_ML_APP || "stock-watchlist-agent-eve";
export const site = process.env.DD_SITE || "datadoghq.com";
export const env = process.env.DD_ENV || process.env.NODE_ENV || "dev";
const explicitEnabled = process.env.DD_LLMOBS_ENABLED;
export const llmobsEnabled =
  explicitEnabled == null
    ? Boolean(process.env.DD_API_KEY)
    : explicitEnabled !== "false";

if (llmobsEnabled) {
  process.env.DD_LLMOBS_ENABLED ??= "true";
  process.env.DD_LLMOBS_ML_APP ??= mlApp;
  process.env.DD_LLMOBS_AGENTLESS_ENABLED ??= "true";
}

let tracer: any = null;
let llmobs: any = null;
let tracerInitialized = false;

function ensureTracer(): void {
  if (tracerInitialized) return;
  tracerInitialized = true;

  if (!llmobsEnabled) {
    return;
  }

  try {
    tracer = require("dd-trace").init({
      service: process.env.DD_SERVICE || "stock-watchlist-agent-eve",
      site,
      env,
      // App-authored spans use standalone LLMObs intake. Eve runtime spans are
      // exported separately by agent/instrumentation/datadog.ts.
      apmTracingEnabled: process.env.DD_APM_TRACING_ENABLED === "true",
      llmobs: llmobsEnabled
        ? {
            mlApp,
            agentlessEnabled:
              process.env.DD_LLMOBS_AGENTLESS_ENABLED !== "false",
          }
        : undefined,
    });
    llmobs = tracer.llmobs;
  } catch (error) {
    console.warn(`LLMObs initialization failed: ${(error as Error).message}`);
  }
}

export function isLLMObsEnabled(): boolean {
  ensureTracer();
  return Boolean(llmobs && llmobs.enabled);
}

export async function traceSpan<T>(
  options: Record<string, unknown>,
  fn: (span: any) => Promise<T> | T,
): Promise<T> {
  if (!isLLMObsEnabled()) {
    return fn(null);
  }
  return llmobs.trace(options, fn);
}

export function annotate(
  spanOrOptions: any,
  maybeOptions?: Record<string, unknown>,
): void {
  if (!isLLMObsEnabled()) return;
  try {
    if (maybeOptions === undefined) {
      llmobs.annotate(spanOrOptions);
    } else {
      llmobs.annotate(spanOrOptions, maybeOptions);
    }
  } catch (error) {
    console.warn(`LLMObs annotation failed: ${(error as Error).message}`);
  }
}

export function exportSpan(span: any): any | null {
  if (!isLLMObsEnabled() || !span) return null;
  try {
    return llmobs.exportSpan(span);
  } catch (error) {
    console.warn(`LLMObs span export failed: ${(error as Error).message}`);
    return null;
  }
}

export function submitEvaluation(
  spanContext: any,
  options: Record<string, unknown>,
): void {
  if (!isLLMObsEnabled() || !spanContext) return;
  llmobs.submitEvaluation(spanContext, options);
}

export function flush(): void {
  if (isLLMObsEnabled()) {
    llmobs.flush();
  }
}
