import { createHash } from "node:crypto";

import { OTLPTraceExporter } from "@opentelemetry/exporter-trace-otlp-http";
import type { ReadableSpan } from "@opentelemetry/sdk-trace-base";
import { disableInstrumentation } from "eve/instrumentation";
import { otelIntegration, type SpanExporter } from "eve/instrumentation/otel";

const DATADOG_OTLP_HOST_BY_SITE: Record<string, string> = {
  "datadoghq.com": "otlp.datadoghq.com",
  "us3.datadoghq.com": "otlp.us3.datadoghq.com",
  "us5.datadoghq.com": "otlp.us5.datadoghq.com",
  "datadoghq.eu": "otlp.datadoghq.eu",
  "ap1.datadoghq.com": "otlp.ap1.datadoghq.com",
  "ap2.datadoghq.com": "otlp.ap2.datadoghq.com",
  "ddog-gov.com": "otlp.ddog-gov.com",
};

const LLMOBS_APM_TRACE_ID_NAMESPACE = "f47ac10b58cc4372a5670e02b2c3d479";
const SAFE_ATTRIBUTE_KEYS = new Set([
  "agent.framework.name",
  "agent.framework.version",
  "agent.name",
  "agent.step.attempt",
  "agent.step.index",
  "agent.trace.schema.version",
  "agent.turn.id",
  "agent.turn.outcome",
  "agent.turn.sequence",
  "deployment.environment",
  "eve.environment",
  "eve.session.id",
  "eve.version",
  "gen_ai.agent.name",
  "gen_ai.conversation.id",
  "gen_ai.operation.name",
  "gen_ai.provider.name",
  "gen_ai.request.model",
  "gen_ai.response.finish_reasons",
  "gen_ai.response.id",
  "operation.name",
  "resource.name",
  "service.name",
  "service.version",
  "stock_watchlist.channel",
  "stock_watchlist.session_id",
  "stock_watchlist.step_index",
  "stock_watchlist.turn_id",
  "stock_watchlist.turn_sequence",
  "telemetry.sdk.language",
  "telemetry.sdk.name",
  "telemetry.sdk.version",
]);

function boolEnv(name: string, fallback: boolean): boolean {
  const value = process.env[name];
  if (value == null || value === "") return fallback;
  return value === "1" || value.toLowerCase() === "true";
}

function datadogOtlpTraceEndpoint(): string | undefined {
  if (process.env.OTEL_EXPORTER_OTLP_TRACES_ENDPOINT) {
    return process.env.OTEL_EXPORTER_OTLP_TRACES_ENDPOINT;
  }

  if (process.env.OTEL_EXPORTER_OTLP_ENDPOINT) {
    return `${process.env.OTEL_EXPORTER_OTLP_ENDPOINT.replace(/\/$/, "")}/v1/traces`;
  }

  if (!process.env.DD_API_KEY) return undefined;

  const site = process.env.DD_SITE || "datadoghq.com";
  const host = DATADOG_OTLP_HOST_BY_SITE[site] || `otlp.${site}`;
  return `https://${host}/v1/traces`;
}

function traceHeaders(): Record<string, string> | undefined {
  // Let the exporter parse the standard OTEL header environment variable.
  if (process.env.OTEL_EXPORTER_OTLP_HEADERS) return undefined;
  if (process.env.DD_API_KEY) return { "DD-API-KEY": process.env.DD_API_KEY };
  return undefined;
}

function sanitizedEndpoint(endpoint: string): string {
  try {
    const parsed = new URL(endpoint);
    return `${parsed.protocol}//${parsed.host}${parsed.pathname}`;
  } catch {
    return "<invalid endpoint>";
  }
}

function decimalSpanId(spanId: string | undefined): string | undefined {
  if (!spanId || !/^[0-9a-f]{16}$/i.test(spanId)) return undefined;
  return BigInt(`0x${spanId}`).toString(10);
}

function expectedLlmobsTraceId(apmTraceId: string): string | undefined {
  if (!/^[0-9a-f]{32}$/i.test(apmTraceId)) return undefined;
  const canonicalApmTraceId = apmTraceId.toLowerCase().slice(-16).padStart(32, "0");
  const hash = createHash("sha1")
    .update(Buffer.from(LLMOBS_APM_TRACE_ID_NAMESPACE, "hex"))
    .update(canonicalApmTraceId)
    .digest()
    .subarray(0, 16);
  hash[6] = (hash[6]! & 0x0f) | 0x50;
  hash[8] = (hash[8]! & 0x3f) | 0x80;
  return hash.toString("hex");
}

function safeAttributes(attributes: Readonly<Record<string, unknown>>): {
  attributes: Record<string, unknown>;
  omittedAttributeKeys: string[];
} {
  const safe: Record<string, unknown> = {};
  const omitted: string[] = [];
  for (const [key, value] of Object.entries(attributes)) {
    if (SAFE_ATTRIBUTE_KEYS.has(key) || key.startsWith("gen_ai.usage.")) {
      safe[key] = value;
    } else {
      omitted.push(key);
    }
  }
  return { attributes: safe, omittedAttributeKeys: omitted.sort() };
}

function auditedTraceExporter(exporter: SpanExporter, endpoint: string): SpanExporter {
  if (!boolEnv("EVE_OTEL_DEBUG_PAYLOADS", false)) return exporter;

  let batch = 0;
  return {
    export(spans, callback) {
      batch += 1;
      try {
        const readableSpans = spans as ReadableSpan[];
        const llmobsEligibleSpans = readableSpans.filter(
          (span) => typeof span.attributes["gen_ai.operation.name"] === "string",
        );
        const payload = {
          schema: "sanitized-otlp-trace-export-v1",
          batch,
          endpoint: sanitizedEndpoint(endpoint),
          site: process.env.DD_SITE || "datadoghq.com",
          submittedSpanCount: spans.length,
          llmobsEligibleSpanCount: llmobsEligibleSpans.length,
          spans: llmobsEligibleSpans.map((span) => {
            const context = span.spanContext();
            const parentSpanId = span.parentSpanContext?.spanId;
            return {
              traceId: context.traceId,
              spanId: context.spanId,
              parentSpanId,
              expectedLlmobsTraceId: expectedLlmobsTraceId(context.traceId),
              expectedLlmobsSpanId: decimalSpanId(context.spanId),
              expectedLlmobsParentId: decimalSpanId(parentSpanId),
              name: span.name,
              kind: span.kind,
              status: span.status,
              startTime: span.startTime,
              duration: span.duration,
              resource: safeAttributes(span.resource.attributes),
              scope: {
                name: span.instrumentationScope.name,
                version: span.instrumentationScope.version,
              },
              ...safeAttributes(span.attributes),
              events: span.events.map((event) => ({
                name: event.name,
                time: event.time,
                attributeKeys: Object.keys(event.attributes ?? {}).sort(),
              })),
            };
          }),
        };
        // Never print headers, API keys, prompt/response bodies, tool arguments,
        // or event attribute values. The IDs and structural fields are enough
        // to find the source APM trace and its converted LLMObs event.
        console.log(`[EVE_OTEL_PAYLOAD] ${JSON.stringify(payload)}`);
      } catch (error) {
        console.warn(`[EVE_OTEL_PAYLOAD_ERROR] ${(error as Error).message}`);
      }

      exporter.export(spans, (result) => {
        console.log(
          `[EVE_OTEL_EXPORT] ${JSON.stringify({
            batch,
            endpoint: sanitizedEndpoint(endpoint),
            submittedSpanCount: spans.length,
            llmobsEligibleSpanCount: spans.filter(
              (span) =>
                typeof (span as ReadableSpan).attributes["gen_ai.operation.name"] === "string",
            ).length,
            resultCode: result.code,
          })}`,
        );
        callback(result);
      });
    },
    shutdown: () => exporter.shutdown(),
  };
}

const endpoint = datadogOtlpTraceEndpoint();

if (!endpoint) {
  console.warn(
    "Datadog OTel export disabled: set DD_API_KEY or OTEL_EXPORTER_OTLP_TRACES_ENDPOINT.",
  );
}

export default endpoint
  ? otelIntegration({
      traceExporter: auditedTraceExporter(
        new OTLPTraceExporter({
          url: endpoint,
          headers: traceHeaders(),
        }),
        endpoint,
      ),
      runtimeContext: ({ channel, session, step, turn }) => ({
        "stock_watchlist.channel": channel.kind,
        "stock_watchlist.session_id": session.id,
        "stock_watchlist.turn_id": turn.id,
        "stock_watchlist.turn_sequence": turn.sequence,
        "stock_watchlist.step_index": step.index,
      }),
    })
  : disableInstrumentation();
