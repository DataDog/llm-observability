/**
 * Build one metric map per evaluator per row (the LLM Obs metric wire format).
 *
 * The metric type is inferred from the evaluator's return value:
 *   - boolean -> "boolean", value under `boolean_value`
 *   - number  -> "score",   value under `score_value`
 *   - other   -> "categorical", value stringified under `categorical_value`
 *
 * If `errorMessage` is set the metric carries an `error` instead of a value.
 * The auto tag `experiment_id` always wins over user tags.
 */
export function toMetric(
  label: string,
  value: unknown,
  errorMessage: string | null,
  spanId: string,
  timestampMs: number,
  experimentId: string,
  userTags: Record<string, string>,
): Record<string, unknown> {
  const metric: Record<string, unknown> = {
    label,
    span_id: spanId,
    timestamp_ms: timestampMs,
    tags: buildTags(userTags, experimentId),
  };

  if (errorMessage !== null) {
    // Type is irrelevant on an errored metric, but the wire shape still needs one.
    metric.metric_type = "categorical";
    metric.error = { message: errorMessage };
    return metric;
  }

  const type = inferType(value);
  metric.metric_type = type;
  switch (type) {
    case "boolean":
      metric.boolean_value = value as boolean;
      break;
    case "score":
      metric.score_value = value as number;
      break;
    default:
      metric.categorical_value = stringify(value);
      break;
  }
  return metric;
}

function inferType(value: unknown): "boolean" | "score" | "categorical" {
  if (typeof value === "boolean") return "boolean";
  if (typeof value === "number" && Number.isFinite(value)) return "score";
  return "categorical";
}

function stringify(value: unknown): string {
  if (value === null || value === undefined) return "";
  if (typeof value === "string") return value;
  if (typeof value === "object") return JSON.stringify(value);
  return String(value);
}

function buildTags(userTags: Record<string, string>, experimentId: string): string[] {
  const tags = new Map<string, string>();
  for (const [k, v] of Object.entries(userTags ?? {})) {
    tags.set(k, `${k}:${v}`);
  }
  tags.set("experiment_id", `experiment_id:${experimentId}`);
  return [...tags.values()];
}
