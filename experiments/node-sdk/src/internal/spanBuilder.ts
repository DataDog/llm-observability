import type { Row } from "../ExperimentResult.js";

/**
 * Build the raw span map for one experiment row (the LLM Obs experiment span
 * wire format).
 *
 * Tag precedence: the auto tags `experiment_id`, `dataset_id`,
 * `dataset_record_id` always win over user-supplied tags on key conflict.
 * `metadata` lives only in `meta.metadata` — never in tags.
 */
export function toSpan(
  row: Row,
  metadata: Record<string, unknown>,
  experimentId: string,
  projectId: string,
  datasetId: string,
  datasetRecordId: string,
  spanName: string,
  userTags: Record<string, string>,
): Record<string, unknown> {
  const meta: Record<string, unknown> = {
    input: row.input ?? null,
    output: row.output ?? null,
    expected_output: row.expectedOutput ?? null,
  };

  // metadata: raw / unwrapped. Only emitted when present (W4).
  if (metadata && Object.keys(metadata).length > 0) {
    meta.metadata = metadata;
  }

  if (row.isError) {
    meta.error = {
      type: row.errorType ?? "",
      message: row.errorMessage ?? "",
      stack: "",
    };
  }

  return {
    span_id: row.spanId,
    trace_id: row.traceId,
    project_id: projectId,
    dataset_id: datasetId,
    name: spanName,
    start_ns: row.startNs,
    duration: row.durationNs,
    status: row.isError ? "error" : "ok",
    meta,
    tags: buildTags(userTags, experimentId, datasetId, datasetRecordId),
  };
}

function buildTags(
  userTags: Record<string, string>,
  experimentId: string,
  datasetId: string,
  datasetRecordId: string,
): string[] {
  const tags = new Map<string, string>();
  // User tags first, so the auto tags below overwrite on conflict.
  for (const [k, v] of Object.entries(userTags ?? {})) {
    tags.set(k, `${k}:${v}`);
  }
  tags.set("experiment_id", `experiment_id:${experimentId}`);
  tags.set("dataset_id", `dataset_id:${datasetId}`);
  tags.set("dataset_record_id", `dataset_record_id:${datasetRecordId}`);
  return [...tags.values()];
}
