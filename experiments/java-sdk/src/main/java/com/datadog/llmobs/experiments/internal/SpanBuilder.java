package com.datadog.llmobs.experiments.internal;

import com.datadog.llmobs.experiments.ExperimentResult;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Maps a row result to the JSON shape expected by the experiment events endpoint.
 *
 * <p>v0.1 builds raw {@code Map<String, Object>} structures rather than the generated
 * {@code LLMObsExperimentSpan} model. The model class types {@code expected_output} as
 * {@code Map<String, Object>}, which forces a wrapper ({@code {"value": ...}}) around any
 * non-Map value. The live API accepts the raw value (string, number, map, list) directly,
 * matching the Python SDK's behavior — so we bypass the typed model and send raw.
 */
public final class SpanBuilder {

    private SpanBuilder() {}

    public static Map<String, Object> toSpan(
        ExperimentResult.Row row,
        Map<String, Object> recordMetadata,
        String projectId,
        String datasetId,
        String datasetRecordId,
        String experimentId,
        String spanName,
        Map<String, String> userTags
    ) {
        Map<String, Object> meta = new LinkedHashMap<>();
        if (row.input() != null) {
            meta.put("input", row.input());
        }
        if (row.output() != null) {
            meta.put("output", row.output());
        }
        if (row.expectedOutput() != null) {
            meta.put("expected_output", row.expectedOutput());
        }
        if (recordMetadata != null && !recordMetadata.isEmpty()) {
            meta.put("metadata", recordMetadata);
        }
        if (row.isError()) {
            Map<String, Object> err = new LinkedHashMap<>();
            err.put("type", row.errorType());
            err.put("message", row.errorMessage());
            meta.put("error", err);
        }

        Map<String, Object> span = new LinkedHashMap<>();
        span.put("span_id", row.spanId());
        span.put("trace_id", row.traceId());
        span.put("project_id", projectId);
        span.put("dataset_id", datasetId);
        span.put("name", spanName);
        span.put("start_ns", row.startNs());
        span.put("duration", row.durationNs());
        span.put("status", row.isError() ? "error" : "ok");
        span.put("meta", meta);
        span.put("tags", buildTags(userTags, experimentId, datasetId, datasetRecordId));
        return span;
    }

    private static List<String> buildTags(
        Map<String, String> userTags,
        String experimentId,
        String datasetId,
        String datasetRecordId
    ) {
        Map<String, String> merged = new LinkedHashMap<>();
        if (userTags != null) {
            merged.putAll(userTags);
        }
        merged.put("experiment_id", experimentId);
        merged.put("dataset_id", datasetId);
        if (datasetRecordId != null && !datasetRecordId.isEmpty()) {
            merged.put("dataset_record_id", datasetRecordId);
        }
        List<String> out = new ArrayList<>(merged.size());
        for (Map.Entry<String, String> e : merged.entrySet()) {
            out.add(e.getKey() + ":" + e.getValue());
        }
        return out;
    }
}
