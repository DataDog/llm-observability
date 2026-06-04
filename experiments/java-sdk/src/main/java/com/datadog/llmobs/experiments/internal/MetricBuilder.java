package com.datadog.llmobs.experiments.internal;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * Maps an evaluator's returned value to the JSON shape expected by the experiment events
 * endpoint. Pure mapping, no I/O. See {@link SpanBuilder} for the rationale behind raw maps.
 */
public final class MetricBuilder {

    private MetricBuilder() {}

    public static Map<String, Object> toMetric(
        String label,
        Object value,
        String errorMessage,
        String spanId,
        long timestampMs,
        String experimentId,
        Map<String, String> userTags
    ) {
        String metricType = inferType(value);

        Map<String, Object> metric = new LinkedHashMap<>();
        metric.put("label", label);
        metric.put("metric_type", metricType);
        metric.put("span_id", spanId);
        metric.put("timestamp_ms", timestampMs);
        metric.put("tags", buildTags(userTags, experimentId));

        if (errorMessage != null) {
            Map<String, Object> err = new LinkedHashMap<>();
            err.put("message", errorMessage);
            metric.put("error", err);
            return metric;
        }
        if (value == null) {
            return metric;
        }

        switch (metricType) {
            case "boolean":
                metric.put("boolean_value", value);
                break;
            case "score":
                metric.put("score_value", ((Number) value).doubleValue());
                break;
            default:
                metric.put("categorical_value", String.valueOf(value));
        }
        return metric;
    }

    private static String inferType(Object value) {
        if (value instanceof Boolean) {
            return "boolean";
        }
        if (value instanceof Number) {
            return "score";
        }
        return "categorical";
    }

    private static List<String> buildTags(Map<String, String> userTags, String experimentId) {
        Map<String, String> merged = new LinkedHashMap<>();
        if (userTags != null) {
            merged.putAll(userTags);
        }
        merged.put("experiment_id", experimentId);
        List<String> out = new ArrayList<>(merged.size());
        for (Map.Entry<String, String> e : merged.entrySet()) {
            out.add(e.getKey() + ":" + e.getValue());
        }
        return out;
    }
}
