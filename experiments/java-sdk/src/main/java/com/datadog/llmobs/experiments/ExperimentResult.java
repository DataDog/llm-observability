package com.datadog.llmobs.experiments;

import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * Result of an {@link Experiment#run()}.
 *
 * <p>Each {@link Row} carries the per-record input, output, error, and the evaluator results.
 * {@link #url()} points at the experiment in the Datadog dashboard.
 */
public final class ExperimentResult {

    private final String experimentId;
    private final List<Row> rows;
    private final String url;

    ExperimentResult(String experimentId, List<Row> rows, String url) {
        this.experimentId = experimentId;
        this.rows = Collections.unmodifiableList(rows);
        this.url = url;
    }

    public String experimentId() {
        return experimentId;
    }

    public List<Row> rows() {
        return rows;
    }

    public String url() {
        return url;
    }

    /** Per-record result. */
    public static final class Row {
        private final int index;
        private final String spanId;
        private final String traceId;
        private final long startNs;
        private final long durationNs;
        private final Object input;
        private final Object output;
        private final Object expectedOutput;
        private final String errorType;
        private final String errorMessage;
        private final Map<String, Object> evaluations;
        private final Map<String, String> evaluationErrors;

        Row(int index, String spanId, String traceId, long startNs, long durationNs,
            Object input, Object output, Object expectedOutput,
            String errorType, String errorMessage,
            Map<String, Object> evaluations, Map<String, String> evaluationErrors) {
            this.index = index;
            this.spanId = spanId;
            this.traceId = traceId;
            this.startNs = startNs;
            this.durationNs = durationNs;
            this.input = input;
            this.output = output;
            this.expectedOutput = expectedOutput;
            this.errorType = errorType;
            this.errorMessage = errorMessage;
            this.evaluations = Collections.unmodifiableMap(evaluations);
            this.evaluationErrors = Collections.unmodifiableMap(evaluationErrors);
        }

        public int index() { return index; }
        public String spanId() { return spanId; }
        public String traceId() { return traceId; }
        public long startNs() { return startNs; }
        public long durationNs() { return durationNs; }
        public Object input() { return input; }
        public Object output() { return output; }
        public Object expectedOutput() { return expectedOutput; }
        public boolean isError() { return errorType != null; }
        public String errorType() { return errorType; }
        public String errorMessage() { return errorMessage; }
        public Map<String, Object> evaluations() { return evaluations; }
        public Map<String, String> evaluationErrors() { return evaluationErrors; }
    }
}
