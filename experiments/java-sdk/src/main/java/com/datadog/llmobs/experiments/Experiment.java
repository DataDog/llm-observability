package com.datadog.llmobs.experiments;

import com.datadog.api.client.v2.api.LlmObservabilityApi;
import com.datadog.api.client.v2.model.LLMObsExperimentDataAttributesRequest;
import com.datadog.api.client.v2.model.LLMObsExperimentDataRequest;
import com.datadog.api.client.v2.model.LLMObsExperimentRequest;
import com.datadog.api.client.v2.model.LLMObsExperimentResponse;
import com.datadog.api.client.v2.model.LLMObsExperimentType;
import com.datadog.llmobs.experiments.internal.DirectPost;
import com.datadog.llmobs.experiments.internal.MetricBuilder;
import com.datadog.llmobs.experiments.internal.SpanBuilder;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.UUID;

/**
 * Runs a user's {@link Task} against every record of a {@link Dataset}, executes
 * {@link Evaluator}s on each output, and ships spans + eval metrics to Datadog.
 *
 * <p>Build via {@link #builder(ExperimentsClient)}. Execute with {@link #run()}.
 *
 * <p>v0.1: sequential execution only. Parallelism is a v0.2 add.
 */
public final class Experiment<I, O> {

    private final ExperimentsClient client;
    private final String name;
    private final String description;
    private final Dataset dataset;
    private final Task<I, O> task;
    private final Map<String, Evaluator<?>> evaluators;
    private final Map<String, Object> config;
    private final Map<String, String> tags;

    private String experimentId;

    private Experiment(Builder<I, O> b) {
        this.client = b.client;
        this.name = b.name;
        this.description = b.description == null ? "" : b.description;
        this.dataset = b.dataset;
        this.task = b.task;
        this.evaluators = new LinkedHashMap<>(b.evaluators);
        this.config = b.config == null ? Map.of() : Map.copyOf(b.config);
        this.tags = b.tags == null ? Map.of() : Map.copyOf(b.tags);
    }

    public static <I, O> Builder<I, O> builder(ExperimentsClient client) {
        return new Builder<>(client);
    }

    public String name() { return name; }
    public String experimentId() { return experimentId; }

    public String url() {
        if (experimentId == null) {
            return null;
        }
        return "https://app." + client.site() + "/llm/experiments/" + experimentId;
    }

    @SuppressWarnings("unchecked")
    public ExperimentResult run() {
        // Shutdown hook updates the experiment status to "interrupted" if the JVM is
        // killed (Ctrl-C, SIGTERM, System.exit) while run() is in flight. No-op until
        // experimentId is populated. Removed in the finally block on clean exit.
        Thread interruptHook = new Thread(() -> {
            if (this.experimentId != null) {
                try {
                    updateStatus("interrupted", "Process terminated before run completed");
                } catch (Exception ignored) {
                    // best effort during JVM shutdown
                }
            }
        }, "experiment-interrupt-hook");
        Runtime.getRuntime().addShutdownHook(interruptHook);

        String finalStatus = "completed";
        String finalError = null;

        try {
            LlmObservabilityApi api = client.api();

            String projectId = client.ensureProjectId();
            dataset.ensureCreatedAndPushed(projectId);

            LLMObsExperimentRequest expReq = new LLMObsExperimentRequest(
                new LLMObsExperimentDataRequest()
                    .type(LLMObsExperimentType.EXPERIMENTS)
                    .attributes(
                        new LLMObsExperimentDataAttributesRequest(name, projectId)
                            .datasetId(dataset.id())
                            .description(description)
                            .config(config.isEmpty() ? null : new LinkedHashMap<>(config))
                            .ensureUnique(true)
                    )
            );

            LLMObsExperimentResponse expResp;
            try {
                expResp = api.createLLMObsExperiment(expReq);
            } catch (Exception e) {
                throw new RuntimeException("Failed to create experiment '" + name + "': " + e.getMessage(), e);
            }
            this.experimentId = expResp.getData().getId();

            List<ExperimentResult.Row> rows = new ArrayList<>();
            List<Map<String, Object>> spans = new ArrayList<>();
            List<Map<String, Object>> metrics = new ArrayList<>();

            List<DatasetRecord> records = dataset.records();
            List<String> recordIds = dataset.recordIds();
            int idx = 0;
            for (DatasetRecord record : records) {
                String recordId = idx < recordIds.size() ? recordIds.get(idx) : "";
                String spanId = UUID.randomUUID().toString().replace("-", "");
                String traceId = UUID.randomUUID().toString().replace("-", "");
                long startNs = System.currentTimeMillis() * 1_000_000L;
                long startMonoNs = System.nanoTime();

                Object output = null;
                String errType = null;
                String errMsg = null;
                try {
                    output = task.execute((I) record.input(), config);
                } catch (Exception e) {
                    errType = e.getClass().getName();
                    errMsg = e.getMessage() == null ? e.getClass().getSimpleName() : e.getMessage();
                }

                long durationNs = System.nanoTime() - startMonoNs;
                long endTimestampMs = (startNs + durationNs) / 1_000_000L;

                Map<String, Object> evalValues = new LinkedHashMap<>();
                Map<String, String> evalErrors = new LinkedHashMap<>();
                for (Map.Entry<String, Evaluator<?>> e : evaluators.entrySet()) {
                    try {
                        Object v = e.getValue().evaluate(record.input(), output, record.expectedOutput());
                        evalValues.put(e.getKey(), v);
                    } catch (Exception ex) {
                        String msg = ex.getMessage() == null ? ex.getClass().getSimpleName() : ex.getMessage();
                        evalErrors.put(e.getKey(), msg);
                        evalValues.put(e.getKey(), null);
                    }
                }

                ExperimentResult.Row row = new ExperimentResult.Row(
                    idx, spanId, traceId, startNs, durationNs,
                    record.input(), output, record.expectedOutput(),
                    errType, errMsg, evalValues, evalErrors
                );
                rows.add(row);

                spans.add(SpanBuilder.toSpan(row, record.metadata(), projectId, dataset.id(), recordId, experimentId, name, tags));
                for (Map.Entry<String, Object> eval : evalValues.entrySet()) {
                    metrics.add(MetricBuilder.toMetric(
                        eval.getKey(), eval.getValue(), evalErrors.get(eval.getKey()),
                        spanId, endTimestampMs, experimentId, tags
                    ));
                }
                idx++;
            }

            EventsPoster.post(client, experimentId, spans, metrics);

            return new ExperimentResult(experimentId, rows, url());
        } catch (RuntimeException | Error e) {
            finalStatus = "failed";
            String msg = e.getMessage();
            finalError = msg != null ? msg : e.getClass().getSimpleName();
            throw e;
        } finally {
            try {
                Runtime.getRuntime().removeShutdownHook(interruptHook);
            } catch (IllegalStateException ignored) {
                // JVM is already shutting down; the hook will run on its own
            }
            if (this.experimentId != null) {
                try {
                    updateStatus(finalStatus, finalError);
                } catch (Exception ignored) {
                    // never mask the original exception with a status-update failure
                }
            }
        }
    }

    /**
     * PATCH the experiment with a new status (and optional error message). Hand-rolled HTTP:
     * the generated {@code LLMObsExperimentUpdateDataAttributesRequest} only exposes {@code name}
     * and {@code description} — the spec doesn't surface {@code status} / {@code error} despite
     * the live API accepting them. Once the spec catches up, this can be replaced with
     * {@code api.updateLLMObsExperiment(...)}.
     */
    private void updateStatus(String status, String errorMessage) {
        Map<String, Object> attributes = new HashMap<>();
        attributes.put("status", status);
        if (errorMessage != null) {
            attributes.put("error", errorMessage);
        }

        Map<String, Object> data = new HashMap<>();
        data.put("type", "experiments");
        data.put("id", experimentId);
        data.put("attributes", attributes);

        Map<String, Object> body = new HashMap<>();
        body.put("data", data);

        String path = "/api/v2/llm-obs/v1/experiments/" + experimentId;
        DirectPost.patch(client.site(), path, client.apiKey(), client.applicationKey(), body);
    }

public static final class Builder<I, O> {
        private final ExperimentsClient client;
        private String name;
        private String description;
        private Dataset dataset;
        private Task<I, O> task;
        private final Map<String, Evaluator<?>> evaluators = new LinkedHashMap<>();
        private Map<String, Object> config;
        private Map<String, String> tags;

        Builder(ExperimentsClient client) {
            this.client = Objects.requireNonNull(client, "client");
        }

        public Builder<I, O> name(String name) {
            this.name = name;
            return this;
        }

        public Builder<I, O> description(String description) {
            this.description = description;
            return this;
        }

        public Builder<I, O> dataset(Dataset dataset) {
            this.dataset = dataset;
            return this;
        }

        public Builder<I, O> task(Task<I, O> task) {
            this.task = task;
            return this;
        }

        public Builder<I, O> evaluator(String label, Evaluator<?> evaluator) {
            Objects.requireNonNull(label, "evaluator label");
            Objects.requireNonNull(evaluator, "evaluator");
            this.evaluators.put(label, evaluator);
            return this;
        }

        public Builder<I, O> config(Map<String, Object> config) {
            this.config = config;
            return this;
        }

        public Builder<I, O> tags(Map<String, String> tags) {
            this.tags = tags;
            return this;
        }

        public Experiment<I, O> build() {
            Objects.requireNonNull(name, "name is required");
            Objects.requireNonNull(dataset, "dataset is required");
            Objects.requireNonNull(task, "task is required");
            return new Experiment<>(this);
        }
    }
}
