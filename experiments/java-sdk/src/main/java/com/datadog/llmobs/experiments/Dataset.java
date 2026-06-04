package com.datadog.llmobs.experiments;

import com.datadog.api.client.v2.api.LlmObservabilityApi;
import com.datadog.api.client.v2.model.LLMObsDatasetDataAttributesRequest;
import com.datadog.api.client.v2.model.LLMObsDatasetDataRequest;
import com.datadog.api.client.v2.model.LLMObsDatasetRequest;
import com.datadog.api.client.v2.model.LLMObsDatasetResponse;
import com.datadog.api.client.v2.model.LLMObsDatasetType;
import com.datadog.llmobs.experiments.internal.DirectPost;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * A dataset of records to run an experiment against.
 *
 * <p>Construct with {@link ExperimentsClient#createDataset(String)}. Add records with
 * {@link #addRecord}. The dataset is auto-created in Datadog and records are pushed when
 * {@link Experiment#run(int)} executes — no explicit push step in v0.1.
 */
public final class Dataset {

    private static final ObjectMapper JSON = new ObjectMapper();

    private final ExperimentsClient client;
    private final String name;
    private final String description;
    private final List<DatasetRecord> records = new ArrayList<>();
    private final List<String> recordIds = new ArrayList<>();

    private String id;
    private String projectId;
    private int pushedCount;

    Dataset(ExperimentsClient client, String name, String description) {
        this.client = Objects.requireNonNull(client, "client");
        this.name = Objects.requireNonNull(name, "name");
        this.description = description == null ? "" : description;
    }

    /**
     * Construct a Dataset that already exists in Datadog, pre-populated with records.
     * Package-private — used by {@link ExperimentsClient#pullDataset(String)}.
     */
    static Dataset fromExisting(
        ExperimentsClient client,
        String name,
        String description,
        String datasetId,
        String projectId,
        List<DatasetRecord> records,
        List<String> recordIds
    ) {
        Dataset d = new Dataset(client, name, description);
        d.id = datasetId;
        d.projectId = projectId;
        d.records.addAll(records);
        d.recordIds.addAll(recordIds);
        d.pushedCount = records.size();
        return d;
    }

    /**
     * Idempotently create this dataset in Datadog (if not already) and push any locally-buffered
     * records that haven't been pushed yet. Safe to call multiple times — only new records are
     * pushed on subsequent calls.
     */
    public void push() {
        ensureCreatedAndPushed(client.ensureProjectId());
    }

    public Dataset addRecord(DatasetRecord record) {
        Objects.requireNonNull(record, "record");
        records.add(record);
        return this;
    }

    public Dataset addRecord(Object input, Object expectedOutput) {
        return addRecord(new DatasetRecord(input, expectedOutput));
    }

    public Dataset addRecord(Object input, Object expectedOutput, Map<String, Object> metadata) {
        return addRecord(new DatasetRecord(input, expectedOutput, metadata));
    }

    public String name() {
        return name;
    }

    public List<DatasetRecord> records() {
        return Collections.unmodifiableList(records);
    }

    /** Datadog record IDs in the same order as {@link #records()}. Empty entries for any
     * records whose IDs we failed to parse from the push response. Package-private —
     * consumed by {@link Experiment#run()}. */
    List<String> recordIds() {
        return recordIds;
    }

    /** Datadog dataset ID. Null until the dataset has been created in Datadog. */
    public String id() {
        return id;
    }

    /** Datadog project ID. Null until the dataset has been pushed under a project. */
    public String projectId() {
        return projectId;
    }

    /**
     * Idempotently creates the dataset in Datadog under the given project and pushes any
     * pending local records. Called by {@link Experiment#run(int)}; users do not call this.
     */
    void ensureCreatedAndPushed(String projectId) {
        this.projectId = projectId;
        LlmObservabilityApi api = client.api();

        if (id == null) {
            LLMObsDatasetRequest req = new LLMObsDatasetRequest(
                new LLMObsDatasetDataRequest()
                    .type(LLMObsDatasetType.DATASETS)
                    .attributes(new LLMObsDatasetDataAttributesRequest(name).description(description))
            );
            try {
                LLMObsDatasetResponse resp = api.createLLMObsDataset(projectId, req);
                this.id = resp.getData().getId();
            } catch (Exception e) {
                throw new RuntimeException("Failed to create dataset '" + name + "': " + e.getMessage(), e);
            }
        }

        // Push only newly-added records (idempotent across multiple run() calls).
        if (pushedCount < records.size()) {
            // Hand-rolled HTTP: the generated client sends "type":"records" but the API
            // expects "type":"datasets" for this child-resource endpoint. Same spec/API
            // mismatch we hit for experiment events. Delete once aligned.
            List<DatasetRecord> toPush = records.subList(pushedCount, records.size());
            List<Map<String, Object>> items = new ArrayList<>(toPush.size());
            for (DatasetRecord r : toPush) {
                Map<String, Object> item = new LinkedHashMap<>();
                item.put("input", r.input());
                if (r.expectedOutput() != null) {
                    item.put("expected_output", r.expectedOutput());
                }
                if (!r.metadata().isEmpty()) {
                    item.put("metadata", r.metadata());
                }
                items.add(item);
            }
            Map<String, Object> attributes = new HashMap<>();
            attributes.put("records", items);

            Map<String, Object> body = new HashMap<>();
            Map<String, Object> data = new HashMap<>();
            data.put("type", "datasets");
            data.put("attributes", attributes);
            body.put("data", data);

            String path = "/api/v2/llm-obs/v1/" + projectId + "/datasets/" + id + "/records";
            String responseBody;
            try {
                responseBody = DirectPost.postReturning(
                    client.site(), path, client.apiKey(), client.applicationKey(), body
                );
            } catch (Exception e) {
                throw new RuntimeException("Failed to push records to dataset '" + name + "': " + e.getMessage(), e);
            }
            try {
                JsonNode dataNode = JSON.readTree(responseBody).path("data");
                if (dataNode.isArray()) {
                    for (JsonNode rec : dataNode) {
                        recordIds.add(rec.path("id").asText(""));
                    }
                }
            } catch (Exception e) {
                // Non-fatal: we lose dataset_record_id tagging for these rows, but the
                // experiment still runs. Log + continue with empty IDs for the new rows.
                int gap = (records.size() - pushedCount) - (recordIds.size() - pushedCount);
                for (int i = 0; i < gap; i++) {
                    recordIds.add("");
                }
            }
            pushedCount = records.size();
        }
    }
}
