package com.datadog.llmobs.experiments;

import com.datadog.api.client.ApiClient;
import com.datadog.api.client.v2.api.LlmObservabilityApi;
import com.datadog.api.client.v2.api.LlmObservabilityApi.ListLLMObsDatasetsOptionalParameters;
import com.datadog.api.client.v2.model.LLMObsDatasetDataResponse;
import com.datadog.api.client.v2.model.LLMObsDatasetsResponse;
import com.datadog.llmobs.experiments.internal.DirectPost;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.datadog.api.client.v2.model.LLMObsProjectDataAttributesRequest;
import com.datadog.api.client.v2.model.LLMObsProjectDataRequest;
import com.datadog.api.client.v2.model.LLMObsProjectRequest;
import com.datadog.api.client.v2.model.LLMObsProjectType;
import com.fasterxml.jackson.databind.DeserializationFeature;
import com.fasterxml.jackson.databind.introspect.AnnotatedMember;
import com.fasterxml.jackson.databind.introspect.JacksonAnnotationIntrospector;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Entry point for the Datadog LLM Experiments Java SDK.
 *
 * <p>Wraps {@code datadog-api-client-java}'s {@link ApiClient} and {@link LlmObservabilityApi},
 * applying the LLM Observability auth, site, and unstable-operation opt-ins the SDK needs.
 *
 * <p>Construct via {@link #builder()}.
 */
public final class ExperimentsClient {

    private static final String[] UNSTABLE_OPERATIONS = new String[] {
        "v2.createLLMObsProject",
        "v2.listLLMObsProjects",
        "v2.createLLMObsDataset",
        "v2.listLLMObsDatasets",
        "v2.createLLMObsDatasetRecords",
        "v2.updateLLMObsDatasetRecords",
        "v2.listLLMObsDatasetRecords",
        "v2.deleteLLMObsDatasetRecords",
        "v2.createLLMObsExperiment",
        "v2.createLLMObsExperimentEvents",
    };

    private static final ObjectMapper JSON = new ObjectMapper();

    private final ApiClient apiClient;
    private final LlmObservabilityApi api;
    private final String projectName;
    private final String site;
    private final String apiKey;
    private final String appKey;
    private volatile String cachedProjectId;

    private ExperimentsClient(Builder b) {
        // Use a fresh ApiClient rather than getDefaultApiClient(): the latter eagerly reads
        // DD_SITE from the environment and validates it against a fixed allow-list of public
        // sites, which rejects staging hosts like datad0g.com before we get a chance to
        // configure anything.
        ApiClient ac = new ApiClient();

        // Server index 2 is the "any Datadog deployment" variant with an unrestricted site
        // variable (the public-sites enum lives at index 0). Using index 2 lets us point at
        // staging, custom, or any future site without bumping the client library.
        ac.setServerIndex(2);

        // The LLM Obs endpoints are still being shaped — some response fields the OpenAPI
        // spec marks "required" are actually omitted by the live API (e.g. project.description).
        // Strict deserialization fails those responses. Override Jackson's annotation
        // introspector to claim no property is required, which relaxes response parsing
        // without affecting request serialization. We also disable FAIL_ON_MISSING_CREATOR_PROPERTIES
        // as a belt-and-suspenders. Remove once the spec aligns with the live API.
        ac.getJSON().getMapper()
            .setAnnotationIntrospector(new JacksonAnnotationIntrospector() {
                @Override
                public Boolean hasRequiredMarker(AnnotatedMember m) {
                    return Boolean.FALSE;
                }
            });
        ac.getJSON().getMapper()
            .configure(DeserializationFeature.FAIL_ON_MISSING_CREATOR_PROPERTIES, false);

        Map<String, String> authKeys = new HashMap<>();
        authKeys.put("apiKeyAuth", b.apiKey);
        authKeys.put("appKeyAuth", b.appKey);
        ac.configureApiKeys(authKeys);

        Map<String, String> serverVars = new HashMap<>();
        serverVars.put("site", b.site);
        ac.setServerVariables(serverVars);

        for (String op : UNSTABLE_OPERATIONS) {
            ac.setUnstableOperationEnabled(op, true);
        }

        this.apiClient = ac;
        this.api = new LlmObservabilityApi(ac);
        this.projectName = b.projectName;
        this.site = b.site;
        this.apiKey = b.apiKey;
        this.appKey = b.appKey;
    }

    // Package-private accessors used by EventsPoster (hand-rolled HTTP workaround).
    // Not public — credentials should not leak beyond the package.
    String apiKey() {
        return apiKey;
    }

    String applicationKey() {
        return appKey;
    }

    /** Generated API surface for any endpoint this SDK does not wrap. */
    public LlmObservabilityApi api() {
        return api;
    }

    /** Create a new {@link Dataset} bound to this client. Not pushed to Datadog until used. */
    public Dataset createDataset(String name) {
        return new Dataset(this, name, null);
    }

    public Dataset createDataset(String name, String description) {
        return new Dataset(this, name, description);
    }

    /**
     * Pull an existing dataset by name. Returns a {@link Dataset} pre-populated with every
     * record (and each record's stable ID) from Datadog. Useful for inspecting datasets the
     * UI / Python SDK created, or for resuming work on a dataset across sessions.
     *
     * @throws IllegalArgumentException if no dataset with that name exists in the project.
     */
    public Dataset pullDataset(String name) {
        Objects.requireNonNull(name, "name");
        String projectId = ensureProjectId();

        // The list endpoint is index-backed and lags briefly behind dataset creation. Retry
        // up to a few times with backoff so a "push immediately followed by pull" pattern
        // (common in examples and tests) works without the caller having to sleep.
        List<LLMObsDatasetDataResponse> hits = null;
        long[] backoffMs = {0L, 1000L, 2000L, 3000L};
        for (long delay : backoffMs) {
            if (delay > 0) {
                try {
                    Thread.sleep(delay);
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
            LLMObsDatasetsResponse listResp;
            try {
                listResp = api.listLLMObsDatasets(projectId, new ListLLMObsDatasetsOptionalParameters().filterName(name));
            } catch (Exception e) {
                throw new RuntimeException("Failed to list datasets in project '" + projectName + "': " + e.getMessage(), e);
            }
            hits = listResp.getData();
            if (hits != null && !hits.isEmpty()) {
                break;
            }
        }

        if (hits == null || hits.isEmpty()) {
            throw new IllegalArgumentException("Dataset '" + name + "' not found in project '" + projectName + "' (after retries)");
        }

        LLMObsDatasetDataResponse dsData = hits.get(0);
        String datasetId = dsData.getId();
        String description = dsData.getAttributes() != null && dsData.getAttributes().getDescription() != null
            ? dsData.getAttributes().getDescription()
            : "";

        // Hand-rolled GET for records: the generated LLMObsDatasetRecordDataResponse expects
        // flat fields (getInput, getExpectedOutput, getMetadata directly on the item) but the
        // live API wraps them inside data[].attributes — yet another spec/API mismatch on the
        // read side. Parse JSON manually until the spec catches up.
        //
        // Records have their own eventual-consistency window separate from the dataset list,
        // so retry independently if we see 0 records (could be true empty, or could be lag).
        List<DatasetRecord> records = new ArrayList<>();
        List<String> recordIds = new ArrayList<>();
        for (long delay : backoffMs) {
            if (delay > 0) {
                try {
                    Thread.sleep(delay);
                } catch (InterruptedException ie) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
            String recordsBody = DirectPost.get(
                site,
                "/api/v2/llm-obs/v1/" + projectId + "/datasets/" + datasetId + "/records",
                apiKey,
                appKey
            );
            records.clear();
            recordIds.clear();
            try {
                JsonNode root = JSON.readTree(recordsBody);
                JsonNode dataArray = root.path("data");
                if (dataArray.isArray()) {
                    for (JsonNode item : dataArray) {
                        recordIds.add(item.path("id").asText(""));
                        JsonNode attrs = item.path("attributes");
                        Object input = JSON.treeToValue(attrs.path("input"), Object.class);
                        Object expected = attrs.has("expected_output") && !attrs.path("expected_output").isNull()
                            ? JSON.treeToValue(attrs.path("expected_output"), Object.class)
                            : null;
                        @SuppressWarnings("unchecked")
                        Map<String, Object> metadata = attrs.has("metadata") && !attrs.path("metadata").isNull()
                            ? JSON.treeToValue(attrs.path("metadata"), Map.class)
                            : Collections.emptyMap();
                        records.add(new DatasetRecord(
                            input != null ? input : Collections.emptyMap(),
                            expected,
                            metadata
                        ));
                    }
                }
            } catch (Exception e) {
                throw new RuntimeException("Failed to parse records for dataset '" + name + "': " + e.getMessage(), e);
            }
            if (!records.isEmpty()) {
                break;
            }
        }

        return Dataset.fromExisting(this, name, description, datasetId, projectId, records, recordIds);
    }

    /**
     * Idempotently resolve the project ID for {@link #projectName}. Cached after first call —
     * the backend dedupes projects by name, so repeated calls return the same ID, but caching
     * avoids unnecessary round-trips when the same client is reused across many push/pull/run
     * operations.
     *
     * <p>Package-private: used by {@link Dataset#push()} and {@link Experiment#run()}.
     */
    String ensureProjectId() {
        if (cachedProjectId != null) {
            return cachedProjectId;
        }
        LLMObsProjectRequest req = new LLMObsProjectRequest(
            new LLMObsProjectDataRequest()
                .type(LLMObsProjectType.PROJECTS)
                .attributes(new LLMObsProjectDataAttributesRequest(projectName))
        );
        try {
            this.cachedProjectId = api.createLLMObsProject(req).getData().getId();
        } catch (Exception e) {
            throw new RuntimeException("Failed to create or get project '" + projectName + "': " + e.getMessage(), e);
        }
        return cachedProjectId;
    }

    public String projectName() {
        return projectName;
    }

    public String site() {
        return site;
    }

    public static Builder builder() {
        return new Builder();
    }

    public static final class Builder {
        private String apiKey;
        private String appKey;
        private String site = "datadoghq.com";
        private String projectName;

        public Builder apiKey(String apiKey) {
            this.apiKey = apiKey;
            return this;
        }

        public Builder applicationKey(String appKey) {
            this.appKey = appKey;
            return this;
        }

        public Builder site(String site) {
            this.site = site;
            return this;
        }

        public Builder projectName(String projectName) {
            this.projectName = projectName;
            return this;
        }

        public ExperimentsClient build() {
            Objects.requireNonNull(apiKey, "apiKey is required");
            Objects.requireNonNull(appKey, "applicationKey is required");
            Objects.requireNonNull(projectName, "projectName is required");
            Objects.requireNonNull(site, "site is required");
            return new ExperimentsClient(this);
        }
    }
}
