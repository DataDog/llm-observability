package com.datadog.llmobs.experiments.examples;

import com.datadog.llmobs.experiments.Dataset;
import com.datadog.llmobs.experiments.DatasetRecord;
import com.datadog.llmobs.experiments.ExperimentsClient;

import java.util.Map;

/**
 * Standalone dataset example: create a dataset, push a few records, then pull the
 * same dataset back to verify the round-trip. No experiment is run here — this is
 * intentionally just the data layer, useful for getting a feel for {@link Dataset}
 * in isolation.
 *
 * <p>Run with: {@code ./gradlew runDatasetOperations}
 */
public final class DatasetOperations {

    public static void main(String[] args) {
        String apiKey = requireEnv("DD_API_KEY");
        String appKey = requireEnv("DD_APPLICATION_KEY");
        String site = System.getenv().getOrDefault("DD_SITE", "datadoghq.com");

        ExperimentsClient client = ExperimentsClient.builder()
            .apiKey(apiKey)
            .applicationKey(appKey)
            .site(site)
            .projectName("java-sdk-bootstrap")
            .build();

        String datasetName = "dataset-ops-demo-" + System.currentTimeMillis();

        // -- Create + push -----------------------------------------------------
        System.out.println("=== Creating dataset and pushing 3 records ===");
        Dataset created = client.createDataset(datasetName, "Demo of create/push/pull")
            .addRecord(
                Map.of("question", "What is 2+2?"),
                "4",
                Map.of("source", "arithmetic", "difficulty", "easy")
            )
            .addRecord(
                Map.of("question", "What is the capital of France?"),
                "Paris",
                Map.of("source", "geography", "difficulty", "medium")
            )
            .addRecord(
                Map.of("question", "Define photosynthesis."),
                "The process by which plants convert light into energy.",
                Map.of("source", "biology", "difficulty", "hard")
            );

        created.push();   // explicit — no Experiment.run() needed
        System.out.println("Created dataset id : " + created.id());
        System.out.println("Pushed records     : " + created.records().size());
        System.out.println();

        // -- Pull back ----------------------------------------------------------
        System.out.println("=== Pulling the same dataset back from Datadog ===");
        Dataset pulled = client.pullDataset(datasetName);
        System.out.println("Pulled dataset id  : " + pulled.id());
        System.out.println("Pulled records     : " + pulled.records().size());
        int idx = 0;
        for (DatasetRecord r : pulled.records()) {
            System.out.println("  [" + idx + "] input=" + r.input()
                + "  expected=" + r.expectedOutput()
                + "  metadata=" + r.metadata());
            idx++;
        }

        // -- Verify -------------------------------------------------------------
        System.out.println();
        if (created.id().equals(pulled.id()) && created.records().size() == pulled.records().size()) {
            System.out.println("Round-trip OK: created id matches pulled id, record counts match.");
        } else {
            System.out.println("Mismatch: created.id=" + created.id() + " pulled.id=" + pulled.id()
                + " created.count=" + created.records().size()
                + " pulled.count=" + pulled.records().size());
        }
    }

    private static String requireEnv(String name) {
        String v = System.getenv(name);
        if (v == null || v.isBlank()) {
            System.err.println("Missing required env var: " + name);
            System.exit(1);
        }
        return v;
    }
}
