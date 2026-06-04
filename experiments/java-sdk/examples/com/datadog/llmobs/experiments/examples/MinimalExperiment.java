package com.datadog.llmobs.experiments.examples;

import com.datadog.llmobs.experiments.Dataset;
import com.datadog.llmobs.experiments.Experiment;
import com.datadog.llmobs.experiments.ExperimentResult;
import com.datadog.llmobs.experiments.ExperimentsClient;

import java.util.Map;

/**
 * Smallest end-to-end experiment: one record, one evaluator, one URL printed. Use this as
 * a copy-paste starting point. Other examples in this directory add evaluators, metadata,
 * multi-record datasets, and dataset round-tripping on top of this skeleton.
 *
 * <p>Run with: {@code ./gradlew runMinimalExperiment}
 */
public final class MinimalExperiment {

    public static void main(String[] args) {
        ExperimentsClient client = ExperimentsClient.builder()
            .apiKey(System.getenv("DD_API_KEY"))
            .applicationKey(System.getenv("DD_APPLICATION_KEY"))
            .site(System.getenv().getOrDefault("DD_SITE", "datadoghq.com"))
            .projectName("java-sdk-minimal")
            .build();

        Dataset dataset = client.createDataset("minimal-" + System.currentTimeMillis())
            .addRecord(Map.of("name", "World"), "Hello, World");

        ExperimentResult result = Experiment.<Map<String, Object>, String>builder(client)
            .name("minimal-experiment")
            .dataset(dataset)
            .task((input, config) -> "Hello, " + input.get("name"))
            .evaluator("exact_match", (in, out, expected) -> out.equals(expected))
            .build()
            .run();

        System.out.println("Experiment URL: " + result.url());
    }
}
