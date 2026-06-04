package com.datadog.llmobs.experiments.examples;

import com.datadog.llmobs.experiments.Dataset;
import com.datadog.llmobs.experiments.Evaluator;
import com.datadog.llmobs.experiments.Experiment;
import com.datadog.llmobs.experiments.ExperimentResult;
import com.datadog.llmobs.experiments.ExperimentsClient;
import com.datadog.llmobs.experiments.Task;

import java.util.Map;

/**
 * End-to-end example: create a dataset, run a task per record, ship results to Datadog.
 *
 * <p>Reads {@code DD_API_KEY}, {@code DD_APPLICATION_KEY}, and (optional) {@code DD_SITE} from
 * the environment. The task here is a mock keyword-overlap classifier so the example needs no
 * external LLM dependency — swap in a real OpenAI / Anthropic call for your own use.
 *
 * <p>Run with: {@code ./gradlew run}
 */
public final class TopicRelevance {

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

        long ts = System.currentTimeMillis();
        // Each record can carry optional metadata as a Map<String, Object>. Metadata is stored
        // on the dataset record in Datadog and is visible in the dataset view of the dashboard.
        // The span for each row links back to the record via the dataset_record_id tag, so the
        // UI can surface this metadata when drilling into a row's results.
        Dataset dataset = client.createDataset("topic-relevance-demo-" + ts, "Java SDK v0.1 smoke test")
            .addRecord(
                Map.of("prompt", "I love hiking in the mountains on weekends.", "topics", "outdoor, travel"),
                "true",
                Map.of("source", "synthetic", "difficulty", "easy")
            )
            .addRecord(
                Map.of("prompt", "Explain quantum entanglement in two sentences.", "topics", "outdoor, travel"),
                "false",
                Map.of("source", "synthetic", "difficulty", "easy")
            )
            .addRecord(
                Map.of("prompt", "Best Italian restaurants in Brooklyn?", "topics", "food, nyc"),
                "true",
                Map.of("source", "user-report", "difficulty", "medium", "reviewer", "alex")
            )
            .addRecord(
                Map.of("prompt", "How do I configure nginx for HTTPS?", "topics", "food, nyc"),
                "false",
                Map.of("source", "user-report", "difficulty", "hard", "reviewer", "alex")
            );

        Task<Map<String, Object>, Map<String, Object>> classify = (input, config) -> {
            String prompt = (String) input.get("prompt");
            String topics = (String) input.get("topics");
            boolean overlap = keywordOverlap(prompt, topics);
            return Map.of(
                "response", String.valueOf(overlap),
                "confidence", overlap ? 0.85 : 0.65
            );
        };

        Evaluator<Boolean> exactMatch = (input, output, expected) -> {
            Map<?, ?> out = (Map<?, ?>) output;
            return out.get("response").equals(expected);
        };

        Evaluator<Double> confidenceScore = (input, output, expected) -> {
            Map<?, ?> out = (Map<?, ?>) output;
            return ((Number) out.get("confidence")).doubleValue();
        };

        Evaluator<String> verdict = (input, output, expected) -> {
            Map<?, ?> out = (Map<?, ?>) output;
            return "true".equals(out.get("response")) ? "in-topic" : "off-topic";
        };

        Experiment<Map<String, Object>, Map<String, Object>> exp =
            Experiment.<Map<String, Object>, Map<String, Object>>builder(client)
                .name("topic-relevance-demo")
                .dataset(dataset)
                .task(classify)
                .evaluator("exact_match", exactMatch)
                .evaluator("confidence_score", confidenceScore)
                .evaluator("verdict_category", verdict)
                .config(Map.of("approach", "keyword-overlap", "version", "v0.1"))
                .tags(Map.of("variant", "java-v0.1", "owner", "design-partner-bootstrap"))
                .build();

        ExperimentResult result = exp.run();
        System.out.println();
        System.out.println("Experiment URL : " + result.url());
        System.out.println("Experiment ID  : " + result.experimentId());
        System.out.println("Rows           : " + result.rows().size());
        for (ExperimentResult.Row row : result.rows()) {
            System.out.println("  row " + row.index()
                + " status=" + (row.isError() ? "error" : "ok")
                + " evals=" + row.evaluations());
        }
    }

    private static boolean keywordOverlap(String prompt, String topics) {
        String p = prompt.toLowerCase();
        for (String topic : topics.split(",")) {
            for (String word : topic.trim().toLowerCase().split("\\s+")) {
                if (!word.isEmpty() && p.contains(word)) {
                    return true;
                }
            }
        }
        return false;
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
