# datadog-llm-experiments-java

A thin Java SDK for running experiments against [Datadog LLM Observability](https://docs.datadoghq.com/llm_observability/). Push a dataset, run your task against each record, ship eval results — and see scored rows in the experiments dashboard.

Built as a workflow layer on top of [`datadog-api-client-java`](https://github.com/DataDog/datadog-api-client-java). Does **not** depend on `dd-trace-java` — no tracer required.

## Status

**Early preview (v0.1.x).** Distributed as source — not on Maven Central, no published binary. You'll build it locally from this directory. The API may change without notice during the preview.

## Prerequisites

You'll need three things installed before any of the steps below work.

### 1. JDK 11+ (Temurin 21 LTS recommended)

```bash
brew install --cask temurin@21
export JAVA_HOME=$(/usr/libexec/java_home -v 21)
java -version    # should print "openjdk version 21.0.x"
```

If you already have a JDK 11 or newer, you can skip the install — just make sure `JAVA_HOME` points at it.

### 2. Maven

Needed once, to build the `datadog-api-client-java` dependency from source (the LLM Observability classes aren't in any published Maven Central release yet).

```bash
brew install maven
mvn -v
```

### 3. Datadog credentials

An API key + an application key with the `llmobs_data_read` and `llmobs_data_write` scopes. Get them from https://app.datadoghq.com/organization-settings/application-keys.

## Setup

### Step 1 — Clone this repo

```bash
git clone https://github.com/DataDog/llm-observability.git
cd llm-observability/experiments/java-sdk
```

### Step 2 — Build `datadog-api-client-java` from source

The published `datadog-api-client-java` (latest release: 2.55.0) does not yet include `LlmObservabilityApi`. Build the default branch at a pinned commit and install it to your local Maven repository:

```bash
git clone https://github.com/DataDog/datadog-api-client-java.git /tmp/datadog-api-client-java
cd /tmp/datadog-api-client-java
git checkout 01929f1fc94c8d42d29c0dd2e011ed80acc76549   # pinned to a known-good commit
mvn install -DskipTests -Dmaven.javadoc.skip=true -Dgpg.skip=true
cd -
```

> **Why pin?** The api-client repo's default branch evolves continuously. Pinning to a specific commit guarantees reproducible builds against the exact snapshot this SDK was tested with. If you skip the `git checkout` step and a later commit changes a constructor signature, your build will fail. Refresh the pin only when you intentionally want to test against a newer api-client.

Verify:

```bash
ls ~/.m2/repository/com/datadoghq/datadog-api-client/2.55.0/
# expect: datadog-api-client-2.55.0.jar  datadog-api-client-2.55.0.pom  (among others)
```

This step **goes away** once `datadog-api-client-java` cuts a release that contains `LlmObservabilityApi`. Our `build.gradle.kts` reads from `mavenLocal()`; once the artifact is on Maven Central, you can delete the local install and the build will resolve from Central instead.

### Step 3 — Configure credentials

Export the env vars in the shell you'll run from:

```bash
export DD_API_KEY=...
export DD_APPLICATION_KEY=...
export DD_SITE=datadoghq.com    # or us3.datadoghq.com, us5.datadoghq.com, eu, etc.
```

For convenience, you can put them in a `.env` file in this directory (it's in `.gitignore`) and source it:

```bash
set -a; source ./.env; set +a
```

## Run the example

From `experiments/java-sdk/`:

```bash
./gradlew run
```

First run downloads Gradle 8.10.2 via the wrapper (~100 MB, cached after) and resolves all dependencies. Subsequent runs take seconds.

Expected output:

```
Experiment URL : https://app.<site>/llm/experiments/<uuid>
Experiment ID  : <uuid>
Rows           : 4
  row 0 status=ok evals={exact_match=..., confidence_score=..., verdict_category=...}
  row 1 ...
  row 2 ...
  row 3 ...

BUILD SUCCESSFUL
```

Open the printed URL — you should see 4 rows with eval scores, status `completed`, and per-row spans showing `input`, `output`, `expected_output`, and `metadata`.

The example source is at [`examples/com/datadog/llmobs/experiments/examples/TopicRelevance.java`](examples/com/datadog/llmobs/experiments/examples/TopicRelevance.java) — adapt it as a starting point for your own experiments.

If TopicRelevance feels like a lot to read, there's a much smaller starter at [`examples/com/datadog/llmobs/experiments/examples/MinimalExperiment.java`](examples/com/datadog/llmobs/experiments/examples/MinimalExperiment.java) — a ~20-line end-to-end with one record and one evaluator. Run it with `./gradlew runMinimalExperiment`.

## Working with datasets directly

If you want to manage datasets without running an experiment — for example, to seed a dataset programmatically, then inspect or re-use it later — the SDK exposes `Dataset.push()` and `client.pullDataset(name)`.

A second runnable example covers this end-to-end:

```bash
./gradlew runDatasetOperations
```

It creates a dataset, pushes 3 records, then pulls the same dataset back and verifies the round-trip. Source: [`examples/com/datadog/llmobs/experiments/examples/DatasetOperations.java`](examples/com/datadog/llmobs/experiments/examples/DatasetOperations.java).

### Create a dataset and push records

```java
Dataset dataset = client.createDataset("my-dataset", "Optional description")
    .addRecord(
        Map.of("question", "What is 2+2?"),
        "4",
        Map.of("source", "arithmetic", "difficulty", "easy")    // optional metadata
    )
    .addRecord(
        Map.of("question", "What is the capital of France?"),
        "Paris"
    );

dataset.push();   // create the dataset in Datadog and push pending records
System.out.println("Dataset id: " + dataset.id());
```

`push()` is idempotent — re-running it after `addRecord(...)` only sends the records added since the last push.

If you skip `push()` and pass the dataset directly to `Experiment.builder(...).dataset(dataset)`, it's pushed automatically when the experiment runs.

### Pull an existing dataset

```java
Dataset existing = client.pullDataset("my-dataset");
for (DatasetRecord r : existing.records()) {
    System.out.println(r.input() + " → " + r.expectedOutput() + " (metadata: " + r.metadata() + ")");
}
```

Use this to resume work on a dataset across sessions, or to inspect datasets the UI / Python SDK created.

> **Note:** push and pull have brief eventual-consistency windows on the server (a few seconds at most). The SDK retries internally for the "push immediately followed by pull" pattern — you don't need to add sleeps in your own code.

## Use the SDK in your own project

After verifying the example works, publish the SDK to your local Maven repo:

```bash
./gradlew publishToMavenLocal
```

That installs `com.datadog.llmobs:datadog-llm-experiments-java:0.1.0-SNAPSHOT` to `~/.m2/repository/`. Reference it from your own build:

### Gradle (`build.gradle.kts`)

```kotlin
repositories {
    mavenLocal()
    mavenCentral()
}

dependencies {
    implementation("com.datadog.llmobs:datadog-llm-experiments-java:0.1.0-SNAPSHOT")
}
```

### Maven (`pom.xml`)

```xml
<dependency>
    <groupId>com.datadog.llmobs</groupId>
    <artifactId>datadog-llm-experiments-java</artifactId>
    <version>0.1.0-SNAPSHOT</version>
</dependency>
```

The `datadog-api-client` dependency is transitive, so you don't need to declare it separately — but it still has to be in your local Maven (from step 2 above).

## Minimal example

```java
import com.datadog.llmobs.experiments.*;
import java.util.Map;

ExperimentsClient client = ExperimentsClient.builder()
    .apiKey(System.getenv("DD_API_KEY"))
    .applicationKey(System.getenv("DD_APPLICATION_KEY"))
    .site("datadoghq.com")
    .projectName("my-experiments")
    .build();

Dataset dataset = client.createDataset("my-dataset")
    .addRecord(
        Map.of("prompt", "What is the capital of France?"),
        "Paris",
        Map.of("source", "synthetic")           // optional metadata
    );

Task<Map<String, Object>, Map<String, Object>> task = (input, config) -> {
    // your task — call an LLM, run a heuristic, etc.
    return Map.of("response", "Paris");
};

Evaluator<Boolean> exactMatch =
    (in, out, expected) -> ((Map<?, ?>) out).get("response").equals(expected);

Experiment<Map<String, Object>, Map<String, Object>> exp =
    Experiment.<Map<String, Object>, Map<String, Object>>builder(client)
        .name("capitals-quiz")
        .dataset(dataset)
        .task(task)
        .evaluator("exact_match", exactMatch)
        .build();

ExperimentResult result = exp.run();
System.out.println("View results: " + result.url());
```

## Concepts

| Type                | Purpose                                                                                                  |
| ------------------- | -------------------------------------------------------------------------------------------------------- |
| `ExperimentsClient` | Auth + site + project context. One per Datadog environment.                                              |
| `Dataset`           | Buffer of records. Auto-pushed to Datadog on first experiment run.                                       |
| `DatasetRecord`     | One row: `input`, optional `expectedOutput`, optional `metadata`.                                        |
| `Task<I, O>`        | The function you run per record.                                                                         |
| `Evaluator<V>`      | Scores a row's output. `Boolean` → boolean metric, `Number` → score metric, anything else → categorical. |
| `Experiment<I, O>`  | Builder + `run()`. Ships spans and metrics. Updates experiment status (`completed` / `failed` / `interrupted`) at the end of the run. |
| `ExperimentResult`  | Returned by `run()` — rows, eval values, dashboard URL.                                                  |

## Limitations (v0.1)

- **No tracer integration.** The SDK emits one root experiment span per row; nested LLM/tool spans require your own in-house tracing.
- **Sequential execution only.** No parallelism flag yet — runs records in a single thread.
- **No multi-run iterations.** Loop in your own code if you need to repeat.
- **No summary evaluators** (cross-row aggregation).
- **No CSV bulk upload.** Add records programmatically; ~5 MB practical limit per push.
- **Not on Maven Central.** Build from this directory.

## How this differs from the Python SDK

The Python SDK (`ddtrace.llmobs`) is bundled inside Datadog's tracer and depends on `dd-trace-py`. That's appropriate for Python where the tracer is widely deployed.

This Java SDK is **standalone** — no tracer dependency, no auto-instrumentation. It calls the Datadog LLM Observability HTTP API directly through `datadog-api-client-java`. Use your existing in-house tracing for everything else; this SDK only emits the spans the experiments dashboard needs to render results.

## Advanced: build a self-contained fat JAR

If you'd rather use the SDK as a single drop-in JAR instead of via `mavenLocal()`:

```bash
./gradlew shadowJar
ls build/libs/datadog-llm-experiments-java-0.1.0-SNAPSHOT-all.jar    # ~28 MB
```

That JAR bundles the SDK plus every transitive dependency. Drop it on your classpath:

```bash
javac -cp build/libs/datadog-llm-experiments-java-0.1.0-SNAPSHOT-all.jar YourApp.java
java  -cp build/libs/datadog-llm-experiments-java-0.1.0-SNAPSHOT-all.jar:. YourApp
```

This is not a supported distribution path for v0.1 — we don't publish this JAR. Use the `mavenLocal()` flow above for everyday integration; the fat JAR is here if you specifically need classpath-only deployment.

## Feedback

This is an early preview. File issues against this repo with feedback — what hurts, what's missing, what surprised you.
