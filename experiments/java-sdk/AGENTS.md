# Notes for coding agents

You're working in `experiments/java-sdk/` of `DataDog/llm-observability` — a thin Java SDK for Datadog LLM Observability experiments. This file gives you the context you'd otherwise have to infer from source.

## Read this first

- The SDK is in **early preview (v0.1.x)**. Public API may change between releases. Don't optimize for long-term API stability.
- **Build-from-source only.** Not on Maven Central. Setup is documented in [README.md](README.md) — follow it literally, don't skip steps.

## Do NOT remove or "clean up"

These look like cleanup candidates but they exist for documented reasons:

- The `git checkout 01929f1fc94c8d42d29c0dd2e011ed80acc76549` step in README §2 pins `datadog-api-client-java` to a tested commit. That repo's default branch changes constructor signatures frequently — without the pin, the SDK build will randomly break.
- `internal/DirectPost.java` and its uses in `EventsPoster.java`, `Dataset.java` (records push), `ExperimentsClient.pullDataset`, and `Experiment.updateStatus`. These hand-roll HTTP for endpoints where the generated client + OpenAPI spec disagree with the live API. Each call site has an inline comment explaining the specific drift.
- Raw-map JSON building in `internal/SpanBuilder.java` and `internal/MetricBuilder.java`. We bypass the typed model classes because the spec types `expected_output` as `Map<String,Object>` (forcing a `{"value": ...}` wrapper for scalar inputs) and omits `meta.metadata` entirely.
- The `JacksonAnnotationIntrospector` override in `ExperimentsClient.java` (the `hasRequiredMarker → false` override). Required because some response fields the spec marks `required: true` are not always returned by the live API.

## Eventual consistency

`pullDataset(name)` retries internally for both the dataset-lookup call and the records-list call (separate index lag windows on the server). Push-then-immediately-pull may take up to ~6 seconds total. Don't wrap it in a shorter timeout and don't add your own retry layer on top — you'll double-retry.

## Where new code goes

- Public SDK types: `src/main/java/com/datadog/llmobs/experiments/*.java`
- Internal helpers: `src/main/java/com/datadog/llmobs/experiments/internal/*.java`
- Runnable examples: `examples/com/datadog/llmobs/experiments/examples/*.java`

Each new example needs a corresponding `tasks.register<JavaExec>("runX")` block in `build.gradle.kts` so it can be invoked.

## Credentials and runtime config

Required env vars:

- `DD_API_KEY` — 32-char API key
- `DD_APPLICATION_KEY` — 40-char app key with `llmobs_data_read` and `llmobs_data_write` scopes
- `DD_SITE` — optional, default `datadoghq.com`. Set to `datad0g.com` for staging.

Never commit `.env`. It's in `.gitignore`.

## How to verify a change

After modifying SDK code:

```bash
./gradlew build                      # compile (no tests yet)
./gradlew runMinimalExperiment       # ~15-line minimal end-to-end (fastest)
./gradlew runDatasetOperations       # dataset create / push / pull round-trip
./gradlew run                        # TopicRelevance — multi-evaluator full example
```

If credentials aren't set, only `./gradlew build` is safe — the runnable examples hit the live Datadog API.

## API design patterns to follow

- New API endpoints: prefer the generated `client.api()` methods. Hand-roll via `DirectPost` only if the generated path is broken by spec drift (and only after verifying via curl against a real org).
- New methods on `Dataset` / `Experiment` / `ExperimentsClient`: package-private workhorses + thin public facades. Builder pattern for anything with more than 2-3 fields.
- Evaluator return types drive metric typing automatically: `Boolean` → boolean metric, `Number` → score, anything else → categorical. Don't introduce explicit `metric_type` parameters unless something forces it.
- Status of experiments is updated in `Experiment.run()`'s `finally` block — keep that semantics intact when refactoring the run loop.

## Known follow-ups

Several spec-drift workarounds in this SDK should disappear once `DataDog/datadog-api-spec` is updated. The catalogue lives in the PR description (W1–W10). When closing one of those workarounds, also remove the corresponding comment in the source — search for "spec drift" or "workaround" in `src/`.
