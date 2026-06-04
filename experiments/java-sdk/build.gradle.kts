plugins {
    `java-library`
    application
    `maven-publish`
    id("com.gradleup.shadow") version "8.3.5"
}

group = "com.datadog.llmobs"
version = "0.1.0-SNAPSHOT"

java {
    sourceCompatibility = JavaVersion.VERSION_11
    targetCompatibility = JavaVersion.VERSION_11
    withSourcesJar()
}

repositories {
    mavenLocal()
    mavenCentral()
}

dependencies {
    // v0.1: vendored build of datadog-api-client from its default branch (includes the
    // LLM Obs API which was not yet in any Maven Central release). Run `mvn install` in
    // a checkout of https://github.com/DataDog/datadog-api-client-java to populate this.
    // Switch back to `mavenCentral()` only once the API client team cuts a release that
    // contains LlmObservabilityApi.
    api("com.datadoghq:datadog-api-client:2.55.0")
}

sourceSets {
    create("examples") {
        java.srcDir("examples")
        compileClasspath += sourceSets.main.get().output + configurations.runtimeClasspath.get()
        runtimeClasspath += output + compileClasspath
    }
}

application {
    mainClass.set("com.datadog.llmobs.experiments.examples.TopicRelevance")
}

tasks.named<JavaExec>("run") {
    classpath = sourceSets["examples"].runtimeClasspath
}

// Standalone dataset example — creates a dataset, pushes records, then pulls them back.
// Useful for trying out the data layer in isolation without running an experiment.
tasks.register<JavaExec>("runDatasetOperations") {
    classpath = sourceSets["examples"].runtimeClasspath
    mainClass.set("com.datadog.llmobs.experiments.examples.DatasetOperations")
}

// Smallest end-to-end experiment — one record, one evaluator, one URL.
// The simplest copy-paste starting point.
tasks.register<JavaExec>("runMinimalExperiment") {
    classpath = sourceSets["examples"].runtimeClasspath
    mainClass.set("com.datadog.llmobs.experiments.examples.MinimalExperiment")
}

tasks.test {
    useJUnitPlatform()
}

// Publish the SDK JAR + sources to the local Maven repo so a consuming project can
// reference the SDK from its own build.gradle.kts / pom.xml via mavenLocal().
//
// Workflow:
//   $ ./gradlew publishToMavenLocal
//   # then in the consuming project's build:
//   #   repositories { mavenLocal(); mavenCentral() }
//   #   dependencies { implementation("com.datadog.llmobs:datadog-llm-experiments-java:0.1.0-SNAPSHOT") }
publishing {
    publications {
        create<MavenPublication>("maven") {
            from(components["java"])
        }
    }
}

// Optional: produce a single self-contained JAR (~28 MB) containing the SDK + every
// transitive dependency. v0.1 does not ship this artifact via releases — it exists for
// users who prefer classpath-only integration. Run `./gradlew shadowJar` to build it.
//
// `mergeServiceFiles()` is non-optional: Jersey and Jackson use META-INF/services for SPI
// discovery, and naive shading would clobber those files instead of concatenating them.
tasks.shadowJar {
    archiveClassifier.set("all")
    mergeServiceFiles()
}
