package com.datadog.llmobs.experiments.internal;

import com.fasterxml.jackson.databind.ObjectMapper;

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;

/**
 * Direct HTTP POST helper for endpoints where the generated client's serialized payload
 * doesn't match the live API (specifically: the {@code "type"} discriminator field).
 *
 * <p>Several v0.1 endpoints declare a {@code "type"} value in the spec that disagrees with
 * what the live API expects:
 * <ul>
 *   <li>records push: spec says {@code "records"}, API wants {@code "datasets"}</li>
 *   <li>events post:  spec says {@code "events"},  API wants {@code "experiments"}</li>
 * </ul>
 * <p>The generated {@code ModelEnum} subclasses are locked to the spec values and cannot
 * be coerced, so we serialize the body ourselves with the corrected discriminator.
 * Delete this helper and switch back to the generated methods once the spec aligns.
 */
public final class DirectPost {

    private static final ObjectMapper MAPPER = new ObjectMapper();
    private static final HttpClient HTTP = HttpClient.newBuilder()
        .connectTimeout(Duration.ofSeconds(10))
        .build();

    private DirectPost() {}

    /**
     * POST a JSON body to the given LLM Obs path against {@code https://api.<site>}.
     *
     * @param site             the Datadog site (no scheme, no subdomain) — e.g. {@code datadoghq.com}
     * @param path             the path including leading slash
     * @param apiKey           DD-API-KEY header value
     * @param applicationKey   DD-APPLICATION-KEY header value
     * @param body             a Map / List / scalar — anything Jackson can serialize
     * @throws RuntimeException on non-2xx response
     */
    public static void post(
        String site,
        String path,
        String apiKey,
        String applicationKey,
        Object body
    ) {
        send("POST", site, path, apiKey, applicationKey, body);
    }

    /** PATCH variant — used for partial updates like experiment status transitions. */
    public static void patch(
        String site,
        String path,
        String apiKey,
        String applicationKey,
        Object body
    ) {
        send("PATCH", site, path, apiKey, applicationKey, body);
    }

    /** GET variant — returns the response body. Use when the generated client's response
     * model doesn't match the live API shape. */
    public static String get(
        String site,
        String path,
        String apiKey,
        String applicationKey
    ) {
        return send("GET", site, path, apiKey, applicationKey, null);
    }

    /** Same as {@link #post} but returns the response body for callers that need to parse it. */
    public static String postReturning(
        String site,
        String path,
        String apiKey,
        String applicationKey,
        Object body
    ) {
        return send("POST", site, path, apiKey, applicationKey, body);
    }

    private static String send(
        String method,
        String site,
        String path,
        String apiKey,
        String applicationKey,
        Object body
    ) {
        URI uri = URI.create("https://api." + site + path);

        HttpRequest.Builder builder = HttpRequest.newBuilder()
            .uri(uri)
            .timeout(Duration.ofSeconds(30))
            .header("DD-API-KEY", apiKey)
            .header("DD-APPLICATION-KEY", applicationKey);

        HttpRequest.BodyPublisher publisher;
        if (body != null) {
            byte[] json;
            try {
                json = MAPPER.writeValueAsBytes(body);
            } catch (Exception e) {
                throw new RuntimeException("Failed to serialize request body for " + path, e);
            }
            publisher = HttpRequest.BodyPublishers.ofByteArray(json);
            builder.header("Content-Type", "application/json");
        } else {
            publisher = HttpRequest.BodyPublishers.noBody();
        }

        HttpRequest req = builder
            .method(method, publisher)
            .build();

        HttpResponse<String> resp;
        try {
            resp = HTTP.send(req, HttpResponse.BodyHandlers.ofString());
        } catch (Exception e) {
            throw new RuntimeException(method + " " + path + " failed", e);
        }

        int code = resp.statusCode();
        if (code < 200 || code >= 300) {
            throw new RuntimeException(
                method + " " + path + " failed: HTTP " + code + " " + resp.body()
            );
        }
        return resp.body();
    }
}
