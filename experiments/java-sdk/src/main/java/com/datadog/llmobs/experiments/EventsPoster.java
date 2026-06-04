package com.datadog.llmobs.experiments;

import com.datadog.llmobs.experiments.internal.DirectPost;

import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Posts experiment spans + metrics via direct HTTP. Workaround for a spec/API mismatch:
 * the generated {@code LlmObservabilityApi.createLLMObsExperimentEvents} serializes
 * {@code "type":"events"} but the API expects {@code "type":"experiments"}.
 *
 * <p>Builds the request body from raw {@code Map} structures (see {@code SpanBuilder} and
 * {@code MetricBuilder}) so we can send shapes the typed model classes can't express
 * — notably a raw value for {@code expected_output} rather than the model's
 * {@code Map<String, Object>} wrapper.
 *
 * <p>Delete this class and switch to the generated method once the spec/API are aligned.
 * Package-private so it does not appear in the public SDK surface.
 */
final class EventsPoster {

    private EventsPoster() {}

    static void post(
        ExperimentsClient client,
        String experimentId,
        List<Map<String, Object>> spans,
        List<Map<String, Object>> metrics
    ) {
        Map<String, Object> attributes = new HashMap<>();
        attributes.put("spans", spans);
        attributes.put("metrics", metrics);

        Map<String, Object> data = new HashMap<>();
        data.put("type", "experiments");
        data.put("attributes", attributes);

        Map<String, Object> body = new HashMap<>();
        body.put("data", data);

        String path = "/api/v2/llm-obs/v1/experiments/" + experimentId + "/events";
        DirectPost.post(client.site(), path, client.apiKey(), client.applicationKey(), body);
    }
}
