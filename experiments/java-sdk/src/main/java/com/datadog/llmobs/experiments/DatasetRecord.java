package com.datadog.llmobs.experiments;

import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * An immutable input record for a {@link Dataset}.
 *
 * <p>{@code input} is typically a {@code Map<String, Object>} for structured inputs, or a
 * {@code String} for simple inputs. {@code expectedOutput} is optional and follows the same
 * rules. {@code metadata} attaches arbitrary fields surfaced in the dashboard.
 */
public final class DatasetRecord {

    private final Object input;
    private final Object expectedOutput;
    private final Map<String, Object> metadata;

    public DatasetRecord(Object input, Object expectedOutput) {
        this(input, expectedOutput, null);
    }

    public DatasetRecord(Object input, Object expectedOutput, Map<String, Object> metadata) {
        this.input = Objects.requireNonNull(input, "input is required");
        this.expectedOutput = expectedOutput;
        this.metadata = metadata == null
            ? Collections.emptyMap()
            : Collections.unmodifiableMap(new HashMap<>(metadata));
    }

    public Object input() {
        return input;
    }

    public Object expectedOutput() {
        return expectedOutput;
    }

    public Map<String, Object> metadata() {
        return metadata;
    }
}
