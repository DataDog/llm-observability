package com.datadog.llmobs.experiments.internal;

import com.datadog.api.client.v2.model.AnyValue;

import java.util.Map;

/**
 * Maps Java values to the API client's {@link AnyValue} JSON wrapper.
 *
 * <p>v0.1 supports String, Boolean, Number, Map&lt;String,Object&gt;, and passthrough of an
 * existing {@code AnyValue}. Other types throw — extend here if you need them.
 */
public final class AnyValues {

    private AnyValues() {}

    @SuppressWarnings("unchecked")
    public static AnyValue of(Object value) {
        if (value == null) {
            return null;
        }
        if (value instanceof AnyValue) {
            return (AnyValue) value;
        }
        if (value instanceof String) {
            return new AnyValue((String) value);
        }
        if (value instanceof Boolean) {
            return new AnyValue((Boolean) value);
        }
        if (value instanceof Number) {
            return new AnyValue(((Number) value).doubleValue());
        }
        if (value instanceof Map) {
            return new AnyValue((Map<String, Object>) value);
        }
        throw new IllegalArgumentException(
            "Unsupported value type for AnyValue conversion: " + value.getClass().getName()
                + ". v0.1 supports String, Boolean, Number, Map<String,Object>."
        );
    }
}
