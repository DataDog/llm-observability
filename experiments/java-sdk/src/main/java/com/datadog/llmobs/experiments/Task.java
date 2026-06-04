package com.datadog.llmobs.experiments;

import java.util.Map;

/**
 * The user-supplied function run for each dataset record during an experiment.
 *
 * <p>{@code I} is the input type (typically {@code Map<String, Object>} matching your
 * {@link DatasetRecord#input()} shape, but free-form). {@code O} is the output type.
 * {@code config} is the per-experiment configuration map (model name, temperature, etc.).
 *
 * <p>Any thrown exception is captured per-row in the span and does not abort the experiment.
 */
@FunctionalInterface
public interface Task<I, O> {

    O execute(I input, Map<String, Object> config) throws Exception;
}
