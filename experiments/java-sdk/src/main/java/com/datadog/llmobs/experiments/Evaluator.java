package com.datadog.llmobs.experiments;

/**
 * A user-supplied function that scores a single row's output.
 *
 * <p>The return type {@code V} drives the metric type sent to Datadog:
 * <ul>
 *   <li>{@link Boolean} -&gt; boolean metric
 *   <li>{@link Number}  -&gt; score metric (numeric)
 *   <li>anything else   -&gt; categorical metric (value stringified)
 * </ul>
 *
 * <p>Inputs are passed as {@code Object} — cast inside the lambda to match the types your
 * {@link Task} produces. An exception thrown here is recorded on the metric as an error and
 * does not abort the experiment.
 */
@FunctionalInterface
public interface Evaluator<V> {

    V evaluate(Object input, Object output, Object expectedOutput) throws Exception;
}
