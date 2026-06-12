/**
 * User-provided evaluator: `(input, output, expectedOutput) -> value`.
 *
 * May be sync or async. The return type drives the metric type (see MetricBuilder):
 *   - boolean -> boolean metric
 *   - number  -> score metric
 *   - anything else -> categorical metric (stringified)
 *
 * A thrown error / rejected promise is captured per-evaluation as an `error` on
 * the metric; the experiment continues.
 */
export type Evaluator<V = unknown> = (
  input: unknown,
  output: unknown,
  expectedOutput: unknown,
) => V | Promise<V>;
