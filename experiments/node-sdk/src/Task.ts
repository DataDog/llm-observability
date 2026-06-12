/**
 * User-provided task: `(input, config) -> output`.
 *
 * May be sync or async (return a value or a Promise). Any thrown error / rejected
 * promise is captured per-row in the span without aborting the experiment (see
 * Experiment.run).
 */
export type Task<I = unknown, O = unknown> = (
  input: I,
  config: Record<string, unknown>,
) => O | Promise<O>;
