/**
 * Exponential-backoff retry helper.
 *
 * Used by read-after-write operations (e.g. `pullDataset`) to absorb the
 * eventual-consistency lag between pushing data and it becoming queryable on the
 * backend. Delays double each round (250ms, 500ms, 1s, 2s, …) and are capped per
 * round; the loop stops once the cumulative elapsed time reaches `maxTotalMs`.
 */

export interface BackoffOptions {
  /** Hard ceiling on total time spent waiting (sleeps + attempts). Default 30s. */
  maxTotalMs?: number;
  /** First sleep, doubled each round. Default 250ms. */
  baseDelayMs?: number;
  /** Per-round sleep cap. Default 8s. */
  maxDelayMs?: number;
}

const sleep = (ms: number): Promise<void> =>
  new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Run `attempt` repeatedly until it returns `true` (success) or the time budget
 * is exhausted. `attempt` is always invoked at least once. Returns `true` if it
 * succeeded, `false` if the budget ran out first.
 */
export async function retryWithBackoff(
  attempt: () => Promise<boolean>,
  options: BackoffOptions = {},
): Promise<boolean> {
  const maxTotalMs = options.maxTotalMs ?? 30_000;
  const baseDelayMs = options.baseDelayMs ?? 250;
  const maxDelayMs = options.maxDelayMs ?? 8_000;

  const start = Date.now();
  let delay = baseDelayMs;

  for (;;) {
    if (await attempt()) return true;

    const elapsed = Date.now() - start;
    const remaining = maxTotalMs - elapsed;
    if (remaining <= 0) return false;

    await sleep(Math.min(delay, maxDelayMs, remaining));
    delay *= 2;
  }
}
