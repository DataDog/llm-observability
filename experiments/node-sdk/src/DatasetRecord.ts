/**
 * Immutable dataset record: `{ input, expectedOutput?, metadata? }`.
 *
 * `metadata` is propagated raw (unwrapped) to the experiment span's
 * `meta.metadata`.
 */
export class DatasetRecord {
  readonly input: unknown;
  readonly expectedOutput: unknown;
  readonly metadata: Record<string, unknown>;

  constructor(
    input: unknown,
    expectedOutput?: unknown,
    metadata?: Record<string, unknown>,
  ) {
    this.input = input;
    this.expectedOutput = expectedOutput ?? null;
    this.metadata = metadata ?? {};
  }
}
