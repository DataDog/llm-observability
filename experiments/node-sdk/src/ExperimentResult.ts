/**
 * One row of an experiment run — mirrors the Java `ExperimentResult.Row`.
 *
 * Implemented as a class (rather than a plain object) so `isError` is a derived
 * accessor, matching the Java getter `isError()`.
 */
export class Row {
  readonly index: number;
  readonly spanId: string;
  readonly traceId: string;
  readonly startNs: number;
  readonly durationNs: number;
  readonly input: unknown;
  readonly output: unknown;
  readonly expectedOutput: unknown;
  readonly errorType: string | null;
  readonly errorMessage: string | null;
  readonly evaluations: Record<string, unknown>;
  readonly evaluationErrors: Record<string, string>;

  constructor(args: {
    index: number;
    spanId: string;
    traceId: string;
    startNs: number;
    durationNs: number;
    input: unknown;
    output: unknown;
    expectedOutput: unknown;
    errorType: string | null;
    errorMessage: string | null;
    evaluations: Record<string, unknown>;
    evaluationErrors: Record<string, string>;
  }) {
    this.index = args.index;
    this.spanId = args.spanId;
    this.traceId = args.traceId;
    this.startNs = args.startNs;
    this.durationNs = args.durationNs;
    this.input = args.input;
    this.output = args.output;
    this.expectedOutput = args.expectedOutput;
    this.errorType = args.errorType;
    this.errorMessage = args.errorMessage;
    this.evaluations = args.evaluations;
    this.evaluationErrors = args.evaluationErrors;
  }

  get isError(): boolean {
    return this.errorType !== null;
  }
}

/** Returned by `Experiment.run()` — mirrors the Java `ExperimentResult`. */
export class ExperimentResult {
  readonly experimentId: string;
  readonly rows: Row[];
  readonly url: string;

  constructor(experimentId: string, rows: Row[], url: string) {
    this.experimentId = experimentId;
    this.rows = rows;
    this.url = url;
  }
}
