import type { Dataset } from "./Dataset.js";
import type { Evaluator } from "./Evaluator.js";
import { ExperimentResult, Row } from "./ExperimentResult.js";
import type { ExperimentsClient } from "./ExperimentsClient.js";
import type { Task } from "./Task.js";
import * as http from "./internal/http.js";
import { appBase } from "./internal/http.js";
import { hexId } from "./internal/ids.js";
import { toMetric } from "./internal/metricBuilder.js";
import { toSpan } from "./internal/spanBuilder.js";

/**
 * Builder + `run()` orchestration.
 *
 * v0.1 runs sequentially (one row at a time), emits one root span per dataset
 * row, and posts all spans + metrics in a single events call.
 */
export class Experiment<I = unknown, O = unknown> {
  private readonly client: ExperimentsClient;
  private readonly _name: string;
  private readonly _description: string;
  private readonly dataset: Dataset;
  private readonly task: Task<I, O>;
  private readonly evaluators: Map<string, Evaluator>;
  private readonly config: Record<string, unknown>;
  private readonly tags: Record<string, string>;
  private _experimentId: string | null = null;

  private constructor(b: ExperimentBuilder<I, O>) {
    if (!b._name) throw new Error("Experiment name is required");
    if (!b._dataset) throw new Error("Experiment dataset is required");
    if (!b._task) throw new Error("Experiment task is required");
    this.client = b.client;
    this._name = b._name;
    this._description = b._description;
    this.dataset = b._dataset;
    this.task = b._task;
    this.evaluators = new Map(b._evaluators);
    this.config = { ...b._config };
    this.tags = { ...b._tags };
  }

  static builder<I = unknown, O = unknown>(client: ExperimentsClient): ExperimentBuilder<I, O> {
    return new ExperimentBuilder<I, O>(client);
  }

  /** @internal — bridge so ExperimentBuilder.build() can reach the private ctor. */
  static fromBuilder<I, O>(b: ExperimentBuilder<I, O>): Experiment<I, O> {
    return new Experiment<I, O>(b);
  }

  name(): string {
    return this._name;
  }

  experimentId(): string | null {
    return this._experimentId;
  }

  /** Dashboard URL for the experiment. Null until `run()` has created it. */
  url(): string | null {
    if (this._experimentId === null) return null;
    return `${appBase(this.client.site())}/llm/experiments/${this._experimentId}`;
  }

  /** Run the experiment sequentially and return the result. */
  async run(): Promise<ExperimentResult> {
    const projectId = await this.client.ensureProjectId();

    // Ensure the dataset exists remotely and all records are pushed.
    await this.dataset.ensureCreatedAndPushed(projectId);
    const datasetId = this.dataset.id();
    if (datasetId === null) {
      throw new Error(`Dataset '${this.dataset.name()}' has no id after push`);
    }

    // Create the experiment through the generated client. ensureUnique:true makes
    // the backend mint a fresh experiment under the project on every run.
    try {
      const resp = await this.client.api().createLLMObsExperiment({
        body: {
          data: {
            type: "experiments",
            attributes: {
              name: this._name,
              projectId,
              datasetId,
              description: this._description,
              ...(Object.keys(this.config).length > 0 ? { config: this.config } : {}),
              ensureUnique: true,
            },
          },
        },
      });
      this._experimentId = String(resp?.data?.id ?? "");
    } catch (err) {
      throw new Error(`Failed to create experiment '${this._name}': ${(err as Error).message}`);
    }
    const experimentId = this._experimentId as string;

    // Best-effort interrupt handling: if the process is signalled mid-run, mark
    // the experiment interrupted.
    let finished = false;
    const interruptHandler = () => {
      if (!finished) {
        void this.updateStatus("interrupted", "Process terminated before run completed");
      }
    };
    process.once("SIGINT", interruptHandler);
    process.once("SIGTERM", interruptHandler);

    try {
      const records = this.dataset.records();
      const recordIds = this.dataset.recordIds();
      const rows: Row[] = [];
      const spans: Record<string, unknown>[] = [];
      const metrics: Record<string, unknown>[] = [];

      for (let i = 0; i < records.length; i++) {
        const record = records[i];
        const datasetRecordId = i < recordIds.length ? recordIds[i] : "";
        const spanId = hexId();
        const traceId = hexId();
        const startMs = Date.now();
        const startNs = startMs * 1_000_000;
        const startHr = process.hrtime.bigint();

        let output: unknown = null;
        let errorType: string | null = null;
        let errorMessage: string | null = null;

        try {
          output = await this.task(record.input as I, this.config);
        } catch (err) {
          const e = err as Error;
          errorType = e.name || "Error";
          errorMessage = e.message ?? String(err);
        }

        const durationNs = Number(process.hrtime.bigint() - startHr);

        // Run evaluators (only when the task itself did not error).
        const evaluations: Record<string, unknown> = {};
        const evaluationErrors: Record<string, string> = {};
        const timestampMs = Date.now();

        for (const [label, evaluator] of this.evaluators) {
          if (errorType !== null) {
            // Task failed — record an evaluation error so the metric reflects it.
            const msg = "task error; evaluation skipped";
            evaluationErrors[label] = msg;
            metrics.push(
              toMetric(label, null, msg, spanId, timestampMs, experimentId, this.tags),
            );
            continue;
          }
          try {
            const value = await evaluator(record.input, output, record.expectedOutput);
            evaluations[label] = value;
            metrics.push(
              toMetric(label, value, null, spanId, timestampMs, experimentId, this.tags),
            );
          } catch (err) {
            const msg = (err as Error).message ?? String(err);
            evaluationErrors[label] = msg;
            metrics.push(
              toMetric(label, null, msg, spanId, timestampMs, experimentId, this.tags),
            );
          }
        }

        const row = new Row({
          index: i,
          spanId,
          traceId,
          startNs,
          durationNs,
          input: record.input,
          output,
          expectedOutput: record.expectedOutput,
          errorType,
          errorMessage,
          evaluations,
          evaluationErrors,
        });
        rows.push(row);
        spans.push(
          toSpan(
            row,
            record.metadata,
            experimentId,
            projectId,
            datasetId,
            datasetRecordId,
            this._name,
            this.tags,
          ),
        );
      }

      // Post all spans + metrics in a single events call.
      await this.postEvents(experimentId, spans, metrics);
      await this.updateStatus("completed", null);
      finished = true;

      return new ExperimentResult(experimentId, rows, this.url() as string);
    } catch (err) {
      await this.updateStatus("failed", (err as Error).message ?? String(err));
      finished = true;
      throw err;
    } finally {
      process.removeListener("SIGINT", interruptHandler);
      process.removeListener("SIGTERM", interruptHandler);
    }
  }

  private async postEvents(
    experimentId: string,
    spans: Record<string, unknown>[],
    metrics: Record<string, unknown>[],
  ): Promise<void> {
    // W2: the events POST uses type "experiments" (not "events").
    const body = {
      data: {
        type: "experiments",
        attributes: { spans, metrics },
      },
    };
    await http.post(
      this.client.credentials(),
      `/api/v2/llm-obs/v1/experiments/${experimentId}/events`,
      body,
    );
  }

  private async updateStatus(status: string, error: string | null): Promise<void> {
    // Hand-rolled: the generated updateLLMObsExperiment model exposes only
    // name/description, not `status`/`error`, so the lifecycle PATCH bypasses it.
    if (this._experimentId === null) return;
    const attributes: Record<string, unknown> = { status };
    if (error !== null) attributes.error = error;
    try {
      await http.patch(
        this.client.credentials(),
        `/api/v2/llm-obs/v1/experiments/${this._experimentId}`,
        { data: { type: "experiments", attributes } },
      );
    } catch {
      // Status update is best-effort; never let it mask the real result/error.
    }
  }
}

/** Builder for {@link Experiment}. */
export class ExperimentBuilder<I = unknown, O = unknown> {
  /** @internal */ readonly client: ExperimentsClient;
  /** @internal */ _name = "";
  /** @internal */ _description = "";
  /** @internal */ _dataset: Dataset | null = null;
  /** @internal */ _task: Task<I, O> | null = null;
  /** @internal */ _evaluators = new Map<string, Evaluator>();
  /** @internal */ _config: Record<string, unknown> = {};
  /** @internal */ _tags: Record<string, string> = {};

  constructor(client: ExperimentsClient) {
    this.client = client;
  }

  name(name: string): this {
    this._name = name;
    return this;
  }

  description(description: string): this {
    this._description = description;
    return this;
  }

  dataset(dataset: Dataset): this {
    this._dataset = dataset;
    return this;
  }

  task(task: Task<I, O>): this {
    this._task = task;
    return this;
  }

  /** Register an evaluator under a label. Multiple calls allowed. */
  evaluator(label: string, evaluator: Evaluator): this {
    this._evaluators.set(label, evaluator);
    return this;
  }

  config(config: Record<string, unknown>): this {
    this._config = { ...config };
    return this;
  }

  tags(tags: Record<string, string>): this {
    this._tags = { ...tags };
    return this;
  }

  build(): Experiment<I, O> {
    return Experiment.fromBuilder<I, O>(this);
  }
}
