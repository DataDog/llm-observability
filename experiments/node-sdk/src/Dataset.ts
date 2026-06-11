import { DatasetRecord } from "./DatasetRecord.js";
import type { ExperimentsClient } from "./ExperimentsClient.js";
import * as http from "./internal/http.js";
import { appBase } from "./internal/http.js";

/**
 * A local buffer of dataset records, auto-created and pushed remotely on the
 * first `experiment.run()` (or eagerly via `push()`). Mirrors the Java `Dataset`.
 *
 * `addRecord` is fluent. `id()`/`projectId()` are null until the dataset has been
 * pushed.
 */
export class Dataset {
  private readonly client: ExperimentsClient;
  private readonly _name: string;
  private readonly _description: string;
  private readonly _records: DatasetRecord[] = [];
  private readonly _recordIds: string[] = [];
  private _id: string | null = null;
  private _projectId: string | null = null;
  private pushedCount = 0;

  constructor(client: ExperimentsClient, name: string, description = "") {
    this.client = client;
    this._name = name;
    this._description = description;
  }

  /** @internal — build a Dataset that already exists remotely (used by pullDataset). */
  static fromExisting(
    client: ExperimentsClient,
    name: string,
    description: string,
    id: string,
    projectId: string,
    records: DatasetRecord[],
    recordIds: string[],
  ): Dataset {
    const ds = new Dataset(client, name, description);
    ds._id = id;
    ds._projectId = projectId;
    ds._records.push(...records);
    ds._recordIds.push(...recordIds);
    ds.pushedCount = records.length;
    return ds;
  }

  /** Append a record. Accepts either a DatasetRecord or `(input, expectedOutput?, metadata?)`. */
  addRecord(record: DatasetRecord): Dataset;
  addRecord(
    input: unknown,
    expectedOutput?: unknown,
    metadata?: Record<string, unknown>,
  ): Dataset;
  addRecord(
    recordOrInput: DatasetRecord | unknown,
    expectedOutput?: unknown,
    metadata?: Record<string, unknown>,
  ): Dataset {
    const record =
      recordOrInput instanceof DatasetRecord
        ? recordOrInput
        : new DatasetRecord(recordOrInput, expectedOutput, metadata);
    this._records.push(record);
    return this;
  }

  name(): string {
    return this._name;
  }

  records(): readonly DatasetRecord[] {
    return [...this._records];
  }

  /** @internal — record ids aligned by index with `records()` (after push). */
  recordIds(): readonly string[] {
    return [...this._recordIds];
  }

  /** Dataset id, or null until pushed. */
  id(): string | null {
    return this._id;
  }

  /** Project id, or null until pushed. */
  projectId(): string | null {
    return this._projectId;
  }

  /** Dashboard URL for this dataset, or null until it has been pushed/pulled. */
  url(): string | null {
    if (this._id === null) return null;
    return `${appBase(this.client.site())}/llm/datasets/${this._id}`;
  }

  /** Eagerly create the dataset (if needed) and push any unpushed records. */
  async push(): Promise<void> {
    const projectId = await this.client.ensureProjectId();
    await this.ensureCreatedAndPushed(projectId);
  }

  /**
   * @internal — create the remote dataset if it does not yet exist, then push any
   * records added since the last push. Idempotent and incremental: tracks
   * `pushedCount` so repeated calls only send new records.
   */
  async ensureCreatedAndPushed(projectId: string): Promise<void> {
    const creds = this.client.credentials();

    if (this._id === null) {
      // Dataset creation goes through the generated client (no active workaround).
      try {
        const resp = await this.client.api().createLLMObsDataset({
          projectId,
          body: {
            data: {
              type: "datasets",
              attributes: { name: this._name, description: this._description },
            },
          },
        });
        this._id = String(resp?.data?.id ?? "");
        this._projectId = projectId;
      } catch (err) {
        throw new Error(`Failed to create dataset '${this._name}': ${(err as Error).message}`);
      }
    }

    if (this.pushedCount >= this._records.length) {
      return;
    }

    const pending = this._records.slice(this.pushedCount);
    const recordPayload = pending.map((rec) => {
      const r: Record<string, unknown> = { input: rec.input };
      if (rec.expectedOutput !== null && rec.expectedOutput !== undefined) {
        r.expected_output = rec.expectedOutput;
      }
      if (rec.metadata && Object.keys(rec.metadata).length > 0) {
        r.metadata = rec.metadata;
      }
      return r;
    });

    // W1: the records POST must send type "datasets"; the generated client's
    // model still serializes "records", which the API rejects. Hand-roll it.
    const body = {
      data: {
        type: "datasets",
        attributes: { records: recordPayload },
      },
    };

    let raw: string;
    try {
      raw = await http.postReturning(
        creds,
        `/api/v2/llm-obs/v1/${projectId}/datasets/${this._id}/records`,
        body,
      );
    } catch (err) {
      throw new Error(`Failed to push records to dataset '${this._name}': ${(err as Error).message}`);
    }

    // Capture returned record ids in order; pad with "" if the response omits them.
    const parsed = JSON.parse(raw);
    const data = parsed?.data;
    if (Array.isArray(data)) {
      for (const node of data) {
        this._recordIds.push(String((node as Record<string, any>)?.id ?? ""));
      }
    } else {
      for (let i = 0; i < pending.length; i++) {
        this._recordIds.push("");
      }
    }

    this.pushedCount = this._records.length;
  }
}
