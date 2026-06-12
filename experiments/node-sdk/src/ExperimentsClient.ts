import type { v2 } from "@datadog/datadog-api-client";
import { Dataset } from "./Dataset.js";
import { DatasetRecord } from "./DatasetRecord.js";
import { retryWithBackoff } from "./internal/backoff.js";
import { buildLlmObsApi, type HttpLibrary } from "./internal/generatedClient.js";
import type { ApiCredentials } from "./internal/http.js";

/** Options for {@link ExperimentsClient.pullDataset}. */
export interface PullDatasetOptions {
  /**
   * Wait (with exponential backoff) until at least this many records are
   * readable from the backend. Use when pulling right after a push, to absorb
   * read-after-write lag. If omitted, the pull only waits for the dataset itself
   * to become visible.
   */
  expectedRecordCount?: number;
  /** Maximum total time to wait for the dataset/records to appear. Default 30s. */
  maxWaitMs?: number;
}

export interface ExperimentsClientOptions {
  apiKey: string;
  applicationKey: string;
  /** Bare site host. Defaults to "datadoghq.com". */
  site?: string;
  projectName: string;
  /**
   * @internal — inject a custom `HttpLibrary` into the generated client. Used by
   * the test suite to mock transport; not part of the public contract.
   */
  httpApi?: HttpLibrary;
}

/**
 * Auth + site + project; the factory for everything else.
 *
 * Object-argument constructor, mirroring the Java `ExperimentsClient.builder()`
 * surface. A static `builder()` is also provided so code reads the same as the
 * Java/other-language examples.
 */
export class ExperimentsClient {
  private readonly creds: ApiCredentials;
  private readonly _projectName: string;
  private readonly _api: v2.LLMObservabilityApi;
  private cachedProjectId: string | null = null;

  constructor(options: ExperimentsClientOptions) {
    if (!options.apiKey) throw new Error("apiKey is required");
    if (!options.applicationKey) throw new Error("applicationKey is required");
    if (!options.projectName) throw new Error("projectName is required");
    this.creds = {
      apiKey: options.apiKey,
      appKey: options.applicationKey,
      site: options.site ?? "datadoghq.com",
    };
    this._projectName = options.projectName;
    this._api = buildLlmObsApi(
      this.creds.apiKey,
      this.creds.appKey,
      this.creds.site,
      options.httpApi,
    );
  }

  /** Fluent builder, parity with the Java `ExperimentsClient.builder()`. */
  static builder(): ExperimentsClientBuilder {
    return new ExperimentsClientBuilder();
  }

  projectName(): string {
    return this._projectName;
  }

  site(): string {
    return this.creds.site;
  }

  /** @internal — used by Dataset/Experiment for the hand-rolled HTTP calls (W1/W2/status). */
  apiKey(): string {
    return this.creds.apiKey;
  }

  /** @internal */
  applicationKey(): string {
    return this.creds.appKey;
  }

  /** @internal — credentials bundle for the internal HTTP helpers. */
  credentials(): ApiCredentials {
    return this.creds;
  }

  /**
   * Escape hatch: the underlying generated `LLMObservabilityApi`.
   *
   * Parity with the Java SDK's `api()`. Use it to reach LLM Obs endpoints the six
   * public types do not model (annotation queues, custom evals, deletes, etc.).
   */
  api(): v2.LLMObservabilityApi {
    return this._api;
  }

  /** Create a local dataset buffer. Pushed remotely on first `experiment.run()`. */
  createDataset(name: string, description = ""): Dataset {
    return new Dataset(this, name, description);
  }

  /**
   * Resolve the project id for `projectName`, creating the project if it does not
   * already exist. The create endpoint is get-or-create on name, so a repeated
   * call simply returns the existing id. Cached after the first resolution.
   */
  async ensureProjectId(): Promise<string> {
    if (this.cachedProjectId !== null) return this.cachedProjectId;
    try {
      const resp = await this._api.createLLMObsProject({
        body: { data: { type: "projects", attributes: { name: this._projectName } } },
      });
      this.cachedProjectId = String(resp?.data?.id ?? "");
      return this.cachedProjectId;
    } catch (err) {
      throw new Error(
        `Failed to create or get project '${this._projectName}': ${(err as Error).message}`,
      );
    }
  }

  /**
   * Pull an existing dataset by name (with its records) from the current project.
   *
   * Reads on the LLM Obs API are eventually consistent, so a pull issued right
   * after a push can momentarily see the dataset missing or only some of its
   * records. This polls with exponential backoff (up to `maxWaitMs`, default 30s)
   * until the dataset is visible and — when `expectedRecordCount` is given — at
   * least that many records are readable, confirming the push fully landed.
   */
  async pullDataset(name: string, options: PullDatasetOptions = {}): Promise<Dataset> {
    const projectId = await this.ensureProjectId();
    const { expectedRecordCount, maxWaitMs = 30_000 } = options;

    let datasetId: string | null = null;
    let description = "";
    let records: DatasetRecord[] = [];
    let recordIds: string[] = [];
    let lastError = "";

    const succeeded = await retryWithBackoff(async () => {
      try {
        // 1. Find the dataset by name.
        if (datasetId === null) {
          const listed = await this._api.listLLMObsDatasets({ projectId, filterName: name });
          for (const item of listed?.data ?? []) {
            if (item?.attributes?.name === name) {
              datasetId = String(item?.id ?? "");
              description = String(item?.attributes?.description ?? "");
              break;
            }
          }
          if (datasetId === null) return false; // not visible yet — keep waiting
        }

        // 2. Fetch its records via the generated client.
        //
        // Spec drift: the API returns record content nested under
        // data[].attributes (JSON:API), but the generated
        // LLMObsDatasetRecordDataResponse model is flat (input/expected_output/
        // metadata at the top level). The deserializer therefore can't map the
        // nested object onto the typed fields (they come back undefined) and
        // instead stashes the whole `attributes` object under
        // `additionalProperties`. Read from there, falling back to the typed
        // fields in case the API/model are realigned later.
        const recs: DatasetRecord[] = [];
        const ids: string[] = [];
        const resp = await this._api.listLLMObsDatasetRecords({ projectId, datasetId });
        for (const item of resp?.data ?? []) {
          const nested = (item as any)?.additionalProperties?.attributes;
          const src = nested ?? item;
          recs.push(
            new DatasetRecord(
              unwrapAnyValue(src?.input) ?? null,
              unwrapAnyValue(src?.expected_output ?? src?.expectedOutput) ?? null,
              (unwrapAnyValue(src?.metadata) as Record<string, unknown>) ?? {},
            ),
          );
          ids.push(String(item?.id ?? ""));
        }
        records = recs;
        recordIds = ids;

        // 3. If a count was expected, keep waiting until everything has landed.
        if (expectedRecordCount != null && recs.length < expectedRecordCount) {
          return false;
        }
        return true;
      } catch (err) {
        lastError = (err as Error).message;
        return false; // transient read error — retry within the budget
      }
    }, { maxTotalMs: maxWaitMs });

    if (datasetId === null) {
      if (lastError) {
        throw new Error(`Failed to list datasets in project '${this._projectName}': ${lastError}`);
      }
      throw new Error(`Dataset '${name}' not found in project '${this._projectName}' (after ${maxWaitMs}ms)`);
    }
    if (!succeeded && expectedRecordCount != null) {
      throw new Error(
        `Dataset '${name}' has ${records.length} record(s) after ${maxWaitMs}ms, expected ${expectedRecordCount} — backend may not have finished ingesting the push`,
      );
    }

    return Dataset.fromExisting(this, name, description, datasetId, projectId, records, recordIds);
  }
}

/**
 * Record `input` / `expectedOutput` are typed `AnyValue` in the generated model;
 * the deserializer wraps some scalar values in an `UnparsedObject` of the form
 * `{ _data: <raw> }`. Unwrap it back to the raw JSON value. (Values read out of
 * `additionalProperties` are already raw and pass through unchanged.)
 */
function unwrapAnyValue(value: unknown): unknown {
  if (value && typeof value === "object" && "_data" in (value as Record<string, unknown>)) {
    return (value as { _data: unknown })._data;
  }
  return value;
}

/** Fluent builder mirroring the Java `ExperimentsClient.Builder`. */
export class ExperimentsClientBuilder {
  private _apiKey = "";
  private _appKey = "";
  private _site = "datadoghq.com";
  private _projectName = "";

  apiKey(apiKey: string): this {
    this._apiKey = apiKey;
    return this;
  }

  applicationKey(appKey: string): this {
    this._appKey = appKey;
    return this;
  }

  site(site: string): this {
    this._site = site;
    return this;
  }

  projectName(projectName: string): this {
    this._projectName = projectName;
    return this;
  }

  build(): ExperimentsClient {
    return new ExperimentsClient({
      apiKey: this._apiKey,
      applicationKey: this._appKey,
      site: this._site,
      projectName: this._projectName,
    });
  }
}
