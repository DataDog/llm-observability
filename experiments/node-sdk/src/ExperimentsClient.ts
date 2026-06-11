import type { v2 } from "@datadog/datadog-api-client";
import { Dataset } from "./Dataset.js";
import { DatasetRecord } from "./DatasetRecord.js";
import { buildLlmObsApi, type HttpLibrary } from "./internal/generatedClient.js";
import type { ApiCredentials } from "./internal/http.js";

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
   * Retries the list a few times to absorb read-after-write lag, mirroring the
   * Java `pullDataset`.
   */
  async pullDataset(name: string): Promise<Dataset> {
    const projectId = await this.ensureProjectId();

    let datasetId: string | null = null;
    let description = "";
    let lastError = "";
    for (let attempt = 0; attempt < 3 && datasetId === null; attempt++) {
      try {
        const resp = await this._api.listLLMObsDatasets({ projectId, filterName: name });
        for (const item of resp?.data ?? []) {
          if (item?.attributes?.name === name) {
            datasetId = String(item?.id ?? "");
            description = String(item?.attributes?.description ?? "");
            break;
          }
        }
      } catch (err) {
        lastError = (err as Error).message;
      }
    }

    if (datasetId === null) {
      if (lastError) {
        throw new Error(`Failed to list datasets in project '${this._projectName}': ${lastError}`);
      }
      throw new Error(`Dataset '${name}' not found in project '${this._projectName}' (after retries)`);
    }

    // Fetch the records.
    const records: DatasetRecord[] = [];
    const recordIds: string[] = [];
    try {
      const resp = await this._api.listLLMObsDatasetRecords({ projectId, datasetId });
      for (const item of resp?.data ?? []) {
        records.push(
          new DatasetRecord(
            unwrapAnyValue((item as any)?.input) ?? null,
            unwrapAnyValue((item as any)?.expectedOutput) ?? null,
            (unwrapAnyValue((item as any)?.metadata) as Record<string, unknown>) ?? {},
          ),
        );
        recordIds.push(String(item?.id ?? ""));
      }
    } catch (err) {
      throw new Error(`Failed to parse records for dataset '${name}': ${(err as Error).message}`);
    }

    return Dataset.fromExisting(this, name, description, datasetId, projectId, records, recordIds);
  }
}

/**
 * The generated client deserializes `AnyValue` fields (record input / expected
 * output / metadata) into an `UnparsedObject` wrapper of the form
 * `{ _data: <raw> }`. Unwrap it back to the raw JSON value.
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
