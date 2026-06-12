import { client, v2 } from "@datadog/datadog-api-client";

/**
 * W5: the generated `ObjectSerializer.deserialize` throws "missing required
 * property '<x>'" whenever a response omits a field the spec marks `required:
 * true` (e.g. `created_at`, `description`). Several LLM Obs response models hit
 * this. The spec's prescribed fix is to "configure the deserializer to ignore the
 * required flag on response classes" — here we do exactly that, once, by clearing
 * the `required` flag on every v2 model's attribute map. Relaxing it on request
 * models too is harmless: the SDK always supplies the fields it sends.
 */
let relaxed = false;
function relaxRequiredFlags(): void {
  if (relaxed) return;
  relaxed = true;
  const ns = v2 as unknown as Record<string, unknown>;
  for (const key of Object.keys(ns)) {
    const model = ns[key] as { getAttributeTypeMap?: () => Record<string, { required?: boolean }> };
    if (typeof model?.getAttributeTypeMap !== "function") continue;
    try {
      const map = model.getAttributeTypeMap();
      for (const attr of Object.keys(map)) {
        if (map[attr]?.required) map[attr].required = false;
      }
    } catch {
      // best-effort; skip any model whose map can't be read/mutated
    }
  }
}

/**
 * Builds the generated `datadog-api-client` LLM Observability API, configured for
 * the SDK's needs:
 *
 *  - W7: construct a fresh `Configuration` via `createConfiguration` rather than
 *    the env-driven default, so `DD_SITE` is never read eagerly.
 *  - W6: point at an explicit `baseServer` (`https://api.<site>`) instead of the
 *    restricted, enum-validated default server — this makes every site work,
 *    including staging (`datad0g.com`) and GovCloud, with no allow-list errors.
 *  - Enable the unstable LLM Obs operations the SDK calls.
 *
 * The LLM Obs endpoints used through the generated client are:
 *   createLLMObsProject, createLLMObsDataset, listLLMObsDatasets,
 *   listLLMObsDatasetRecords, createLLMObsExperiment.
 *
 * Only the three endpoints the generated client genuinely cannot perform are
 * hand-rolled in `http.ts`: records POST (W1 — wrong `type` discriminator),
 * events POST (W2 — wrong `type` discriminator), and the experiment status PATCH
 * (the update model has no `status` field). The records LIST read still goes
 * through the generated client; its response just needs a small extraction shim
 * because
 * the flat record model leaves the nested `attributes` under
 * `additionalProperties` (see ExperimentsClient.pullDataset).
 */

/** Operations the SDK invokes that are flagged "unstable" and must be enabled. */
const UNSTABLE_OPERATIONS = [
  "v2.createLLMObsProject",
  "v2.createLLMObsDataset",
  "v2.listLLMObsDatasets",
  "v2.listLLMObsDatasetRecords",
  "v2.createLLMObsExperiment",
];

/** Anything implementing the generated client's `HttpLibrary` (for test injection). */
export type HttpLibrary = client.HttpLibrary;

export function buildLlmObsApi(
  apiKey: string,
  appKey: string,
  site: string,
  httpApi?: HttpLibrary,
): v2.LLMObservabilityApi {
  relaxRequiredFlags(); // W5
  const configuration = client.createConfiguration({
    authMethods: { apiKeyAuth: apiKey, appKeyAuth: appKey },
    // W6/W7: explicit, unrestricted base server for the chosen site.
    baseServer: new client.BaseServerConfiguration(`https://api.${site}`, {}),
    ...(httpApi ? { httpApi } : {}),
  });

  for (const op of UNSTABLE_OPERATIONS) {
    configuration.unstableOperations[op] = true;
  }

  return new v2.LLMObservabilityApi(configuration);
}
