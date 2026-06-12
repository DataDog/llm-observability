/**
 * Hand-rolled HTTP transport for the Datadog LLM Observability API.
 *
 * Used only for the endpoints with active spec-drift workarounds — the records
 * POST (W1), the experiment events POST (W2), and the experiment status PATCH
 * (the generated update model has no `status` field). Everything else goes
 * through the generated `@datadog/datadog-api-client`.
 *
 * Because we build the URL ourselves (`https://api.<site><path>`), the staging
 * site `datad0g.com` and the GovCloud sites just work here too.
 */

export interface ApiCredentials {
  apiKey: string;
  appKey: string;
  /** Bare site host, e.g. "datadoghq.com", "us5.datadoghq.com", "datad0g.com". */
  site: string;
}

/** Base URL of the API for a given site: `https://api.<site>`. */
export function apiBase(site: string): string {
  return `https://api.${site}`;
}

/** Base URL of the web app for a given site: `https://app.<site>`. */
export function appBase(site: string): string {
  return `https://app.${site}`;
}

async function send(
  method: "GET" | "POST" | "PATCH",
  creds: ApiCredentials,
  path: string,
  body?: unknown,
): Promise<string> {
  const url = apiBase(creds.site) + path;

  let payload: string | undefined;
  if (body !== undefined) {
    try {
      payload = JSON.stringify(body);
    } catch (err) {
      throw new Error(
        `Failed to serialize request body for ${method} ${path}: ${(err as Error).message}`,
      );
    }
  }

  const headers: Record<string, string> = {
    "DD-API-KEY": creds.apiKey,
    "DD-APPLICATION-KEY": creds.appKey,
  };
  if (payload !== undefined) {
    headers["Content-Type"] = "application/json";
  }

  let res: Response;
  try {
    res = await fetch(url, { method, headers, body: payload });
  } catch (err) {
    throw new Error(`${method} ${path} failed: ${(err as Error).message}`);
  }

  const text = await res.text();
  if (!res.ok) {
    throw new Error(`${method} ${path} failed: HTTP ${res.status} ${text}`);
  }
  return text;
}

/** POST a JSON body; discards the response. */
export async function post(creds: ApiCredentials, path: string, body: unknown): Promise<void> {
  await send("POST", creds, path, body);
}

/** PATCH a JSON body; discards the response. */
export async function patch(creds: ApiCredentials, path: string, body: unknown): Promise<void> {
  await send("PATCH", creds, path, body);
}

/** GET; returns the raw response body as a string. */
export async function get(creds: ApiCredentials, path: string): Promise<string> {
  return send("GET", creds, path);
}

/** POST a JSON body; returns the raw response body as a string. */
export async function postReturning(
  creds: ApiCredentials,
  path: string,
  body: unknown,
): Promise<string> {
  return send("POST", creds, path, body);
}
