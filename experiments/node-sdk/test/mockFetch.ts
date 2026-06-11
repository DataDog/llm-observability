import { client } from "@datadog/datadog-api-client";

/**
 * Test transport that serves BOTH paths the SDK uses:
 *
 *  1. The generated `datadog-api-client` (project/dataset/experiment creation and
 *     lists) — intercepted by injecting `mock.httpLibrary` as the client's
 *     `HttpLibrary`. The generated client uses `cross-fetch`/`node-fetch`, not
 *     global `fetch`, so overriding `globalThis.fetch` alone would miss it.
 *  2. The hand-rolled `fetch` calls (records POST → W1, events POST → W2, status
 *     PATCH) — intercepted by overriding `globalThis.fetch`.
 *
 * Both transports consult the same route table and append to the same
 * `requests` log, so a test can assert on the full wire traffic regardless of
 * which transport produced it.
 */

export interface RecordedRequest {
  method: string;
  url: string;
  path: string;
  headers: Record<string, string>;
  body: any;
  /** Which transport produced this request. */
  via: "fetch" | "generated";
}

export interface MockRoute {
  /** e.g. "POST /api/v2/llm-obs/v1/projects" — matched as a prefix on the path. */
  match: string;
  status?: number;
  /** JSON body to return, or a function of the request. */
  response: unknown | ((req: RecordedRequest) => unknown);
}

export class MockFetch {
  readonly requests: RecordedRequest[] = [];
  private routes: MockRoute[];
  private original: typeof globalThis.fetch | undefined;

  constructor(routes: MockRoute[]) {
    this.routes = routes;
  }

  install(): void {
    this.original = globalThis.fetch;
    globalThis.fetch = this.fetchHandler as unknown as typeof globalThis.fetch;
  }

  restore(): void {
    if (this.original) globalThis.fetch = this.original;
  }

  /** Requests filtered to a method + path-prefix. */
  matching(prefix: string): RecordedRequest[] {
    const [method, p] = prefix.split(" ");
    return this.requests.filter((r) => r.method === method && r.path.startsWith(p));
  }

  /** Resolve a route + payload for a recorded request. */
  private resolve(req: RecordedRequest): { status: number; payload: unknown } {
    const route = this.routes.find((r) => {
      const [m, p] = r.match.split(" ");
      return m === req.method && req.path.startsWith(p);
    });
    if (!route) {
      return { status: 404, payload: { error: `no mock route for ${req.method} ${req.path}` } };
    }
    const payload =
      typeof route.response === "function"
        ? (route.response as (req: RecordedRequest) => unknown)(req)
        : route.response;
    return { status: route.status ?? 200, payload };
  }

  // ---- transport 1: global fetch (hand-rolled calls) ----
  private fetchHandler = async (input: string, init?: RequestInit): Promise<Response> => {
    const url = String(input);
    const req: RecordedRequest = {
      method: (init?.method ?? "GET").toUpperCase(),
      url,
      path: new URL(url).pathname,
      headers: (init?.headers ?? {}) as Record<string, string>,
      body: init?.body ? JSON.parse(init.body as string) : undefined,
      via: "fetch",
    };
    this.requests.push(req);
    const { status, payload } = this.resolve(req);
    return new Response(JSON.stringify(payload), { status });
  };

  // ---- transport 2: generated client HttpLibrary ----
  get httpLibrary(): client.HttpLibrary {
    return {
      send: async (request: client.RequestContext): Promise<client.ResponseContext> => {
        const url = request.getUrl();
        const bodyStr = request.getBody();
        const req: RecordedRequest = {
          method: String(request.getHttpMethod()).toUpperCase(),
          url,
          path: new URL(url).pathname,
          headers: request.getHeaders(),
          body: typeof bodyStr === "string" && bodyStr ? JSON.parse(bodyStr) : undefined,
          via: "generated",
        };
        this.requests.push(req);
        const { status, payload } = this.resolve(req);
        const json = JSON.stringify(payload);
        const responseBody: client.ResponseBody = {
          text: async () => json,
          binary: async () => Buffer.from(json),
        };
        return new client.ResponseContext(status, {}, responseBody);
      },
    };
  }
}

/** Default happy-path routes for the full experiment flow. */
export function defaultRoutes(opts?: {
  projectId?: string;
  datasetId?: string;
  experimentId?: string;
  recordIds?: string[];
}): MockRoute[] {
  const projectId = opts?.projectId ?? "proj-1";
  const datasetId = opts?.datasetId ?? "ds-1";
  const experimentId = opts?.experimentId ?? "exp-1";
  const recordIds = opts?.recordIds ?? ["rec-0", "rec-1", "rec-2", "rec-3"];

  // NOTE: routes are matched in order by (method, path-prefix); more specific
  // prefixes must come first. Generated-client responses are deserialized by the
  // api-client, so they carry type + attributes (not just id).
  return [
    {
      match: "POST /api/v2/llm-obs/v1/projects",
      response: { data: { id: projectId, type: "projects", attributes: { name: "p" } } },
    },
    {
      // dataset records POST (hand-rolled, W1). Echo one id per record sent.
      match: `POST /api/v2/llm-obs/v1/${projectId}/datasets/${datasetId}/records`,
      response: (req) => {
        const sent: unknown[] = req.body?.data?.attributes?.records ?? [];
        return { data: sent.map((_, i) => ({ id: recordIds[i] ?? `rec-${i}` })) };
      },
    },
    {
      match: `POST /api/v2/llm-obs/v1/${projectId}/datasets`,
      response: {
        data: { id: datasetId, type: "datasets", attributes: { name: "d", description: "" } },
      },
    },
    {
      // experiment events POST (hand-rolled, W2; path ends in /events)
      match: "POST /api/v2/llm-obs/v1/experiments/",
      response: { data: { id: experimentId } },
    },
    {
      // experiment create POST (generated; exact, no trailing slash)
      match: "POST /api/v2/llm-obs/v1/experiments",
      response: {
        data: { id: experimentId, type: "experiments", attributes: { name: "e" } },
      },
    },
    {
      // experiment status PATCH (hand-rolled)
      match: "PATCH /api/v2/llm-obs/v1/experiments/",
      response: { data: { id: experimentId } },
    },
  ];
}
