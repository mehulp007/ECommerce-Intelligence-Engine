import { apiKey, apiUrl } from "@/lib/server/config";

export const dynamic = "force-dynamic";

const PASS_HEADERS = ["content-type", "cache-control", "retry-after"];

/**
 * Forwards /api/* to the FastAPI service and adds the bearer key on the server.
 * Responses are streamed through, so copilot Server-Sent Events arrive live.
 */
async function forward(request: Request, segments: string[]): Promise<Response> {
  if (segments.some((s) => s === "." || s === ".." || s === "")) {
    return Response.json({ detail: "Bad path" }, { status: 400 });
  }
  const route = segments.map(encodeURIComponent).join("/");
  const target = new URL(
    route === "health" ? `${apiUrl()}/health` : `${apiUrl()}/api/${route}`,
  );
  target.search = new URL(request.url).search;

  const headers = new Headers({ Accept: request.headers.get("accept") ?? "*/*" });
  const key = apiKey();
  if (key) headers.set("Authorization", `Bearer ${key}`);
  let body: string | undefined;
  if (request.method === "POST") {
    headers.set("Content-Type", "application/json");
    body = await request.text();
  }

  let upstream: Response;
  try {
    upstream = await fetch(target, {
      method: request.method,
      headers,
      body,
      cache: "no-store",
      signal: request.signal,
    });
  } catch {
    return Response.json(
      { detail: `The API is not reachable at ${apiUrl()}. Start it with: eie serve run` },
      { status: 502 },
    );
  }
  const out = new Headers();
  for (const name of PASS_HEADERS) {
    const value = upstream.headers.get(name);
    if (value) out.set(name, value);
  }
  if (out.get("content-type")?.startsWith("text/event-stream")) {
    out.set("X-Accel-Buffering", "no");
  }
  return new Response(upstream.body, { status: upstream.status, headers: out });
}

export async function GET(request: Request, ctx: RouteContext<"/api/[...path]">) {
  return forward(request, (await ctx.params).path);
}

export async function POST(request: Request, ctx: RouteContext<"/api/[...path]">) {
  return forward(request, (await ctx.params).path);
}
