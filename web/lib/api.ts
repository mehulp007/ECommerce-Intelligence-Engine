/** Browser-side API client. Calls go to this app's /api proxy, never to the API directly. */

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
  }
}

type Params = Record<string, string | number | boolean | null | undefined>;

export function withParams(path: string, params?: Params): string {
  const search = new URLSearchParams();
  for (const [key, value] of Object.entries(params ?? {})) {
    if (value !== undefined && value !== null && value !== "") search.set(key, String(value));
  }
  const query = search.toString();
  return query ? `${path}?${query}` : path;
}

export async function errorFrom(response: Response): Promise<ApiError> {
  let message = `${response.status} ${response.statusText}`;
  try {
    const body = await response.json();
    if (typeof body.detail === "string") message = body.detail;
    else if (Array.isArray(body.detail)) {
      message = body.detail.map((d: { msg?: string }) => d.msg ?? "").join("; ");
    }
  } catch {
    // Keep the status line.
  }
  return new ApiError(response.status, message);
}

export async function getJson<T>(path: string, params?: Params, signal?: AbortSignal): Promise<T> {
  const response = await fetch(withParams(path, params), { signal });
  if (!response.ok) throw await errorFrom(response);
  return response.json() as Promise<T>;
}

export async function postJson<T>(path: string, body: unknown, signal?: AbortSignal): Promise<T> {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!response.ok) throw await errorFrom(response);
  return response.json() as Promise<T>;
}

export const imageUrl = (articleId: string) => `/images/${articleId}.webp`;
