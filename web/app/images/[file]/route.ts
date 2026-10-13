import { apiUrl } from "@/lib/server/config";

export const dynamic = "force-dynamic";

/** Product thumbnails, served by the API from the local image cache. */
export async function GET(_request: Request, ctx: RouteContext<"/images/[file]">) {
  const { file } = await ctx.params;
  if (!/^\d{10}\.webp$/.test(file)) {
    return new Response("Not found", { status: 404 });
  }
  try {
    const upstream = await fetch(`${apiUrl()}/images/${file}`, { cache: "no-store" });
    if (!upstream.ok) return new Response("Not found", { status: 404 });
    return new Response(upstream.body, {
      headers: {
        "Content-Type": "image/webp",
        "Cache-Control": "public, max-age=86400, immutable",
      },
    });
  } catch {
    return new Response("API unavailable", { status: 502 });
  }
}
