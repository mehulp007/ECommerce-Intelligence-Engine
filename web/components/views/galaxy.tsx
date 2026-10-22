"use client";

import { useQuery } from "@tanstack/react-query";
import { Loader2, LocateFixed, Search, X } from "lucide-react";
import dynamic from "next/dynamic";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useMemo, useState, useSyncExternalStore } from "react";

import type { Focus } from "@/components/galaxy/galaxy-scene";
import { ProductImage } from "@/components/product";
import { buttonClasses, ErrorState, Segmented } from "@/components/ui";
import { ApiError, getJson } from "@/lib/api";
import { compact, int, units } from "@/lib/format";
import {
  articleId,
  categoryColour,
  garmentColour,
  loadGalaxy,
  OTHER_COLOUR,
  pointColours,
  type ColourMode,
  type Galaxy,
} from "@/lib/galaxy";
import type { ArticleDetail, SearchHit } from "@/lib/types";
import { cn } from "@/lib/utils";

const GalaxyScene = dynamic(() => import("@/components/galaxy/galaxy-scene"), { ssr: false });

const NAMED_GROUPS = 9;
const NAMED_COLOURS = 14;
const MODES: { value: ColourMode; label: string }[] = [
  { value: "category", label: "Category" },
  { value: "colour", label: "Colour" },
  { value: "price", label: "Price" },
  { value: "sales", label: "Sales" },
];

function subscribeMotion(listener: () => void) {
  const media = matchMedia("(prefers-reduced-motion: reduce)");
  media.addEventListener("change", listener);
  return () => media.removeEventListener("change", listener);
}
const reducedMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

let webgl: boolean | undefined;
function hasWebGL(): boolean {
  if (webgl === undefined) {
    try {
      webgl = Boolean(document.createElement("canvas").getContext("webgl2"));
    } catch {
      webgl = false;
    }
  }
  return webgl;
}

const noSubscribe = () => () => {};
// Phones get every third product, which keeps the map smooth.
const pointStride = () => (window.innerWidth < 768 ? 3 : 1);

/** The `k` products nearest to point `i` on the map, including itself. */
function neighbours(galaxy: Galaxy, i: number, k: number): number[] {
  const [x, y, z] = galaxy.positions.subarray(i * 3, i * 3 + 3);
  const distances: [number, number][] = [];
  for (let j = 0; j < galaxy.count; j++) {
    const dx = galaxy.positions[j * 3] - x;
    const dy = galaxy.positions[j * 3 + 1] - y;
    const dz = galaxy.positions[j * 3 + 2] - z;
    distances.push([dx * dx + dy * dy + dz * dz, j]);
  }
  distances.sort((a, b) => a[0] - b[0]);
  return distances.slice(0, k).map(([, j]) => j);
}

/** Centre and spread of a set of points, for the camera to fly to. */
function focusOn(galaxy: Galaxy, indices: number[]): Focus | null {
  if (!indices.length) return null;
  const centre: [number, number, number] = [0, 0, 0];
  for (const i of indices) for (let k = 0; k < 3; k++) centre[k] += galaxy.positions[i * 3 + k] / indices.length;
  let spread = 0;
  for (const i of indices) {
    const dx = galaxy.positions[i * 3] - centre[0];
    const dy = galaxy.positions[i * 3 + 1] - centre[1];
    const dz = galaxy.positions[i * 3 + 2] - centre[2];
    spread += Math.sqrt(dx * dx + dy * dy + dz * dz) / indices.length;
  }
  return { center: centre, radius: Math.max(0.12, spread) };
}

function counts(codes: Uint8Array): Map<number, number> {
  const out = new Map<number, number>();
  for (const code of codes) out.set(code, (out.get(code) ?? 0) + 1);
  return out;
}

function HoverCard({ galaxy, index, x, y }: { galaxy: Galaxy; index: number; x: number; y: number }) {
  const id = articleId(galaxy.ids[index]);
  const detail = useQuery({
    queryKey: ["product", id],
    queryFn: () => getJson<ArticleDetail>(`/api/products/${id}`),
  });
  const width = 220;
  const left = Math.min(x + 18, window.innerWidth - width - 12);
  const top = Math.min(y + 18, window.innerHeight - 330);
  return (
    <div
      className="pointer-events-none fixed z-50 w-[220px] rounded-2xl border border-border-strong bg-surface-2/95 p-2 shadow-[0_24px_60px_-20px_rgba(0,0,0,0.9)] backdrop-blur"
      style={{ left, top }}
    >
      <ProductImage
        article={{ article_id: id, has_image: detail.data?.has_image ?? true, prod_name: detail.data?.prod_name ?? null }}
        priority
      />
      <div className="px-1 pb-1 pt-2">
        <div className="truncate text-sm font-medium">{detail.data?.prod_name ?? "Loading…"}</div>
        <div className="truncate text-xs text-muted">
          {galaxy.meta.colours[galaxy.colour[index]]} · {galaxy.meta.product_groups[galaxy.group[index]]}
        </div>
        <div className="mt-1.5 flex justify-between font-mono text-[11px] text-muted">
          <span>{compact(galaxy.sales[index])} sold · 4 wk</span>
          {detail.data?.median_price ? <span>{units(detail.data.median_price)} units</span> : null}
        </div>
      </div>
    </div>
  );
}

function Legend({
  galaxy,
  mode,
  hidden,
  onToggle,
}: {
  galaxy: Galaxy;
  mode: ColourMode;
  hidden: Set<number>;
  onToggle: (code: number) => void;
}) {
  const groupCounts = useMemo(() => counts(galaxy.group), [galaxy]);
  const colourCounts = useMemo(() => counts(galaxy.colour), [galaxy]);

  if (mode === "price" || mode === "sales") {
    const gradient =
      mode === "sales"
        ? "linear-gradient(90deg, #333338, #3ddc97, #e6fff2)"
        : "linear-gradient(90deg, #33333d, #9b8cff, #f2eeff)";
    return (
      <div className="w-56">
        <div className="mb-1.5 text-xs font-medium">{mode === "sales" ? "Items sold, last 4 weeks" : "Median price"}</div>
        <div className="h-2 rounded-full" style={{ background: gradient }} />
        <div className="mt-1 flex justify-between text-[11px] text-muted">
          <span>{mode === "sales" ? "None" : "Cheapest"}</span>
          <span>{mode === "sales" ? `${int(galaxy.meta.sales_4w_max)} (log scale)` : "Most expensive"}</span>
        </div>
      </div>
    );
  }

  const isCategory = mode === "category";
  const names = isCategory ? galaxy.meta.product_groups : galaxy.meta.colours;
  const limit = isCategory ? NAMED_GROUPS : NAMED_COLOURS;
  const tally = isCategory ? groupCounts : colourCounts;
  const shown = names.slice(0, limit).map((name, code) => ({ name, code }));
  const otherCodes = names.map((_, code) => code).slice(limit);
  const otherCount = otherCodes.reduce((sum, code) => sum + (tally.get(code) ?? 0), 0);

  return (
    <div className="flex flex-col gap-0.5">
      <div className="mb-1 text-xs font-medium">{isCategory ? "Product groups" : "Garment colours"} · click to hide</div>
      <div className={cn("grid gap-x-3 gap-y-0.5", isCategory ? "grid-cols-1" : "grid-cols-2")}>
        {shown.map(({ name, code }) => (
          <button
            key={code}
            onClick={() => onToggle(code)}
            aria-pressed={!hidden.has(code)}
            className={cn(
              "flex items-center gap-2 rounded-md px-1.5 py-0.5 text-left text-xs transition-opacity hover:bg-white/5",
              hidden.has(code) && "opacity-35",
            )}
          >
            <span
              className="size-2.5 shrink-0 rounded-full ring-1 ring-white/10"
              style={{ background: isCategory ? categoryColour(code) : garmentColour(name) }}
            />
            <span className="truncate">{name}</span>
            <span className="ml-auto pl-2 font-mono text-[10px] text-muted">{compact(tally.get(code) ?? 0)}</span>
          </button>
        ))}
        {otherCount > 0 && (
          <div className="flex items-center gap-2 px-1.5 py-0.5 text-xs text-muted">
            <span className="size-2.5 shrink-0 rounded-full" style={{ background: isCategory ? OTHER_COLOUR : "#8c8c8c" }} />
            Other
            <span className="ml-auto pl-2 font-mono text-[10px]">{compact(otherCount)}</span>
          </div>
        )}
      </div>
    </div>
  );
}

export function GalaxyView() {
  const router = useRouter();
  const params = useSearchParams();
  const focusProduct = params.get("product");
  const supported = useSyncExternalStore(noSubscribe, hasWebGL, () => null);
  const stride = useSyncExternalStore(noSubscribe, pointStride, () => 1);
  const [mode, setMode] = useState<ColourMode>("category");
  const [hiddenGroups, setHiddenGroups] = useState<Set<number>>(new Set());
  const [hiddenColours, setHiddenColours] = useState<Set<number>>(new Set());
  const [text, setText] = useState("");
  const [query, setQuery] = useState("");
  const [hover, setHover] = useState<{ index: number; x: number; y: number } | null>(null);
  const [press, setPress] = useState<{ x: number; y: number } | null>(null);
  const calm = useSyncExternalStore(subscribeMotion, reducedMotion, () => false);

  const galaxy = useQuery({
    queryKey: ["galaxy", stride],
    queryFn: () => loadGalaxy(stride),
    enabled: supported === true,
    staleTime: Infinity,
  });
  const search = useQuery({
    queryKey: ["galaxy-search", query],
    queryFn: () => getJson<SearchHit[]>("/api/search", { q: query, k: 50 }),
    enabled: query.length >= 2,
  });

  const g = galaxy.data;
  const colours = useMemo(() => (g ? pointColours(g, mode) : new Float32Array()), [g, mode]);

  const matches = useMemo(() => {
    if (!g) return null;
    if (focusProduct && !query) {
      const i = g.index.get(focusProduct);
      return i === undefined ? [] : neighbours(g, i, 40);
    }
    if (!query || !search.data) return null;
    return search.data.map((hit) => g.index.get(hit.article.article_id)).filter((i): i is number => i !== undefined);
  }, [g, query, search.data, focusProduct]);

  const highlight = useMemo(() => {
    if (!g) return null;
    const filtering = hiddenGroups.size > 0 || hiddenColours.size > 0;
    if (!filtering && !matches) return null;
    const out = new Float32Array(g.count);
    const matched = matches ? new Set(matches) : null;
    for (let i = 0; i < g.count; i++) {
      const visible = !hiddenGroups.has(g.group[i]) && !hiddenColours.has(g.colour[i]);
      out[i] = visible && (!matched || matched.has(i)) ? 1 : 0;
    }
    return out;
  }, [g, hiddenGroups, hiddenColours, matches]);

  const focus = useMemo(() => (g && matches?.length ? focusOn(g, matches) : null), [g, matches]);
  const marker = useMemo((): [number, number, number] | null => {
    const i = g && focusProduct && !query ? g.index.get(focusProduct) : undefined;
    if (!g || i === undefined) return null;
    return [g.positions[i * 3], g.positions[i * 3 + 1], g.positions[i * 3 + 2]];
  }, [g, focusProduct, query]);

  const toggle = (set: Set<number>, update: (next: Set<number>) => void) => (code: number) => {
    const next = new Set(set);
    if (next.has(code)) next.delete(code);
    else next.add(code);
    update(next);
  };

  const clearFocus = () => {
    setQuery("");
    setText("");
    if (focusProduct) router.replace("/galaxy", { scroll: false });
  };

  if (supported === false) {
    return (
      <div className="flex flex-col items-start gap-3 rounded-2xl border border-border bg-surface p-6 text-sm">
        <div className="font-medium">This browser can&apos;t show 3D graphics (WebGL is off).</div>
        <Link href="/products" className={buttonClasses("secondary", "sm")}>
          Browse products instead
        </Link>
      </div>
    );
  }
  if (galaxy.error) {
    const notBuilt = galaxy.error instanceof ApiError && galaxy.error.status === 503;
    return notBuilt ? (
      <div className="rounded-2xl border border-border bg-surface p-6 text-sm text-muted">
        The product map hasn&apos;t been built yet. Run <code className="font-mono text-fg">eie search galaxy</code>,
        then reload this page.
      </div>
    ) : (
      <ErrorState error={galaxy.error} onRetry={() => galaxy.refetch()} />
    );
  }

  const status = search.isFetching
    ? "Searching…"
    : matches && query
      ? `${matches.length} matches for “${query}”`
      : matches && focusProduct
        ? matches.length
          ? "This product (ringed) and its 39 nearest neighbours"
          : "That product isn't on the map"
        : null;

  return (
    <div className="relative h-[calc(100dvh-3.6rem)] overflow-hidden lg:h-dvh">
      <div
        className="absolute inset-0"
        style={{ background: "radial-gradient(ellipse at 50% 40%, #12201a 0%, #0a0a0a 55%)" }}
        aria-hidden
      />
      {g ? (
        <div
          className="absolute inset-0"
          // A press and release in place opens the hovered product; a drag rotates.
          onPointerDown={(event) => setPress({ x: event.clientX, y: event.clientY })}
          onPointerUp={(event) => {
            const moved = press ? Math.hypot(event.clientX - press.x, event.clientY - press.y) : Infinity;
            setPress(null);
            if (moved < 5 && hover) router.push(`/products/${articleId(g.ids[hover.index])}`);
          }}
        >
          <GalaxyScene
            galaxy={g}
            colours={colours}
            highlight={highlight}
            focus={focus}
            marker={marker}
            autoRotate={!calm}
            animate={!calm}
            onHover={(index, x, y) => setHover(index === null ? null : { index, x, y })}
          />
        </div>
      ) : (
        <div className="absolute inset-0 flex items-center justify-center gap-2 text-sm text-muted">
          <Loader2 className="size-4 animate-spin" aria-hidden /> Loading the product map…
        </div>
      )}

      <div className="pointer-events-none absolute inset-x-0 top-0 flex flex-col gap-3 p-4 sm:p-6 lg:flex-row lg:items-start lg:justify-between">
        <div className="pointer-events-auto w-full max-w-md rounded-2xl border border-border bg-bg/70 p-4 backdrop-blur-md">
          <h1 className="text-lg font-semibold tracking-tight">Product Galaxy</h1>
          <p className="mt-1 hidden text-xs leading-relaxed text-muted sm:block">
            {g ? `${int(g.count)} products` : "Every product"}, placed by how they look. FashionCLIP reads each photo and
            description; UMAP draws them in 3D, so similar items sit together.
          </p>
          <form
            className="mt-3 flex gap-2"
            onSubmit={(event) => {
              event.preventDefault();
              setQuery(text.trim());
            }}
          >
            <div className="relative flex-1">
              <Search className="pointer-events-none absolute left-3 top-2.5 size-4 text-muted" aria-hidden />
              <input
                value={text}
                onChange={(event) => setText(event.target.value)}
                placeholder="Find “white sneakers”"
                maxLength={120}
                aria-label="Search the map"
                className="h-9 w-full rounded-lg border border-border bg-surface-2 pl-9 pr-3 text-sm outline-none placeholder:text-muted focus:border-accent focus:ring-2 focus:ring-accent/20"
              />
            </div>
            <button type="submit" className={buttonClasses("primary")} disabled={text.trim().length < 2}>
              Fly there
            </button>
          </form>
          {(status || search.error) && (
            <div className="mt-2 flex items-center justify-between gap-2 text-xs">
              <span className={search.error ? "text-bad" : "text-accent"}>
                {search.error ? (search.error as Error).message : status}
              </span>
              <button onClick={clearFocus} className="flex items-center gap-1 text-muted hover:text-fg">
                <X className="size-3" /> Clear
              </button>
            </div>
          )}
        </div>
        <div className="pointer-events-auto self-start">
          <Segmented<ColourMode> label="Colour points by" value={mode} options={MODES} onChange={setMode} />
        </div>
      </div>

      {g && (
        <div className="pointer-events-none absolute inset-x-0 bottom-0 flex items-end justify-between gap-3 p-4 sm:p-6">
          <div className="pointer-events-auto max-h-[26dvh] overflow-y-auto rounded-2xl border border-border bg-bg/70 p-3 backdrop-blur-md sm:max-h-[42dvh]">
            <Legend
              galaxy={g}
              mode={mode}
              hidden={mode === "colour" ? hiddenColours : hiddenGroups}
              onToggle={
                mode === "colour"
                  ? toggle(hiddenColours, setHiddenColours)
                  : toggle(hiddenGroups, setHiddenGroups)
              }
            />
          </div>
          <div className="pointer-events-auto hidden flex-col items-end gap-2 sm:flex">
            <button onClick={clearFocus} className={buttonClasses("secondary", "sm")}>
              <LocateFixed className="size-3.5" /> Reset view
            </button>
            <p className="text-right text-[11px] leading-relaxed text-muted">
              Drag to rotate · scroll to zoom · right-drag to pan
              <br />
              Hover a point to preview · click to open
            </p>
          </div>
        </div>
      )}

      {g && hover && <HoverCard galaxy={g} index={hover.index} x={hover.x} y={hover.y} />}
    </div>
  );
}
