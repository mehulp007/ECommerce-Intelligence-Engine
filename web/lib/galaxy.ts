/** Loading and colouring the 3D product map served by /api/galaxy. */

import { errorFrom, getJson } from "./api";
import type { components } from "./api-schema";

export type GalaxyMeta = components["schemas"]["GalaxyMeta"];

export interface Galaxy {
  meta: GalaxyMeta;
  count: number;
  positions: Float32Array;
  /** Article IDs as numbers; pad to 10 digits for URLs. */
  ids: Uint32Array;
  group: Uint8Array;
  colour: Uint8Array;
  sales: Uint16Array;
  priceRank: Uint8Array;
  /** Position of each article ID in the arrays above. */
  index: Map<string, number>;
}

export type ColourMode = "category" | "colour" | "price" | "sales";

export const articleId = (id: number) => String(id).padStart(10, "0");

/** Fetch the map. `stride` keeps every n-th product, for small screens. */
export async function loadGalaxy(stride = 1): Promise<Galaxy> {
  const meta = await getJson<GalaxyMeta>("/api/galaxy");
  const response = await fetch("/api/galaxy/points");
  if (!response.ok) throw await errorFrom(response);
  const buffer = await response.arrayBuffer();
  const field = (name: string) => {
    const f = meta.fields.find((candidate) => candidate.name === name);
    if (!f) throw new Error(`The map has no ${name} field`);
    return f;
  };
  const view = {
    positions: new Float32Array(buffer, field("position").offset, meta.count * 3),
    ids: new Uint32Array(buffer, field("article_id").offset, meta.count),
    group: new Uint8Array(buffer, field("group").offset, meta.count),
    colour: new Uint8Array(buffer, field("colour").offset, meta.count),
    sales: new Uint16Array(buffer, field("sales_4w").offset, meta.count),
    priceRank: new Uint8Array(buffer, field("price_rank").offset, meta.count),
  };
  const keep = stride > 1 ? Math.ceil(meta.count / stride) : meta.count;
  const take = <T extends Float32Array | Uint32Array | Uint8Array | Uint16Array>(source: T, width = 1): T => {
    if (stride <= 1) return source;
    const out = new (source.constructor as new (n: number) => T)(keep * width);
    for (let i = 0, j = 0; i < meta.count; i += stride, j++) {
      for (let k = 0; k < width; k++) out[j * width + k] = source[i * width + k];
    }
    return out;
  };
  const galaxy: Galaxy = {
    meta,
    count: keep,
    positions: take(view.positions, 3),
    ids: take(view.ids),
    group: take(view.group),
    colour: take(view.colour),
    sales: take(view.sales),
    priceRank: take(view.priceRank),
    index: new Map(),
  };
  galaxy.ids.forEach((id, i) => galaxy.index.set(articleId(id), i));
  return galaxy;
}

/** Distinct colours for the largest product groups; the rest share grey. */
const CATEGORY_COLOURS = [
  "#3ddc97",
  "#9b8cff",
  "#5cc8ff",
  "#f5b547",
  "#ff8fb3",
  "#7ee0d6",
  "#c8b6ff",
  "#ffd27a",
  "#ff9e7a",
];
export const OTHER_COLOUR = "#5a5a5a";

export const categoryColour = (code: number) => CATEGORY_COLOURS[code] ?? OTHER_COLOUR;

/** Approximate swatches for H&M's colour group names. */
const GARMENT_COLOURS: Record<string, string> = {
  Black: "#3a3a3a",
  "Dark Blue": "#2f4a8a",
  White: "#f4f4f2",
  "Light Pink": "#f6c7d4",
  Grey: "#9a9a9a",
  "Light Beige": "#e9dcc4",
  Blue: "#3f7fd6",
  Red: "#d93b3b",
  "Light Blue": "#9ccaf0",
  "Greenish Khaki": "#8d8a5a",
  "Dark Grey": "#5e5e5e",
  "Off White": "#efe9dc",
  Beige: "#d8c3a0",
  "Dark Red": "#8e2433",
  "Dark Green": "#2f5a3c",
  "Light Grey": "#c9c9c9",
  Pink: "#ef7fa6",
  Yellow: "#f2d43d",
  "Light Orange": "#f8b98a",
  "Yellowish Brown": "#a87b3a",
  Gold: "#d4af37",
  "Dark Beige": "#b39a73",
  "Light Turquoise": "#98e2dc",
  "Light Yellow": "#f7ec9c",
  "Dark Orange": "#d9661f",
  "Dark Pink": "#c2477a",
  Green: "#3fa35a",
  Orange: "#f28c28",
  "Other Pink": "#e89ab8",
  Silver: "#c0c4c8",
  "Light Green": "#a8dca0",
  "Dark Yellow": "#c9a227",
  "Light Purple": "#c6b3e6",
  "Dark Turquoise": "#1f8a86",
  Turquoise: "#36c2b4",
  "Dark Purple": "#5a3a7a",
  "Light Red": "#f08a80",
  "Greyish Beige": "#b8ad9a",
  "Other Yellow": "#e8d26a",
  Purple: "#8a5ac6",
  "Other Orange": "#eaa060",
  "Other Green": "#6cae6a",
  "Other Red": "#d9605a",
  Other: "#8c8c8c",
  "Bronze/Copper": "#b87333",
  "Other Blue": "#6f9ad6",
  "Other Purple": "#9a7ab8",
  Transparent: "#dfe6ea",
  Unknown: "#6e6e6e",
  "Other Turquoise": "#5cc4bc",
};

export const garmentColour = (name: string) => GARMENT_COLOURS[name] ?? "#8c8c8c";

/** A dim-to-bright ramp for sequential values in [0, 1]. */
function ramp(t: number, stops: [number, number, number][]): [number, number, number] {
  const x = Math.min(1, Math.max(0, t)) * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(x));
  const f = x - i;
  return [0, 1, 2].map((k) => stops[i][k] + (stops[i + 1][k] - stops[i][k]) * f) as [number, number, number];
}

const SALES_RAMP: [number, number, number][] = [
  [0.2, 0.2, 0.22],
  [0.24, 0.86, 0.59],
  [0.9, 1, 0.95],
];
const PRICE_RAMP: [number, number, number][] = [
  [0.2, 0.2, 0.24],
  [0.61, 0.55, 1],
  [0.95, 0.93, 1],
];

function hexToRgb(hex: string): [number, number, number] {
  const n = parseInt(hex.slice(1), 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}

/** Per-point RGB values (0-1) for a colour mode. */
export function pointColours(galaxy: Galaxy, mode: ColourMode): Float32Array {
  const out = new Float32Array(galaxy.count * 3);
  const garment = galaxy.meta.colours.map((name) => hexToRgb(garmentColour(name)));
  const category = galaxy.meta.product_groups.map((_, code) => hexToRgb(categoryColour(code)));
  const logMax = Math.log1p(galaxy.meta.sales_4w_max || 1);
  for (let i = 0; i < galaxy.count; i++) {
    let rgb: [number, number, number];
    if (mode === "category") rgb = category[galaxy.group[i]];
    else if (mode === "colour") rgb = garment[galaxy.colour[i]];
    else if (mode === "sales") rgb = ramp(Math.log1p(galaxy.sales[i]) / logMax, SALES_RAMP);
    else rgb = ramp(galaxy.priceRank[i] / 255, PRICE_RAMP);
    out.set(rgb, i * 3);
  }
  return out;
}
