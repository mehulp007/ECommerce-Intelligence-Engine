/** Data for the scroll story: the anonymous customer sample and scroll state. */

import { errorFrom, getJson } from "./api";
import type { components } from "./api-schema";

export type StoryCustomersMeta = components["schemas"]["StoryCustomersMeta"];

export interface StoryCustomers {
  meta: StoryCustomersMeta;
  count: number;
  positions: Float32Array;
  churn: Float32Array;
  clv: Float32Array;
  segment: Uint8Array;
  risk: Uint8Array;
}

/** Fetch the sample; `stride` keeps every n-th customer, for small screens. */
export async function loadStoryCustomers(stride = 1): Promise<StoryCustomers> {
  const meta = await getJson<StoryCustomersMeta>("/api/story/customers");
  const response = await fetch("/api/story/customers/points");
  if (!response.ok) throw await errorFrom(response);
  const buffer = await response.arrayBuffer();
  const offset = (name: string) => meta.fields.find((f) => f.name === name)!.offset;
  const n = meta.count;
  const full = {
    positions: new Float32Array(buffer, offset("position"), n * 3),
    churn: new Float32Array(buffer, offset("churn_probability"), n),
    clv: new Float32Array(buffer, offset("clv_180d"), n),
    segment: new Uint8Array(buffer, offset("segment_id"), n),
    risk: new Uint8Array(buffer, offset("risk_band"), n),
  };
  if (stride <= 1) return { meta, count: n, ...full };
  const keep = Math.ceil(n / stride);
  const out = {
    positions: new Float32Array(keep * 3),
    churn: new Float32Array(keep),
    clv: new Float32Array(keep),
    segment: new Uint8Array(keep),
    risk: new Uint8Array(keep),
  };
  for (let i = 0, j = 0; i < n; i += stride, j++) {
    out.positions.set(full.positions.subarray(i * 3, i * 3 + 3), j * 3);
    out.churn[j] = full.churn[i];
    out.clv[j] = full.clv[i];
    out.segment[j] = full.segment[i];
    out.risk[j] = full.risk[i];
  }
  return { meta, count: keep, ...out };
}

export const SCENES = 7;

/**
 * Where the reader is in the story, shared by the HTML and the 3D scene
 * without React re-renders: `position` runs from 0 to SCENES, and its integer
 * part is the scene on screen. The values are mutable on purpose.
 */
export const storyScroll = {
  position: 0,
  /** Product indices to light up in the search scene. */
  searchMatches: [] as number[],
  /** How many sampled customers the campaign scene targets. */
  targeted: 0,
};

/** 0..1 progress through scene `index` at story position `position`. */
export function sceneProgress(position: number, index: number): number {
  return Math.min(1, Math.max(0, position - index));
}

/** How visible a scene's 3D layer is, fading over the boundaries. */
export function sceneWeight(position: number, index: number, fade = 0.1): number {
  const enter = smoothstep(index - fade, index + fade * 0.5, position);
  const leave = 1 - smoothstep(index + 1 - fade * 0.5, index + 1 + fade, position);
  return Math.min(enter, leave);
}

export function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}
