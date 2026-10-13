const compactFormat = new Intl.NumberFormat("en", {
  notation: "compact",
  maximumFractionDigits: 1,
});
const intFormat = new Intl.NumberFormat("en");

const missing = (x: number | null | undefined): x is null | undefined =>
  x === null || x === undefined || Number.isNaN(x);

export const compact = (x: number | null | undefined) =>
  missing(x) ? "—" : compactFormat.format(x);

export const int = (x: number | null | undefined) =>
  missing(x) ? "—" : intFormat.format(Math.round(x));

export const pct = (x: number | null | undefined, digits = 1) =>
  missing(x) ? "—" : `${(x * 100).toFixed(digits)}%`;

export const signedPct = (x: number | null | undefined, digits = 1) =>
  missing(x) ? "—" : `${x > 0 ? "+" : ""}${(x * 100).toFixed(digits)}%`;

export const fixed = (x: number | null | undefined, digits = 3) =>
  missing(x) ? "—" : x.toFixed(digits);

/** H&M prices are normalized units, so show significant digits rather than currency. */
export function units(x: number | null | undefined): string {
  if (missing(x)) return "—";
  const size = Math.abs(x);
  if (size >= 1000) return compactFormat.format(x);
  if (size >= 10) return x.toFixed(1);
  if (size >= 1) return x.toFixed(2);
  if (size === 0) return "0";
  return String(Number(x.toPrecision(3)));
}

const dateFormat = new Intl.DateTimeFormat("en-GB", {
  day: "numeric",
  month: "short",
  year: "numeric",
  timeZone: "UTC",
});
const monthFormat = new Intl.DateTimeFormat("en-GB", {
  month: "short",
  year: "2-digit",
  timeZone: "UTC",
});

const utc = (value: string) => new Date(`${value.slice(0, 10)}T00:00:00Z`);

export const date = (value: string | null | undefined) => (value ? dateFormat.format(utc(value)) : "—");

export const month = (value: string) => monthFormat.format(utc(value));

export const shortId = (id: string) => `${id.slice(0, 8)}…${id.slice(-4)}`;

export const sentence = (text: string) =>
  text.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());
