import { fixed, int, pct, units } from "./format";
import type { Reason, RiskBand } from "./types";

export const REASONS: Record<Reason, { label: string; hint: string }> = {
  buy_again: { label: "Buy again", hint: "Bought before; repurchase is common in fashion" },
  bought_together: { label: "Bought together", hint: "Often bought with items this customer owns" },
  similar_style: { label: "Similar style", hint: "Looks like what they buy (FashionCLIP)" },
  similar_shoppers: { label: "Similar shoppers", hint: "Bought by shoppers like them (ALS)" },
  popular_in_age_group: { label: "Popular in age group", hint: "Selling well with their age band" },
  trending: { label: "Trending", hint: "Best sellers in the last week" },
};

export const RISK: Record<RiskBand, { label: string; className: string }> = {
  low: { label: "Low risk", className: "bg-good/12 text-good" },
  medium: { label: "Medium risk", className: "bg-warn/15 text-warn" },
  high: { label: "High risk", className: "bg-bad/12 text-bad" },
  lapsed: { label: "Lapsed", className: "bg-surface-2 text-muted" },
};

type Kind = "days" | "count" | "share" | "units" | "ratio";

/** Churn model features in words, with how to show their values. */
export const FEATURES: Record<string, { label: string; kind: Kind }> = {
  recency_days: { label: "Days since last purchase", kind: "days" },
  tenure_days: { label: "Days since first purchase", kind: "days" },
  mean_gap_days: { label: "Average days between visits", kind: "days" },
  last_gap_days: { label: "Days between the last two visits", kind: "days" },
  visits_30d: { label: "Visits, last 30 days", kind: "count" },
  visits_90d: { label: "Visits, last 90 days", kind: "count" },
  visits_180d: { label: "Visits, last 180 days", kind: "count" },
  visits_365d: { label: "Visits, last 12 months", kind: "count" },
  spend_30d: { label: "Spend, last 30 days", kind: "units" },
  spend_90d: { label: "Spend, last 90 days", kind: "units" },
  spend_365d: { label: "Spend, last 12 months", kind: "units" },
  items_365d: { label: "Items, last 12 months", kind: "count" },
  distinct_articles_365d: { label: "Different articles, last 12 months", kind: "count" },
  avg_visit_spend: { label: "Average spend per visit", kind: "units" },
  items_per_visit: { label: "Items per visit", kind: "ratio" },
  online_share: { label: "Share of items bought online", kind: "share" },
  mean_price: { label: "Average item price", kind: "units" },
  share_ladieswear: { label: "Ladieswear share", kind: "share" },
  share_divided: { label: "Divided (young fashion) share", kind: "share" },
  share_menswear: { label: "Menswear share", kind: "share" },
  share_kids: { label: "Baby/Children share", kind: "share" },
  share_sport: { label: "Sport share", kind: "share" },
  spend_trend: { label: "Spend trend (90 days vs yearly pace)", kind: "ratio" },
  visit_trend: { label: "Visit trend (90 days vs yearly pace)", kind: "ratio" },
  age: { label: "Age", kind: "count" },
};

export function featureLabel(feature: string): string {
  return FEATURES[feature]?.label ?? feature.replace(/_/g, " ");
}

export function featureValue(feature: string, value: number | null | undefined): string {
  if (value === null || value === undefined) return "—";
  switch (FEATURES[feature]?.kind) {
    case "days":
      return `${int(value)} days`;
    case "count":
      return int(value);
    case "share":
      return pct(value, 0);
    case "units":
      return units(value);
    case "ratio":
      return fixed(value, 2);
    default:
      return fixed(value, 3);
  }
}
