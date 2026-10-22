import type { components } from "./api-schema";

type Schemas = components["schemas"];

// Generated from the FastAPI OpenAPI schema (npm run gen:api).
export type Article = Schemas["Article"];
export type ArticleDetail = Schemas["ArticleDetail"];
export type ArticlePage = Schemas["ArticlePage"];
export type Facets = Schemas["Facets"];
export type SearchHit = Schemas["SearchHit"];
export type Recommendation = Schemas["Recommendation"];
export type Reason = Recommendation["reason"];
export type Driver = Schemas["Driver"];
export type Customer = Schemas["Customer"];
export type CustomerSample = Schemas["CustomerSample"];
export type RiskBand = Customer["risk_band"];
export type Purchase = Schemas["Purchase"];
export type Segment = Schemas["Segment"];
export type CampaignRequest = Schemas["CampaignRequest"];
export type CampaignPlan = Schemas["CampaignPlan"];
export type CopilotStatus = Schemas["CopilotStatus"];
export type Health = Schemas["Health"];

// The overview and model reports are free-form JSON in the API.
export interface Overview {
  kpis: {
    purchases: number;
    revenue_units: number;
    buyers: number;
    articles_sold: number;
    orders: number;
    online_share: number;
    first_date: string;
    last_date: string;
  };
  weekly: {
    week: string;
    purchases: number;
    revenue_units: number;
    buyers: number;
    online_share: number;
  }[];
  index_groups: { index_group: string; purchases: number; revenue_units: number }[];
  product_groups: {
    product_group: string;
    purchases_4w: number;
    purchases_same_4w_last_year: number;
  }[];
  age_bands: { age_band: number; buyers: number; revenue_units: number }[];
  cohort_retention: {
    cohort: string;
    months_since: number;
    customers: number;
    retention: number;
  }[];
}

export interface Metrics {
  map_at_12: number;
  recall_at_12: number;
}

export interface ClassifierMetrics {
  rows: number;
  prevalence: number;
  roc_auc: number;
  average_precision: number;
  brier: number;
  log_loss: number;
  precision_at_0_5: number;
  recall_at_0_5: number;
  f1_at_0_5: number;
}

export interface ErrorMetrics {
  mae: number;
  rmse: number;
  correlation: number;
  total_predicted: number;
  total_actual: number;
}

export interface ModelReports {
  recommender_metrics?: {
    as_of: {
      validation_week: [string, string];
      training_label_weeks: string[];
      inference_as_of: string;
    };
    validation_customers: number;
    validation_pairs: number;
    candidates_per_customer: number;
    candidate_sizes: Record<string, number>;
    candidate_recall: Record<string, number>;
    results: Record<string, Metrics>;
    ablations: Record<string, Metrics>;
    ranker: {
      rounds: number;
      features: string[];
      top_features_by_gain: Record<string, number>;
    };
    scoring: {
      customers: number;
      rows: number;
      catalog_coverage: number;
      reason_share: Record<string, number>;
    };
    total_seconds: number;
  };
  search_quality?: {
    precision_at_10: number;
    per_query: Record<string, number>;
    queries: number;
  };
  customer_metrics?: {
    scoring_as_of: string;
    churn: {
      target: string;
      population: string;
      split: {
        train_snapshots: string[];
        calibration_snapshots: string[];
        test_snapshots: string[];
        train_rows_sampled: number;
        calibration_rows: number;
        test_rows: number;
      };
      prevalence_by_snapshot: Record<string, { customers: number; inactive_rate: number }>;
      test_results: Record<string, ClassifierMetrics>;
      test_by_month: Record<string, ClassifierMetrics>;
      calibration_deciles: {
        bin: number;
        mean_predicted: number;
        observed_rate: number;
        customers: number;
      }[];
      mean_abs_log_odds_contribution: Record<string, number>;
    };
    clv: {
      calibration_end: string;
      holdout_days: number;
      customers: number;
      holdout_results: Record<string, { visits: ErrorMetrics; spend: ErrorMetrics }>;
      spend_by_predicted_decile: {
        decile: number;
        mean_predicted: number;
        mean_actual: number;
      }[];
    };
    segments: {
      silhouette_by_k: Record<string, number>;
      chosen_k: number;
    };
    scores: {
      customers: number;
      active_customers: number;
      risk_bands: Record<string, number>;
      mean_churn_probability: number;
      total_clv_180d: number;
    };
    total_seconds: number;
  };
  data_quality?: {
    transactions: {
      rows: number;
      first_date: string;
      last_date: string;
      customers: number;
      articles: number;
      repeated_line_rows: number;
      orphan_customer_rows: number;
      orphan_article_rows: number;
      rows_with_nulls: number;
    };
    customers: { rows: number; missing_age: number };
    articles: { rows: number; with_thumbnail: number; never_sold: number };
    checks: { passed: boolean; failures: string[] };
  };
}

/** One Server-Sent Event from the copilot. */
export type CopilotEvent =
  | { type: "thinking"; text: string }
  | { type: "tool_call"; name: string; args: Record<string, unknown> }
  | {
      type: "table";
      result_id: string;
      purpose: string;
      sql: string;
      columns: string[];
      rows: unknown[][];
      row_count: number;
      truncated: boolean;
      seconds: number;
    }
  | {
      type: "chart";
      kind: "bar" | "line";
      title: string;
      x: string;
      y: string[];
      rows: unknown[][];
    }
  | { type: "customer"; customer: Customer; recommendations: Recommendation[] }
  | { type: "products"; title: string; items: SearchHit[] }
  | { type: "campaign"; plan: CampaignPlan; inputs: Record<string, unknown> }
  | { type: "tool_error"; name: string; message: string }
  | { type: "restart"; message: string }
  | { type: "answer"; text: string }
  | { type: "error"; message: string }
  | {
      type: "done";
      model: string;
      seconds: number;
      usage: Record<string, number>;
    };
