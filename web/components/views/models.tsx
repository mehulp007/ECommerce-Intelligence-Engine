"use client";

import { useQuery } from "@tanstack/react-query";
import { CheckCircle2, XCircle } from "lucide-react";
import type { ReactNode } from "react";

import { BarsChart, TrendChart } from "@/components/charts";
import { Card, CardTitle, ErrorState, PageHeader, Skeleton, Stat } from "@/components/ui";
import { getJson } from "@/lib/api";
import { compact, date, fixed, int, month, pct } from "@/lib/format";
import { featureLabel, REASONS } from "@/lib/labels";
import type { ModelReports, Reason } from "@/lib/types";

const RECSYS_NAMES: Record<string, string> = {
  ranker: "Two-stage ranker (LightGBM LambdaRank)",
  repurchase_then_popular: "Repurchase, then popular",
  cooc_then_popular: "Bought together, then popular",
  als: "ALS matrix factorization",
  popular: "Last week's best sellers",
};

const SOURCE_NAMES: Record<string, string> = {
  repurchase: "Repurchase",
  popular: "Popular",
  age_popular: "Popular in age group",
  cooc: "Bought together",
  als: "ALS",
  style: "Style (FashionCLIP)",
  all: "All sources combined",
};

const CHURN_NAMES: Record<string, string> = {
  xgboost_tuned: "XGBoost, tuned + calibrated",
  xgboost_tuned_uncalibrated: "XGBoost, tuned, uncalibrated",
  xgboost_default: "XGBoost, default + calibrated",
  logistic_all_features: "Logistic regression, all features",
  logistic_recency_only: "Logistic regression, recency only",
};

const CLV_NAMES: Record<string, string> = {
  bg_nbd_gamma_gamma: "BG/NBD + Gamma-Gamma",
  previous_period: "Same as the previous period",
  lifetime_average_rate: "Lifetime average rate",
};

function Section({ title, description, children }: { title: string; description: string; children: ReactNode }) {
  return (
    <section className="mt-10 first:mt-0">
      <h2 className="text-lg font-semibold tracking-tight">{title}</h2>
      <p className="mb-4 mt-1 max-w-3xl text-sm text-muted">{description}</p>
      {children}
    </section>
  );
}

function Table({ head, rows }: { head: string[]; rows: ReactNode[][] }) {
  return (
    <div className="-mx-4 overflow-x-auto sm:-mx-5">
      <table className="w-full min-w-[520px] text-sm">
        <thead className="text-xs text-muted">
          <tr className="border-b border-border">
            {head.map((h, i) => (
              <th key={h} className={`px-4 py-2 font-medium sm:px-5 ${i ? "text-right" : "text-left"}`}>
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, i) => (
            <tr key={i} className="border-b border-border last:border-0">
              {row.map((value, j) => (
                <td key={j} className={`px-4 py-2 sm:px-5 ${j ? "text-right tabular-nums" : ""}`}>
                  {value}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Recommender({ r }: { r: NonNullable<ModelReports["recommender_metrics"]> }) {
  const ranker = r.results.ranker.map_at_12;
  const results = Object.entries(r.results)
    .sort((a, b) => b[1].map_at_12 - a[1].map_at_12)
    .map(([name, m]) => ({ name: RECSYS_NAMES[name] ?? name, map: m.map_at_12, recall: m.recall_at_12 }));
  const ablations = Object.entries(r.ablations)
    .map(([name, m]) => ({ name: name.replace("without_", "").replace(/_/g, " "), change: m.map_at_12 / ranker - 1 }))
    .sort((a, b) => a.change - b.change);
  const recall = Object.entries(r.candidate_recall).map(([source, value]) => ({
    source: SOURCE_NAMES[source] ?? source,
    recall: value,
  }));
  const gains = Object.entries(r.ranker.top_features_by_gain)
    .slice(0, 12)
    .map(([feature, gain]) => ({ feature, gain }));
  const reasons = Object.entries(r.scoring.reason_share).map(([reason, share]) => ({
    reason: REASONS[reason as Reason]?.label ?? reason,
    share,
  }));
  const popular = r.results.popular.map_at_12;

  return (
    <Section
      title="Recommender"
      description={`Trained on the label weeks starting ${r.as_of.training_label_weeks.map(date).join(", ")} and evaluated on the held-out final week starting ${date(r.as_of.validation_week[0])}, with MAP@12 (the Kaggle competition metric) over the ${int(r.validation_customers)} customers who bought that week.`}
    >
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat label="MAP@12, ranker" value={fixed(ranker, 4)} hint={`${(ranker / popular).toFixed(1)}× best-sellers baseline`} tone="good" />
        <Stat label="Recall@12" value={pct(r.results.ranker.recall_at_12)} />
        <Stat label="Candidates per customer" value={fixed(r.candidates_per_customer, 0)} hint={`${pct(r.candidate_recall.all)} of purchases covered`} />
        <Stat label="Catalog coverage" value={pct(r.scoring.catalog_coverage)} hint={`${compact(r.scoring.rows)} recommendations served`} />
      </div>
      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <Card>
          <CardTitle hint="Held-out final week">Ranker vs baselines</CardTitle>
          <Table
            head={["Method", "MAP@12", "Recall@12"]}
            rows={results.map((m) => [m.name, fixed(m.map, 4), pct(m.recall)])}
          />
        </Card>
        <Card>
          <CardTitle hint="Change in MAP@12 when one candidate source or feature group is removed and the ranker is retrained">
            Ablations
          </CardTitle>
          <BarsChart
            data={ablations}
            x="name"
            horizontal
            categoryWidth={130}
            height={ablations.length * 28 + 30}
            series={[{ key: "change", label: "Change in MAP@12" }]}
            yFormat={(v) => `${v > 0 ? "+" : ""}${(v * 100).toFixed(1)}%`}
            colorFor={(row) => (Number(row.change) < 0 ? "var(--bad)" : "var(--good)")}
            zeroLine
          />
        </Card>
        <Card>
          <CardTitle hint="Share of the validation week's purchases each source puts in the candidate set">
            Candidate recall by source
          </CardTitle>
          <BarsChart
            data={recall}
            x="source"
            horizontal
            categoryWidth={150}
            height={recall.length * 30 + 30}
            series={[{ key: "recall", label: "Recall" }]}
            yFormat={(v) => pct(v, 1)}
          />
        </Card>
        <Card>
          <CardTitle hint="LightGBM split gain, top 12 of 48 features">Most useful ranker features</CardTitle>
          <BarsChart
            data={gains}
            x="feature"
            horizontal
            categoryWidth={160}
            height={gains.length * 26 + 30}
            series={[{ key: "gain", label: "Gain" }]}
            yFormat={(v) => compact(v)}
          />
        </Card>
      </div>
      <Card className="mt-4">
        <CardTitle hint={`Across all ${compact(r.scoring.customers)} customers' top-12 lists, as of ${date(r.as_of.inference_as_of)}`}>
          Why products are recommended
        </CardTitle>
        <div className="flex h-3 overflow-hidden rounded-full">
          {reasons.map((item, i) => (
            <div
              key={item.reason}
              style={{ width: `${item.share * 100}%`, background: `var(--chart-${(i % 6) + 1})` }}
              title={`${item.reason}: ${pct(item.share)}`}
            />
          ))}
        </div>
        <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-xs">
          {reasons.map((item, i) => (
            <span key={item.reason} className="flex items-center gap-1.5">
              <span className="size-2 rounded-full" style={{ background: `var(--chart-${(i % 6) + 1})` }} />
              {item.reason} <span className="text-muted">{pct(item.share)}</span>
            </span>
          ))}
        </div>
      </Card>
    </Section>
  );
}

function Churn({ c }: { c: NonNullable<ModelReports["customer_metrics"]> }) {
  const churn = c.churn;
  const best = churn.test_results.xgboost_tuned;
  const prevalence = Object.entries(churn.prevalence_by_snapshot).map(([snapshot, v]) => ({
    snapshot,
    rate: v.inactive_rate,
  }));
  const calibration = churn.calibration_deciles.map((d) => ({
    decile: `${d.bin + 1}`,
    predicted: d.mean_predicted,
    observed: d.observed_rate,
  }));
  const contributions = Object.entries(churn.mean_abs_log_odds_contribution)
    .slice(0, 10)
    .map(([feature, value]) => ({ feature: featureLabel(feature), value }));

  return (
    <Section
      title="Inactivity (churn) model"
      description={`Target: ${churn.target}. Population: ${churn.population}. Monthly snapshots: train ${month(churn.split.train_snapshots[0])}–${month(churn.split.train_snapshots.at(-1)!)}, calibrate ${month(churn.split.calibration_snapshots[0])}–${month(churn.split.calibration_snapshots.at(-1)!)}, test ${month(churn.split.test_snapshots[0])}–${month(churn.split.test_snapshots.at(-1)!)}. Features use only purchases before each snapshot.`}
    >
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat label="ROC AUC, test" value={fixed(best.roc_auc, 3)} tone="good" hint={`${compact(best.rows)} customer-months`} />
        <Stat label="Average precision" value={fixed(best.average_precision, 3)} hint={`Base rate ${pct(best.prevalence)}`} />
        <Stat label="Brier score" value={fixed(best.brier, 3)} hint="Lower is better" />
        <Stat label="Customers scored" value={compact(c.scores.active_customers)} hint={`as of ${date(c.scoring_as_of)}`} />
      </div>
      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <Card className="lg:col-span-2">
          <CardTitle hint="Test months May–Jul 2020, after the spring 2020 store closures">Model comparison</CardTitle>
          <Table
            head={["Model", "ROC AUC", "Avg precision", "Brier", "Log loss"]}
            rows={Object.entries(churn.test_results).map(([name, m]) => [
              CHURN_NAMES[name] ?? name,
              fixed(m.roc_auc, 4),
              fixed(m.average_precision, 4),
              fixed(m.brier, 4),
              fixed(m.log_loss, 4),
            ])}
          />
        </Card>
        <Card>
          <CardTitle hint="Mean predicted probability vs observed inactivity, by predicted decile (test months)">
            Calibration
          </CardTitle>
          <TrendChart
            data={calibration}
            x="decile"
            series={[
              { key: "predicted", label: "Predicted" },
              { key: "observed", label: "Observed", color: "var(--chart-3)" },
            ]}
            yFormat={(v) => pct(v, 0)}
            xFormat={(v) => `D${v}`}
          />
          <p className="mt-2 text-xs text-muted">
            The calibration months, March–April 2020, were store-closure months, so this model overstates risk for
            low-risk customers in the test months (lowest decile: {pct(calibration[0].predicted, 0)} predicted,{" "}
            {pct(calibration[0].observed, 0)} observed). The production model is retrained through May 2020 and
            calibrated on June–July 2020.
          </p>
        </Card>
        <Card>
          <CardTitle hint="Share of customers with no purchase in the next 60 days, by snapshot">
            Inactivity rate over time
          </CardTitle>
          <TrendChart
            data={prevalence}
            x="snapshot"
            series={[{ key: "rate", label: "Inactive within 60 days" }]}
            xFormat={month}
            yFormat={(v) => pct(v, 0)}
          />
        </Card>
        <Card className="lg:col-span-2">
          <CardTitle hint="Mean absolute SHAP contribution to the raw log-odds, top 10 of 25 features">
            What drives the predictions
          </CardTitle>
          <BarsChart
            data={contributions}
            x="feature"
            horizontal
            categoryWidth={220}
            height={contributions.length * 28 + 30}
            series={[{ key: "value", label: "Mean |log-odds|" }]}
            yFormat={(v) => fixed(v, 2)}
          />
        </Card>
      </div>
    </Section>
  );
}

function Value({ c }: { c: NonNullable<ModelReports["customer_metrics"]> }) {
  const clv = c.clv;
  const deciles = clv.spend_by_predicted_decile.map((d) => ({
    decile: `${d.decile + 1}`,
    predicted: d.mean_predicted,
    actual: d.mean_actual,
  }));
  const silhouette = Object.entries(c.segments.silhouette_by_k).map(([k, score]) => ({ k, score }));
  return (
    <Section
      title="Lifetime value and segments"
      description={`BG/NBD (visits) and Gamma-Gamma (spend per visit), fitted on purchases up to ${date(clv.calibration_end)} and checked on the next ${clv.holdout_days} days for ${compact(clv.customers)} customers. Values are in normalized price units.`}
    >
      <div className="grid gap-4 lg:grid-cols-2">
        <Card className="lg:col-span-2">
          <CardTitle hint="Per-customer error on the holdout period, against two simple baselines">Holdout accuracy</CardTitle>
          <Table
            head={["Method", "Visits MAE", "Visits corr.", "Spend MAE", "Spend corr.", "Total spend, predicted / actual"]}
            rows={Object.entries(clv.holdout_results).map(([name, m]) => [
              CLV_NAMES[name] ?? name,
              fixed(m.visits.mae, 3),
              fixed(m.visits.correlation, 3),
              fixed(m.spend.mae, 4),
              fixed(m.spend.correlation, 3),
              `${pct(m.spend.total_predicted / m.spend.total_actual, 0)}`,
            ])}
          />
        </Card>
        <Card>
          <CardTitle hint="Mean holdout spend per customer, by predicted-value decile">Predicted vs actual spend</CardTitle>
          <BarsChart
            data={deciles}
            x="decile"
            series={[
              { key: "predicted", label: "Predicted" },
              { key: "actual", label: "Actual", color: "var(--chart-3)" },
            ]}
            xFormat={(v) => `D${v}`}
            yFormat={(v) => fixed(v, 2)}
          />
        </Card>
        <Card>
          <CardTitle hint={`Higher is better; k = ${c.segments.chosen_k} was chosen`}>Segment silhouette by k</CardTitle>
          <BarsChart
            data={silhouette}
            x="k"
            series={[{ key: "score", label: "Silhouette" }]}
            colorFor={(row) => (Number(row.k) === c.segments.chosen_k ? "var(--chart-1)" : "var(--grid)")}
            yFormat={(v) => fixed(v, 2)}
          />
        </Card>
      </div>
    </Section>
  );
}

function SearchQuality({ s }: { s: NonNullable<ModelReports["search_quality"]> }) {
  const queries = Object.entries(s.per_query).map(([query, precision]) => ({ query, precision }));
  return (
    <Section
      title="Product search"
      description={`FashionCLIP image and text embeddings for every article, searched with FAISS. Precision@10 over ${s.queries} fixed queries; a hit is a result of the product type the query asks for.`}
    >
      <Card>
        <CardTitle hint={`Mean precision@10: ${pct(s.precision_at_10, 0)}`}>Precision@10 by query</CardTitle>
        <BarsChart
          data={queries}
          x="query"
          horizontal
          categoryWidth={170}
          height={queries.length * 26 + 30}
          series={[{ key: "precision", label: "Precision@10" }]}
          yFormat={(v) => pct(v, 0)}
        />
      </Card>
    </Section>
  );
}

function DataQuality({ d }: { d: NonNullable<ModelReports["data_quality"]> }) {
  return (
    <Section
      title="Data quality"
      description="Checks run on the raw CSVs after conversion to Parquet. Repeated lines are kept: they are separate units of the same article bought on the same day."
    >
      <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat label="Transactions" value={compact(d.transactions.rows)} hint={`${date(d.transactions.first_date)} – ${date(d.transactions.last_date)}`} />
        <Stat label="Customers" value={compact(d.customers.rows)} hint={`${int(d.customers.missing_age)} without age`} />
        <Stat label="Articles" value={compact(d.articles.rows)} hint={`${compact(d.articles.with_thumbnail)} with photos`} />
        <Stat
          label="Checks"
          value={
            <span className="flex items-center gap-1.5">
              {d.checks.passed ? <CheckCircle2 className="size-5 text-good" /> : <XCircle className="size-5 text-bad" />}
              {d.checks.passed ? "Passed" : "Failed"}
            </span>
          }
          hint={`${int(d.transactions.orphan_customer_rows + d.transactions.orphan_article_rows)} orphan rows, ${int(d.transactions.rows_with_nulls)} with nulls`}
        />
      </div>
    </Section>
  );
}

export function ModelsView() {
  const { data, error, refetch } = useQuery({
    queryKey: ["models"],
    queryFn: () => getJson<ModelReports>("/api/models"),
  });
  if (error) return <ErrorState error={error} onRetry={() => refetch()} />;
  return (
    <>
      <PageHeader
        title="Model Lab"
        description="Offline results for every model behind the app, measured on held-out data with simple baselines for comparison. These numbers come straight from the training reports."
      />
      {!data ? (
        <div className="flex flex-col gap-4">
          <Skeleton className="h-24" />
          <Skeleton className="h-80" />
          <Skeleton className="h-80" />
        </div>
      ) : (
        <>
          {data.recommender_metrics && <Recommender r={data.recommender_metrics} />}
          {data.customer_metrics && <Churn c={data.customer_metrics} />}
          {data.customer_metrics && <Value c={data.customer_metrics} />}
          {data.search_quality && <SearchQuality s={data.search_quality} />}
          {data.data_quality && <DataQuality d={data.data_quality} />}
        </>
      )}
    </>
  );
}
