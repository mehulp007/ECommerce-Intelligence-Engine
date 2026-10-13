"use client";

import { useQuery } from "@tanstack/react-query";
import { ArrowLeft, Check, Copy, Globe, Sparkles, Store } from "lucide-react";
import Link from "next/link";
import { useMemo, useState } from "react";

import { ProductCard, ProductGrid, ProductImage, ReasonBadge, RiskBadge } from "@/components/product";
import { Badge, buttonClasses, Card, CardTitle, EmptyState, ErrorState, Skeleton, Stat } from "@/components/ui";
import { ApiError, getJson } from "@/lib/api";
import { date, fixed, int, pct, units } from "@/lib/format";
import { featureLabel, featureValue, REASONS } from "@/lib/labels";
import type { Customer, Purchase, Reason, Recommendation } from "@/lib/types";
import { cn } from "@/lib/utils";

function CopyId({ id }: { id: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(id);
          setCopied(true);
          setTimeout(() => setCopied(false), 1500);
        } catch {
          // Clipboard access can be refused; the ID is still selectable.
        }
      }}
      className="rounded p-1 text-muted hover:bg-surface-2 hover:text-fg"
      aria-label="Copy customer ID"
    >
      {copied ? <Check className="size-3.5" /> : <Copy className="size-3.5" />}
    </button>
  );
}

function Drivers({ customer }: { customer: Customer }) {
  if (customer.churn_probability === null || customer.drivers.length === 0) {
    return (
      <p className="text-sm text-muted">
        This customer has not bought anything in the last 12 months, so the inactivity model does not score them.
      </p>
    );
  }
  const largest = Math.max(...customer.drivers.map((d) => Math.abs(d.log_odds)));
  return (
    <div className="flex flex-col gap-4">
      {customer.drivers.map((driver) => {
        const raises = driver.log_odds > 0;
        return (
          <div key={driver.feature}>
            <div className="flex items-baseline justify-between gap-3 text-sm">
              <span>{featureLabel(driver.feature)}</span>
              <span className="tabular-nums font-medium">{featureValue(driver.feature, driver.value)}</span>
            </div>
            <div className="mt-1.5 flex items-center gap-2">
              <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-surface-2">
                <div
                  className={cn("h-full rounded-full", raises ? "bg-bad" : "bg-good")}
                  style={{ width: `${(Math.abs(driver.log_odds) / largest) * 100}%` }}
                />
              </div>
              <span className={cn("w-28 text-right text-xs", raises ? "text-bad" : "text-good")}>
                {raises ? "raises" : "lowers"} risk {fixed(Math.abs(driver.log_odds), 2)}
              </span>
            </div>
          </div>
        );
      })}
      <p className="text-xs text-muted">
        The three largest SHAP contributions to the model&apos;s raw log-odds, before calibration.
      </p>
    </div>
  );
}

function Recommendations({ items }: { items: Recommendation[] }) {
  const [reason, setReason] = useState<Reason | "">("");
  const counts = useMemo(() => {
    const map = new Map<Reason, number>();
    items.forEach((r) => map.set(r.reason, (map.get(r.reason) ?? 0) + 1));
    return [...map.entries()];
  }, [items]);
  const shown = reason ? items.filter((r) => r.reason === reason) : items;
  return (
    <>
      <div className="mb-3 flex flex-wrap gap-1.5">
        <button
          onClick={() => setReason("")}
          className={cn(
            "rounded-full border px-2.5 py-1 text-xs",
            reason === "" ? "border-accent bg-accent-soft text-accent" : "border-border text-muted",
          )}
        >
          All {items.length}
        </button>
        {counts.map(([key, count]) => (
          <button
            key={key}
            onClick={() => setReason(key)}
            title={REASONS[key].hint}
            className={cn(
              "rounded-full border px-2.5 py-1 text-xs",
              reason === key ? "border-accent bg-accent-soft text-accent" : "border-border text-muted",
            )}
          >
            {REASONS[key].label} {count}
          </button>
        ))}
      </div>
      <ProductGrid dense>
        {shown.map((r) =>
          r.article ? (
            <ProductCard
              key={r.article_id}
              article={r.article}
              badge={<ReasonBadge reason={r.reason} />}
              footer={<div className="mt-1 text-xs text-muted">Rank {r.rank}</div>}
            />
          ) : null,
        )}
      </ProductGrid>
    </>
  );
}

function History({ rows }: { rows: Purchase[] }) {
  const visits = useMemo(() => {
    const groups = new Map<string, Purchase[]>();
    rows.forEach((row) => {
      const key = `${row.purchase_date}|${row.channel}`;
      groups.set(key, [...(groups.get(key) ?? []), row]);
    });
    return [...groups.entries()];
  }, [rows]);
  if (!rows.length) return <EmptyState>No purchases on record.</EmptyState>;
  return (
    <ol className="relative flex flex-col gap-5 border-l border-border pl-5">
      {visits.map(([key, items]) => {
        const [day, channel] = key.split("|");
        const total = items.reduce((sum, item) => sum + item.price, 0);
        return (
          <li key={key} className="relative">
            <span className="absolute -left-[26px] top-1 size-2.5 rounded-full border-2 border-surface bg-accent" />
            <div className="mb-2 flex flex-wrap items-center gap-2 text-sm">
              <span className="font-medium">{date(day)}</span>
              <Badge>
                {channel === "online" ? <Globe className="size-3" /> : <Store className="size-3" />}
                {channel === "online" ? "Online" : "In store"}
              </Badge>
              <span className="text-xs text-muted">
                {items.length} item{items.length > 1 ? "s" : ""} · {units(total)} units
              </span>
            </div>
            <div className="flex flex-wrap gap-2">
              {items.map((item, i) => (
                <Link
                  key={`${item.article_id}-${i}`}
                  href={`/products/${item.article_id}`}
                  className="w-20"
                  title={item.article?.prod_name ?? item.article_id}
                >
                  <ProductImage
                    article={{
                      article_id: item.article_id,
                      has_image: item.article?.has_image ?? false,
                      prod_name: item.article?.prod_name ?? null,
                    }}
                  />
                  <div className="mt-1 truncate text-[11px] text-muted">{item.article?.prod_name}</div>
                </Link>
              ))}
            </div>
          </li>
        );
      })}
    </ol>
  );
}

export function CustomerDetailView({ id }: { id: string }) {
  const customer = useQuery({
    queryKey: ["customer", id],
    queryFn: () => getJson<Customer>(`/api/customers/${id}`),
  });
  const recommendations = useQuery({
    queryKey: ["recommendations", id],
    queryFn: () => getJson<Recommendation[]>(`/api/customers/${id}/recommendations`),
    enabled: customer.isSuccess,
  });
  const history = useQuery({
    queryKey: ["history", id],
    queryFn: () => getJson<Purchase[]>(`/api/customers/${id}/history`, { limit: 40 }),
    enabled: customer.isSuccess,
  });

  const back = (
    <Link href="/customers" className="mb-4 inline-flex items-center gap-1 text-sm text-muted hover:text-fg">
      <ArrowLeft className="size-4" /> Customers
    </Link>
  );

  if (customer.error) {
    return (
      <>
        {back}
        {customer.error instanceof ApiError && customer.error.status === 404 ? (
          <EmptyState>No customer has this ID.</EmptyState>
        ) : (
          <ErrorState error={customer.error} onRetry={() => customer.refetch()} />
        )}
      </>
    );
  }

  const c = customer.data;
  const question = `Explain the inactivity risk of customer ${id} and draft a short win-back message with two recommended products.`;

  return (
    <>
      {back}
      <div className="mb-6 flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
        <div className="min-w-0">
          <div className="flex items-center gap-1">
            <h1 className="truncate font-mono text-lg font-semibold sm:text-xl" title={id}>
              {id.slice(0, 16)}…
            </h1>
            <CopyId id={id} />
          </div>
          {c ? (
            <div className="mt-2 flex flex-wrap items-center gap-2">
              <RiskBadge band={c.risk_band} />
              <Badge>{c.segment_name}</Badge>
              {c.profile.age ? <Badge>Age {c.profile.age}</Badge> : null}
              {c.profile.club_member_status ? <Badge>Club: {c.profile.club_member_status.toLowerCase()}</Badge> : null}
            </div>
          ) : (
            <Skeleton className="mt-2 h-6 w-72" />
          )}
        </div>
        <Link href={`/copilot?q=${encodeURIComponent(question)}`} className={buttonClasses("primary")}>
          <Sparkles className="size-4" aria-hidden /> Ask the copilot
        </Link>
      </div>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
        {c ? (
          <>
            <Stat
              label="P(inactive next 60 days)"
              value={c.churn_probability === null ? "—" : pct(c.churn_probability)}
              tone={c.risk_band === "high" ? "bad" : c.risk_band === "medium" ? "warn" : c.risk_band === "low" ? "good" : undefined}
              hint="Calibrated XGBoost"
            />
            <Stat label="Value, next 180 days" value={units(c.clv_180d)} hint="BG/NBD × Gamma-Gamma" />
            <Stat label="P(still a customer)" value={pct(c.p_alive)} hint="BG/NBD" />
            <Stat label="Expected visits, 180 days" value={fixed(c.expected_visits_180d, 2)} />
            <Stat label="Visits, last 12 months" value={int(c.visits_365d)} hint={`${units(c.spend_365d)} units spent`} />
            <Stat
              label="Last purchase"
              value={c.recency_days === null ? "—" : `${int(c.recency_days)}d`}
              hint={`ago, on ${date(c.summary.last_purchase)}`}
            />
          </>
        ) : (
          Array.from({ length: 6 }, (_, i) => <Skeleton key={i} className="h-[92px]" />)
        )}
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-3">
        <Card>
          <CardTitle hint="What moves this customer's predicted risk most">Why this risk</CardTitle>
          {c ? <Drivers customer={c} /> : <Skeleton className="h-40" />}
        </Card>
        <Card className="lg:col-span-2">
          <CardTitle hint="All purchases in the data, 20 Sep 2018 to 22 Sep 2020">History at a glance</CardTitle>
          {c ? (
            <dl className="grid grid-cols-2 gap-x-6 gap-y-3 text-sm sm:grid-cols-3">
              {[
                ["Items bought", int(c.summary.purchases)],
                ["Shopping visits", int(c.summary.visits)],
                ["Total spend", `${units(c.summary.spend_units)} units`],
                ["First purchase", date(c.summary.first_purchase)],
                ["Last purchase", date(c.summary.last_purchase)],
                ["Bought online", pct(c.summary.online_share, 0)],
                ["Fashion news", c.profile.fashion_news_frequency ?? "—"],
                ["Expected spend per visit", units(c.expected_visit_spend)],
                ["Segment", c.segment_name],
              ].map(([label, value]) => (
                <div key={label}>
                  <dt className="text-xs text-muted">{label}</dt>
                  <dd className="mt-0.5 font-medium">{value}</dd>
                </div>
              ))}
            </dl>
          ) : (
            <Skeleton className="h-40" />
          )}
        </Card>
      </div>

      <Card className="mt-4">
        <CardTitle hint="Top 12 from the two-stage recommender, as of 23 Sep 2020, with the source of each candidate">
          Recommended next
        </CardTitle>
        {recommendations.error ? (
          <ErrorState error={recommendations.error} />
        ) : recommendations.data ? (
          recommendations.data.length ? (
            <Recommendations items={recommendations.data} />
          ) : (
            <EmptyState>No recommendations for this customer.</EmptyState>
          )
        ) : (
          <Skeleton className="h-64" />
        )}
      </Card>

      <Card className="mt-4">
        <CardTitle hint="Most recent 40 items, grouped by visit">Purchase timeline</CardTitle>
        {history.error ? (
          <ErrorState error={history.error} />
        ) : history.data ? (
          <History rows={history.data} />
        ) : (
          <Skeleton className="h-64" />
        )}
      </Card>
    </>
  );
}
