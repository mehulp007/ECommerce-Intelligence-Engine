"use client";

import { useQuery } from "@tanstack/react-query";
import { Megaphone, Users } from "lucide-react";
import Link from "next/link";

import { BubbleChart, COLORS } from "@/components/charts";
import { buttonClasses, Card, CardTitle, ErrorState, PageHeader, Skeleton } from "@/components/ui";
import { getJson } from "@/lib/api";
import { compact, int, pct, units } from "@/lib/format";
import type { Segment } from "@/lib/types";

export function SegmentsView() {
  const { data, error, refetch } = useQuery({
    queryKey: ["segments"],
    queryFn: () => getJson<Segment[]>("/api/segments"),
  });
  if (error) return <ErrorState error={error} onRetry={() => refetch()} />;

  const active = data?.filter((s) => s.segment_id >= 0) ?? [];
  const lapsed = data?.find((s) => s.segment_id < 0);
  const total = data?.reduce((sum, s) => sum + s.customers, 0) ?? 0;
  const totalValue = data?.reduce((sum, s) => sum + (s.total_clv_180d ?? 0), 0) ?? 0;

  return (
    <>
      <PageHeader
        title="Segments"
        description="K-Means clusters of active customers on behaviour (visits, spend, recency, channel, category mix, price, age) plus an 8-number summary of the styles they buy, from FashionCLIP. Names come from each cluster's own profile."
      />

      <div className="grid gap-4 lg:grid-cols-[1.3fr_1fr]">
        <Card>
          <CardTitle hint="Bubble size is the number of customers">Risk against value</CardTitle>
          {data ? (
            <BubbleChart
              data={active.map((s) => ({ ...s, risk: s.mean_churn_probability, value: s.median_clv_180d }))}
              x="risk"
              y="value"
              size="customers"
              label="segment_name"
              xLabel="Mean P(inactive 60 days)"
              yLabel="Median value, next 180 days"
              xFormat={(v) => pct(v, 0)}
              yFormat={(v) => units(v)}
              height={280}
            />
          ) : (
            <Skeleton className="h-[280px]" />
          )}
          <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted">
            {active.map((s, i) => (
              <span key={s.segment_id} className="flex items-center gap-1.5">
                <span className="size-2 rounded-full" style={{ background: COLORS[i % COLORS.length] }} />
                {s.segment_name}
              </span>
            ))}
          </div>
        </Card>
        <Card>
          <CardTitle hint="Predicted value over the next 180 days, in price units">Where the value is</CardTitle>
          {data ? (
            <div className="flex flex-col gap-3">
              {data.map((s) => {
                const i = active.indexOf(s);
                const share = (s.total_clv_180d ?? 0) / totalValue;
                return (
                  <div key={s.segment_id}>
                    <div className="mb-1 flex justify-between gap-2 text-sm">
                      <span className="truncate">{s.segment_name}</span>
                      <span className="shrink-0 tabular-nums text-muted">
                        {pct(share, 0)} · {units(s.total_clv_180d)}
                      </span>
                    </div>
                    <div className="h-2 overflow-hidden rounded-full bg-surface-2">
                      <div
                        className="h-full rounded-full"
                        style={{
                          width: `${share * 100}%`,
                          background: i >= 0 ? COLORS[i % COLORS.length] : "var(--muted)",
                        }}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          ) : (
            <Skeleton className="h-[300px]" />
          )}
        </Card>
      </div>

      <div className="mt-4 grid gap-4 md:grid-cols-2 xl:grid-cols-3">
        {data
          ? [...active, ...(lapsed ? [lapsed] : [])].map((s) => {
              const i = active.indexOf(s);
              return (
                <Card key={s.segment_id} className="flex flex-col">
                  <div className="flex items-start gap-3">
                    <span
                      className="mt-1.5 size-2.5 shrink-0 rounded-full"
                      style={{ background: i >= 0 ? COLORS[i % COLORS.length] : "var(--muted)" }}
                    />
                    <div>
                      <h2 className="font-semibold">{s.segment_name}</h2>
                      <p className="text-xs text-muted">
                        {int(s.customers)} customers · {pct(s.customers / total, 0)} of all
                      </p>
                    </div>
                  </div>
                  <dl className="mt-4 grid grid-cols-2 gap-x-4 gap-y-3 text-sm">
                    <div>
                      <dt className="text-xs text-muted">Mean P(inactive 60d)</dt>
                      <dd className="font-medium tabular-nums">{pct(s.mean_churn_probability)}</dd>
                    </div>
                    <div>
                      <dt className="text-xs text-muted">Median value, 180d</dt>
                      <dd className="font-medium tabular-nums">{units(s.median_clv_180d)}</dd>
                    </div>
                    <div>
                      <dt className="text-xs text-muted">Median visits, 12 mo</dt>
                      <dd className="font-medium tabular-nums">{s.median_visits_365d ?? "—"}</dd>
                    </div>
                    <div>
                      <dt className="text-xs text-muted">Median spend, 12 mo</dt>
                      <dd className="font-medium tabular-nums">{units(s.median_spend_365d)}</dd>
                    </div>
                  </dl>
                  {s.segment_id >= 0 ? (
                    <div className="mt-auto flex flex-wrap gap-2 pt-4">
                      <Link href={`/customers?segment=${s.segment_id}`} className={buttonClasses("secondary", "sm")}>
                        <Users className="size-3.5" /> Customers
                      </Link>
                      <Link
                        href={`/campaigns?segment=${encodeURIComponent(s.segment_name)}`}
                        className={buttonClasses("secondary", "sm")}
                      >
                        <Megaphone className="size-3.5" /> Plan a campaign
                      </Link>
                    </div>
                  ) : (
                    <p className="mt-auto pt-4 text-xs text-muted">
                      No purchase in the 12 months before 23 Sep 2020, so they are not scored for inactivity. Their
                      value comes from the lifetime value model only.
                    </p>
                  )}
                </Card>
              );
            })
          : Array.from({ length: 6 }, (_, i) => <Skeleton key={i} className="h-52" />)}
      </div>
      {data && (
        <p className="mt-3 text-xs text-muted">
          {compact(total)} customers in total; predicted 180-day value across all of them is {units(totalValue)} units.
        </p>
      )}
    </>
  );
}
