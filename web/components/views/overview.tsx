"use client";

import { useQuery } from "@tanstack/react-query";
import { Store } from "lucide-react";
import { useMemo } from "react";

import { BarsChart, TrendChart } from "@/components/charts";
import { Card, CardTitle, ErrorState, PageHeader, Skeleton, Stat } from "@/components/ui";
import { getJson } from "@/lib/api";
import { compact, date, int, month, pct, signedPct, units } from "@/lib/format";
import type { Overview } from "@/lib/types";

const DAY = 86_400_000;
const days = (from: string, to: string) => (Date.parse(to) - Date.parse(from)) / DAY;

/** Weeks fully inside the data (the first and last weeks are partial). */
function fullWeeks(overview: Overview) {
  const { first_date, last_date } = overview.kpis;
  return overview.weekly.filter(
    (w) => days(first_date, w.week) >= 0 && days(w.week, last_date) >= 6,
  );
}

const MIN_GROUP_ITEMS = 5000;

/** Buyers per age band, with the small bands from 70 up merged. */
function ageBands(rows: Overview["age_bands"]) {
  const bands = new Map<string, number>();
  for (const row of rows) {
    const band = row.age_band < 20 ? "Under 20" : row.age_band >= 70 ? "70+" : `${row.age_band}s`;
    bands.set(band, (bands.get(band) ?? 0) + row.buyers);
  }
  return [...bands].map(([band, buyers]) => ({ band, buyers }));
}

/** The longest run of weeks with essentially no store sales (over 99.9% online). */
function storeClosure(weeks: Overview["weekly"]) {
  let best: { from: string; to: string; weeks: number } | null = null;
  let start = -1;
  weeks.forEach((w, i) => {
    const online = w.online_share >= 0.999;
    if (online && start < 0) start = i;
    const ends = start >= 0 && (!online || i === weeks.length - 1);
    if (ends) {
      const last = online ? i : i - 1;
      const length = last - start + 1;
      if (!best || length > best.weeks) best = { from: weeks[start].week, to: weeks[last].week, weeks: length };
      start = -1;
    }
  });
  return best as { from: string; to: string; weeks: number } | null;
}

function Cohorts({ rows }: { rows: Overview["cohort_retention"] }) {
  const cohorts = [...new Set(rows.map((r) => r.cohort))].sort();
  const months = Array.from({ length: 12 }, (_, i) => i + 1);
  const cell = new Map(rows.map((r) => [`${r.cohort}|${r.months_since}`, r]));
  const max = Math.max(...rows.filter((r) => r.months_since > 0).map((r) => r.retention));
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[640px] border-separate border-spacing-0.5 text-xs">
        <thead>
          <tr className="text-muted">
            <th className="px-2 py-1 text-left font-medium">First purchase</th>
            <th className="px-2 py-1 text-right font-medium">Customers</th>
            {months.map((m) => (
              <th key={m} className="px-1 py-1 font-medium">
                +{m}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {cohorts.map((cohort) => (
            <tr key={cohort}>
              <td className="whitespace-nowrap px-2 py-1">{month(cohort)}</td>
              <td className="px-2 py-1 text-right tabular-nums text-muted">
                {compact(cell.get(`${cohort}|0`)?.customers)}
              </td>
              {months.map((m) => {
                const r = cell.get(`${cohort}|${m}`);
                return (
                  <td
                    key={m}
                    className="rounded px-1 py-1 text-center tabular-nums"
                    style={
                      r
                        ? {
                            background: `color-mix(in oklab, var(--accent) ${Math.round((r.retention / max) * 85)}%, transparent)`,
                            color: r.retention / max > 0.55 ? "var(--accent-fg)" : undefined,
                          }
                        : undefined
                    }
                    title={r ? `${month(cohort)} cohort, month +${m}: ${pct(r.retention)}` : undefined}
                  >
                    {r ? pct(r.retention, 0) : ""}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export function OverviewView() {
  const { data, error, isPending, refetch } = useQuery({
    queryKey: ["overview"],
    queryFn: () => getJson<Overview>("/api/overview"),
  });

  const derived = useMemo(() => {
    if (!data) return null;
    const weeks = fullWeeks(data);
    const closure = storeClosure(weeks);
    const groups = data.product_groups
      // Small or uncategorized groups swing wildly, so they are left out.
      .filter(
        (g) =>
          g.product_group !== "Unknown" &&
          Math.min(g.purchases_4w, g.purchases_same_4w_last_year) >= MIN_GROUP_ITEMS,
      )
      .map((g) => ({
        group: g.product_group,
        growth: g.purchases_4w / g.purchases_same_4w_last_year - 1,
        purchases: g.purchases_4w,
      }))
      .sort((a, b) => b.growth - a.growth);
    const index = [...data.index_groups].sort((a, b) => b.revenue_units - a.revenue_units);
    const totalRevenue = index.reduce((sum, g) => sum + g.revenue_units, 0);
    return {
      weeks,
      closure,
      groups,
      index: index.map((g) => ({ ...g, share: g.revenue_units / totalRevenue })),
      ages: ageBands(data.age_bands),
    };
  }, [data]);

  if (error) return <ErrorState error={error} onRetry={() => refetch()} />;

  return (
    <>
      <PageHeader
        title="Overview"
        description={
          data
            ? `Every purchase from ${date(data.kpis.first_date)} to ${date(data.kpis.last_date)}. Prices are H&M's normalized price units, not a real currency.`
            : "Loading the purchase history…"
        }
      />

      <div className="grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
        {isPending || !data ? (
          Array.from({ length: 6 }, (_, i) => <Skeleton key={i} className="h-[92px]" />)
        ) : (
          <>
            <Stat label="Items purchased" value={compact(data.kpis.purchases)} hint={int(data.kpis.purchases)} />
            <Stat label="Shopping visits" value={compact(data.kpis.orders)} hint="Customer-days with a purchase" />
            <Stat label="Customers" value={compact(data.kpis.buyers)} hint="With at least one purchase" />
            <Stat label="Articles sold" value={compact(data.kpis.articles_sold)} />
            <Stat label="Revenue" value={units(data.kpis.revenue_units)} hint="Price units" />
            <Stat label="Bought online" value={pct(data.kpis.online_share)} hint="Share of items" />
          </>
        )}
      </div>

      <div className="mt-4 grid gap-4 xl:grid-cols-3">
        <Card className="xl:col-span-2">
          <CardTitle hint="Full weeks only. In the shaded weeks over 99.9% of items were bought online.">
            Weekly revenue and online share
          </CardTitle>
          {derived ? (
            <TrendChart
              data={derived.weeks}
              x="week"
              height={300}
              series={[
                { key: "revenue_units", label: "Revenue (units)" },
                { key: "online_share", label: "Online share", right: true, color: "var(--chart-3)" },
              ]}
              xFormat={month}
              yFormat={(v) => compact(v)}
              rightFormat={(v) => pct(v, 0)}
              bands={
                derived.closure
                  ? [{ from: derived.closure.from, to: derived.closure.to, label: "Stores closed" }]
                  : []
              }
            />
          ) : (
            <Skeleton className="h-[300px]" />
          )}
        </Card>

        <Card>
          <CardTitle>What the data shows</CardTitle>
          {derived?.closure ? (
            <div className="flex flex-col gap-4 text-sm">
              <div className="flex gap-3">
                <span className="mt-0.5 flex size-8 shrink-0 items-center justify-center rounded-lg bg-warn/15 text-warn">
                  <Store className="size-4" aria-hidden />
                </span>
                <p>
                  <strong>{derived.closure.weeks} weeks with almost no store sales.</strong> From the week of{" "}
                  {date(derived.closure.from)} to the week of {date(derived.closure.to)}, over 99.9% of items were
                  bought online, consistent with the spring 2020 store closures.
                </p>
              </div>
              <p className="text-muted">
                Store sales returned in May 2020. The churn model is trained on months before this shift and tested on
                months after it, so its test results include the shift.
              </p>
              <div className="rounded-lg bg-surface-2 p-3">
                <div className="text-xs text-muted">Fastest-growing product group, last 4 weeks vs a year earlier</div>
                <div className="mt-1 font-medium">
                  {derived.groups[0].group}{" "}
                  <span className="text-good">{signedPct(derived.groups[0].growth)}</span>
                </div>
              </div>
            </div>
          ) : (
            <Skeleton className="h-48" />
          )}
        </Card>
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <Card>
          <CardTitle hint="Items sold in the last 4 weeks vs the same 4 weeks a year earlier (groups with 5,000+ items in both)">
            Product group growth
          </CardTitle>
          {derived ? (
            <BarsChart
              data={derived.groups}
              x="group"
              horizontal
              height={Math.max(220, derived.groups.length * 30)}
              categoryWidth={150}
              series={[{ key: "growth", label: "Change" }]}
              yFormat={(v) => signedPct(v, 0)}
              colorFor={(row) => (Number(row.growth) >= 0 ? "var(--good)" : "var(--bad)")}
              zeroLine
            />
          ) : (
            <Skeleton className="h-[300px]" />
          )}
        </Card>

        <Card>
          <CardTitle hint="Share of all revenue, 2018–2020">Revenue by index group</CardTitle>
          {derived ? (
            <div className="flex flex-col gap-3">
              {derived.index.map((g) => (
                <div key={g.index_group}>
                  <div className="mb-1 flex justify-between text-sm">
                    <span>{g.index_group}</span>
                    <span className="tabular-nums text-muted">
                      {pct(g.share)} · {compact(g.purchases)} items
                    </span>
                  </div>
                  <div className="h-2 overflow-hidden rounded-full bg-surface-2">
                    <div className="h-full rounded-full bg-accent" style={{ width: `${g.share * 100}%` }} />
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <Skeleton className="h-[300px]" />
          )}
        </Card>
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-3 lg:items-start">
        <Card>
          <CardTitle hint="Customers with a purchase, by age band">Buyers by age</CardTitle>
          {derived ? (
            <BarsChart
              data={derived.ages}
              x="band"
              height={240}
              series={[{ key: "buyers", label: "Buyers" }]}
              yFormat={(v) => compact(v)}
            />
          ) : (
            <Skeleton className="h-[240px]" />
          )}
        </Card>
        <Card className="lg:col-span-2">
          <CardTitle hint="Share of each monthly cohort of new customers who bought again N months later">
            Cohort retention
          </CardTitle>
          {data ? <Cohorts rows={data.cohort_retention} /> : <Skeleton className="h-[240px]" />}
        </Card>
      </div>
    </>
  );
}
