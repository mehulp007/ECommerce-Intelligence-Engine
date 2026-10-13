"use client";

import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { Info } from "lucide-react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useMemo, useState } from "react";

import { BarsChart, TrendChart } from "@/components/charts";
import { RiskBadge } from "@/components/product";
import { Card, CardTitle, Chips, ErrorState, PageHeader, Skeleton, Spinner, Stat } from "@/components/ui";
import { getJson, postJson } from "@/lib/api";
import { compact, int, pct, shortId, units } from "@/lib/format";
import { useDebounced } from "@/lib/hooks";
import type { CampaignPlan, CampaignRequest, Segment } from "@/lib/types";

function Slider({
  label,
  value,
  min,
  max,
  step,
  display,
  onChange,
  hint,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  display: string;
  onChange: (value: number) => void;
  hint?: string;
}) {
  return (
    <label className="block">
      <div className="flex items-baseline justify-between text-sm">
        <span className="font-medium">{label}</span>
        <span className="tabular-nums text-accent">{display}</span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        className="mt-2 w-full"
      />
      {hint && <p className="mt-1 text-xs text-muted">{hint}</p>}
    </label>
  );
}

export function CampaignsView() {
  const params = useSearchParams();
  const [budget, setBudget] = useState(500);
  const [cost, setCost] = useState(0.005);
  const [uplift, setUplift] = useState(0.1);
  const [segmentNames, setSegmentNames] = useState<string[]>(() => {
    const segment = params.get("segment");
    return segment ? [segment] : [];
  });
  const [riskBands, setRiskBands] = useState<string[]>([]);

  const request: CampaignRequest = useDebounced({
    budget,
    cost_per_contact: cost,
    uplift,
    segment_names: segmentNames.length ? segmentNames : null,
    risk_bands: riskBands.length ? (riskBands as CampaignRequest["risk_bands"]) : null,
    top: 25,
  });

  const segments = useQuery({ queryKey: ["segments"], queryFn: () => getJson<Segment[]>("/api/segments") });
  const plan = useQuery({
    queryKey: ["campaign", request],
    queryFn: ({ signal }) => postJson<CampaignPlan>("/api/campaigns/plan", request, signal),
    placeholderData: keepPreviousData,
  });

  const best = useMemo(() => {
    const curve = plan.data?.curve ?? [];
    return curve.reduce<(typeof curve)[number] | null>(
      (top, point) => (!top || point.expected_net > top.expected_net ? point : top),
      null,
    );
  }, [plan.data]);

  const toggle = (list: string[], set: (next: string[]) => void) => (value: string) =>
    set(list.includes(value) ? list.filter((v) => v !== value) : [...list, value]);

  const p = plan.data;

  return (
    <>
      <PageHeader
        title="Campaign simulator"
        description="Choose a budget and an offer, and see who to contact. Customers are ranked by expected value saved: P(inactive) × predicted 180-day value × uplift − cost per contact."
      />
      <div className="grid gap-4 lg:grid-cols-[320px_1fr]">
        <Card className="flex h-fit flex-col gap-6 lg:sticky lg:top-6">
          <Slider
            label="Budget"
            value={budget}
            min={0}
            max={2000}
            step={10}
            display={`${int(budget)} units`}
            onChange={setBudget}
          />
          <Slider
            label="Cost per contact"
            value={cost}
            min={0.001}
            max={0.05}
            step={0.001}
            display={`${cost.toFixed(3)} units`}
            onChange={setCost}
            hint="For scale, the median item price is about 0.025 units."
          />
          <Slider
            label="Offer uplift"
            value={uplift}
            min={0.01}
            max={0.5}
            step={0.01}
            display={pct(uplift, 0)}
            onChange={setUplift}
            hint="Assumed share of at-risk value the offer keeps. The dataset has no experiment to measure it."
          />
          <div>
            <div className="mb-2 text-sm font-medium">Risk bands</div>
            <Chips
              options={[
                { value: "high", label: "High" },
                { value: "medium", label: "Medium" },
                { value: "low", label: "Low" },
              ]}
              selected={riskBands}
              onToggle={toggle(riskBands, setRiskBands)}
            />
          </div>
          <div>
            <div className="mb-2 text-sm font-medium">Segments</div>
            <Chips
              options={
                segments.data
                  ?.filter((s) => s.segment_id >= 0)
                  .map((s) => ({ value: s.segment_name, label: s.segment_name })) ?? []
              }
              selected={segmentNames}
              onToggle={toggle(segmentNames, setSegmentNames)}
            />
            <p className="mt-2 text-xs text-muted">None selected means all active customers.</p>
          </div>
        </Card>

        <div className="flex min-w-0 flex-col gap-4">
          {plan.error ? (
            <ErrorState error={plan.error} onRetry={() => plan.refetch()} />
          ) : (
            <>
              <div className="grid grid-cols-2 gap-3 xl:grid-cols-4">
                {p ? (
                  <>
                    <Stat
                      label="Customers contacted"
                      value={compact(p.contacted)}
                      hint={`of ${compact(p.eligible_customers)} eligible`}
                    />
                    <Stat label="Spend" value={units(p.spend)} hint="Price units" />
                    <Stat label="Expected value saved" value={units(p.expected_value_saved)} hint="Price units" />
                    <Stat
                      label="Expected net / ROI"
                      value={units(p.expected_net)}
                      tone={p.expected_net >= 0 ? "good" : "bad"}
                      hint={`ROI ${pct(p.roi, 0)}`}
                    />
                  </>
                ) : (
                  Array.from({ length: 4 }, (_, i) => <Skeleton key={i} className="h-[92px]" />)
                )}
              </div>

              {p && (
                <div className="flex items-start gap-2 rounded-xl border border-border bg-surface p-3 text-sm">
                  {plan.isFetching ? <Spinner className="mt-0.5 text-accent" /> : <Info className="mt-0.5 size-4 shrink-0 text-accent" />}
                  <p>
                    {int(p.profitable_customers)} customers are worth contacting at this cost (their expected saved value
                    exceeds it).{" "}
                    {p.contacted > p.profitable_customers
                      ? `This budget reaches ${int(p.contacted - p.profitable_customers)} more, who each cost more than they are expected to save.`
                      : p.contacted < p.profitable_customers
                        ? `A larger budget would reach ${int(p.profitable_customers - p.contacted)} more of them.`
                        : "This budget reaches exactly those customers."}
                    {best && ` Net value peaks at about ${compact(best.contacted)} contacts (${units(best.spend)} units).`}
                  </p>
                </div>
              )}

              <Card>
                <CardTitle hint="Contacting the highest expected value first; the curve covers every eligible customer">
                  Return as the campaign grows
                </CardTitle>
                {p ? (
                  <TrendChart
                    data={p.curve}
                    x="contacted"
                    area
                    height={260}
                    series={[
                      { key: "expected_value_saved", label: "Expected value saved" },
                      { key: "spend", label: "Spend", color: "var(--chart-3)" },
                      { key: "expected_net", label: "Expected net", color: "var(--chart-2)" },
                    ]}
                    xFormat={(v) => compact(Number(v))}
                    yFormat={(v) => units(v)}
                  />
                ) : (
                  <Skeleton className="h-[260px]" />
                )}
              </Card>

              <div className="grid gap-4 2xl:grid-cols-[1fr_1.4fr]">
                <Card>
                  <CardTitle>Contacted by segment</CardTitle>
                  {p ? (
                    Object.keys(p.contacted_by_segment).length ? (
                      <BarsChart
                        data={Object.entries(p.contacted_by_segment).map(([segment, customers]) => ({
                          segment,
                          customers,
                        }))}
                        x="segment"
                        horizontal
                        categoryWidth={170}
                        height={Math.max(160, Object.keys(p.contacted_by_segment).length * 40)}
                        series={[{ key: "customers", label: "Customers" }]}
                        yFormat={(v) => compact(v)}
                      />
                    ) : (
                      <p className="text-sm text-muted">Nobody is contacted with this budget.</p>
                    )
                  ) : (
                    <Skeleton className="h-40" />
                  )}
                </Card>
                <Card className="overflow-hidden">
                  <CardTitle hint="The first 25 customers to contact">Top targets</CardTitle>
                  <div className="-mx-4 overflow-x-auto sm:-mx-5">
                    <table className="w-full min-w-[520px] text-sm">
                      <thead className="text-left text-xs text-muted">
                        <tr className="border-b border-border">
                          <th className="px-4 py-2 font-medium sm:px-5">Customer</th>
                          <th className="px-2 py-2 font-medium">Risk</th>
                          <th className="px-2 py-2 text-right font-medium">P(inactive)</th>
                          <th className="px-2 py-2 text-right font-medium">Value 180d</th>
                          <th className="px-4 py-2 text-right font-medium sm:px-5">Expected saved</th>
                        </tr>
                      </thead>
                      <tbody>
                        {p?.top_customers.map((c) => (
                          <tr key={c.customer_id} className="border-b border-border last:border-0">
                            <td className="px-4 py-2 sm:px-5">
                              <Link href={`/customers/${c.customer_id}`} className="font-mono text-xs text-accent">
                                {shortId(c.customer_id)}
                              </Link>
                            </td>
                            <td className="px-2 py-2">
                              <RiskBadge band={c.risk_band} />
                            </td>
                            <td className="px-2 py-2 text-right tabular-nums">{pct(c.churn_probability, 0)}</td>
                            <td className="px-2 py-2 text-right tabular-nums">{units(c.clv_180d)}</td>
                            <td className="px-4 py-2 text-right tabular-nums sm:px-5">{units(c.expected_value_saved)}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </Card>
              </div>
            </>
          )}
        </div>
      </div>
    </>
  );
}
