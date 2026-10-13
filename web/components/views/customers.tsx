"use client";

import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { Search, Shuffle } from "lucide-react";
import Link from "next/link";
import { useRouter, useSearchParams } from "next/navigation";
import { useDeferredValue, useState } from "react";

import { RiskBadge } from "@/components/product";
import { Button, Card, EmptyState, ErrorState, Input, PageHeader, Select, Skeleton } from "@/components/ui";
import { getJson } from "@/lib/api";
import { int, pct, shortId, units } from "@/lib/format";
import type { CustomerSample, RiskBand, Segment } from "@/lib/types";
import { cn } from "@/lib/utils";

const HEX = /^[0-9a-f]+$/;

function Lookup() {
  const router = useRouter();
  const [text, setText] = useState("");
  const prefix = useDeferredValue(text.trim().toLowerCase());
  const valid = prefix.length >= 4 && HEX.test(prefix);
  const matches = useQuery({
    queryKey: ["lookup", prefix],
    queryFn: () => getJson<string[]>("/api/customers/lookup", { prefix: prefix.slice(0, 64), limit: 8 }),
    enabled: valid,
  });

  return (
    <div className="relative">
      <form
        onSubmit={(event) => {
          event.preventDefault();
          const target = prefix.length === 64 ? prefix : matches.data?.[0];
          if (target) router.push(`/customers/${target}`);
        }}
      >
        <Search className="pointer-events-none absolute left-3 top-2.5 size-4 text-muted" aria-hidden />
        <Input
          value={text}
          onChange={(event) => setText(event.target.value)}
          placeholder="Customer ID (first 4+ characters)"
          className="pl-9 font-mono"
          aria-label="Customer ID"
          spellCheck={false}
        />
      </form>
      {prefix.length > 0 && !HEX.test(prefix) && (
        <p className="mt-1 text-xs text-bad">Customer IDs use only 0–9 and a–f.</p>
      )}
      {valid && (
        <div className="absolute inset-x-0 top-10 z-20 overflow-hidden rounded-lg border border-border bg-surface shadow-lg">
          {matches.isPending ? (
            <div className="p-3 text-xs text-muted">Searching…</div>
          ) : matches.data?.length ? (
            matches.data.map((id) => (
              <Link
                key={id}
                href={`/customers/${id}`}
                className="block truncate px-3 py-2 font-mono text-xs hover:bg-surface-2"
              >
                {id}
              </Link>
            ))
          ) : (
            <div className="p-3 text-xs text-muted">No customer ID starts with “{prefix}”.</div>
          )}
        </div>
      )}
    </div>
  );
}

export function CustomersView() {
  const router = useRouter();
  const params = useSearchParams();
  const segment = params.get("segment") ?? "";
  const risk = (params.get("risk") ?? "") as RiskBand | "";
  const seed = Number(params.get("seed") ?? 0);

  const setParam = (key: string, value: string) => {
    const next = new URLSearchParams(params);
    if (value) next.set(key, value);
    else next.delete(key);
    router.replace(`/customers?${next.toString()}`, { scroll: false });
  };

  const segments = useQuery({ queryKey: ["segments"], queryFn: () => getJson<Segment[]>("/api/segments") });
  const sample = useQuery({
    queryKey: ["sample", segment, risk, seed],
    queryFn: () =>
      getJson<CustomerSample[]>("/api/customers/sample", {
        segment_id: segment,
        risk_band: risk,
        seed,
        limit: 25,
      }),
    placeholderData: keepPreviousData,
  });

  return (
    <>
      <PageHeader
        title="Customer 360"
        description="Look up any of 1.37M customers, or browse a repeatable sample of active ones. Each profile shows inactivity risk and its drivers, predicted value, purchases, and recommendations."
      />
      <Card className="mb-4">
        <div className="grid gap-3 md:grid-cols-[1fr_auto_auto_auto]">
          <Lookup />
          <Select value={segment} onChange={(e) => setParam("segment", e.target.value)} aria-label="Segment">
            <option value="">All segments</option>
            {segments.data
              ?.filter((s) => s.segment_id >= 0)
              .map((s) => (
                <option key={s.segment_id} value={s.segment_id}>
                  {s.segment_name}
                </option>
              ))}
          </Select>
          <Select value={risk} onChange={(e) => setParam("risk", e.target.value)} aria-label="Risk band">
            <option value="">Any risk</option>
            <option value="high">High risk</option>
            <option value="medium">Medium risk</option>
            <option value="low">Low risk</option>
          </Select>
          <Button onClick={() => setParam("seed", String(seed + 1))}>
            <Shuffle className="size-4" aria-hidden /> New sample
          </Button>
        </div>
      </Card>

      {sample.error ? (
        <ErrorState error={sample.error} onRetry={() => sample.refetch()} />
      ) : (
        <Card className="overflow-hidden p-0 sm:p-0">
          <div className="overflow-x-auto">
            <table className="w-full min-w-[760px] text-sm">
              <thead className="whitespace-nowrap border-b border-border bg-surface-2 text-left text-xs text-muted">
                <tr>
                  <th className="px-4 py-2.5 font-medium">Customer</th>
                  <th className="px-4 py-2.5 font-medium">Segment</th>
                  <th className="px-4 py-2.5 font-medium">Risk</th>
                  <th className="px-4 py-2.5 text-right font-medium">P(inactive 60d)</th>
                  <th className="px-4 py-2.5 text-right font-medium">Value, next 180d</th>
                  <th className="px-4 py-2.5 text-right font-medium">Visits, 12 mo</th>
                  <th className="px-4 py-2.5 text-right font-medium">Last purchase</th>
                </tr>
              </thead>
              <tbody className={cn("whitespace-nowrap", sample.isPlaceholderData && "opacity-60")}>
                {sample.isPending
                  ? Array.from({ length: 8 }, (_, i) => (
                      <tr key={i}>
                        <td colSpan={7} className="px-4 py-2">
                          <Skeleton className="h-6" />
                        </td>
                      </tr>
                    ))
                  : sample.data?.map((c) => (
                      <tr
                        key={c.customer_id}
                        onClick={() => router.push(`/customers/${c.customer_id}`)}
                        className="cursor-pointer border-b border-border last:border-0 hover:bg-surface-2"
                      >
                        <td className="px-4 py-2.5">
                          <Link href={`/customers/${c.customer_id}`} className="font-mono text-xs text-accent">
                            {shortId(c.customer_id)}
                          </Link>
                        </td>
                        <td className="px-4 py-2.5 text-muted">{c.segment_name}</td>
                        <td className="px-4 py-2.5">
                          <RiskBadge band={c.risk_band} />
                        </td>
                        <td className="px-4 py-2.5 text-right tabular-nums">{pct(c.churn_probability)}</td>
                        <td className="px-4 py-2.5 text-right tabular-nums">{units(c.clv_180d)}</td>
                        <td className="px-4 py-2.5 text-right tabular-nums">{int(c.visits_365d)}</td>
                        <td className="px-4 py-2.5 text-right tabular-nums text-muted">
                          {int(c.recency_days)} days ago
                        </td>
                      </tr>
                    ))}
              </tbody>
            </table>
          </div>
          {sample.data?.length === 0 && (
            <div className="p-4">
              <EmptyState>No active customers match these filters.</EmptyState>
            </div>
          )}
        </Card>
      )}
      <p className="mt-3 text-xs text-muted">
        Customer IDs are anonymized hashes from the dataset. “Inactive” means no purchase in the next 60 days, a proxy
        for churn. Values are in normalized price units.
      </p>
    </>
  );
}
