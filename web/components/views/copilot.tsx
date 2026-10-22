"use client";

import { useQuery } from "@tanstack/react-query";
import {
  AlertTriangle,
  ArrowUp,
  BarChart3,
  Database,
  Megaphone,
  RotateCcw,
  Search,
  Shirt,
  Sparkles,
  Square,
  User,
  Wrench,
} from "lucide-react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useEffect, useRef, useState, type ReactNode } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { BarsChart, TrendChart } from "@/components/charts";
import { ProductCard, ProductGrid, RiskBadge } from "@/components/product";
import { Badge, Button, Card, ErrorState, PageHeader, Spinner } from "@/components/ui";
import { errorFrom, getJson } from "@/lib/api";
import { compact, int, pct, shortId, units } from "@/lib/format";
import { featureLabel } from "@/lib/labels";
import type { CopilotEvent, CopilotStatus } from "@/lib/types";
import { cn } from "@/lib/utils";

type Done = Extract<CopilotEvent, { type: "done" }>;

type Message =
  | { role: "user"; text: string }
  | {
      role: "assistant";
      events: CopilotEvent[];
      answer?: string;
      done?: Done;
      state: "streaming" | "done" | "stopped" | "failed";
    };

const SUGGESTIONS = [
  "Which product groups grew most in the last 4 weeks compared with the same weeks a year earlier? Show a chart.",
  "Find the highest-value customer at high inactivity risk, explain why, and draft a short win-back email with two recommended products.",
  "With a budget of 300 units, 0.004 per contact and a 10% uplift, which segments should a retention campaign target?",
  "Show weekly online share from February to June 2020 as a line chart and describe what happened.",
];

/** Parse a Server-Sent Events stream into JSON events. */
async function* readEvents(response: Response): AsyncGenerator<CopilotEvent> {
  const reader = response.body!.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += value;
    let end: number;
    while ((end = buffer.indexOf("\n\n")) >= 0) {
      const chunk = buffer.slice(0, end);
      buffer = buffer.slice(end + 2);
      const data = chunk
        .split("\n")
        .filter((line) => line.startsWith("data:"))
        .map((line) => line.slice(5).trimStart())
        .join("\n");
      if (data) yield JSON.parse(data) as CopilotEvent;
    }
  }
}

/** Dates arrive as midnight timestamps; show them as plain dates. */
const plainDate = (value: string) => value.replace(/^(\d{4}-\d{2}-\d{2})[T ]00:00:00(\.0+)?$/, "$1");

function cell(value: unknown): string {
  if (value === null || value === undefined) return "—";
  if (typeof value === "string") return plainDate(value);
  if (typeof value === "number") {
    if (Number.isInteger(value)) return int(value);
    return Math.abs(value) >= 100 ? value.toFixed(1) : String(Number(value.toPrecision(4)));
  }
  return String(value);
}

function TableResult({ event }: { event: Extract<CopilotEvent, { type: "table" }> }) {
  const [all, setAll] = useState(false);
  const rows = all ? event.rows : event.rows.slice(0, 8);
  return (
    <div className="overflow-hidden rounded-lg border border-border bg-surface">
      <div className="flex flex-wrap items-center justify-between gap-2 border-b border-border px-3 py-2 text-xs">
        <span className="font-medium">{event.purpose || "Query result"}</span>
        <span className="text-muted">
          {int(event.row_count)} row{event.row_count === 1 ? "" : "s"}
          {event.truncated ? " (capped)" : ""} · {event.seconds.toFixed(2)}s
        </span>
      </div>
      <details className="border-b border-border px-3 py-1.5 text-xs">
        <summary className="cursor-pointer text-muted">SQL</summary>
        <pre className="mt-1 overflow-x-auto whitespace-pre-wrap font-mono text-[11px] leading-relaxed">{event.sql}</pre>
      </details>
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead className="bg-surface-2 text-left text-muted">
            <tr>
              {event.columns.map((column) => (
                <th key={column} className="whitespace-nowrap px-3 py-1.5 font-medium">
                  {column}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i} className="border-t border-border">
                {row.map((value, j) => (
                  <td
                    key={j}
                    className={cn(
                      "whitespace-nowrap px-3 py-1.5",
                      typeof value === "number" && "text-right tabular-nums",
                      typeof value === "string" && value.length === 64 && "font-mono",
                    )}
                  >
                    {typeof value === "string" && /^[0-9a-f]{64}$/.test(value) ? (
                      <Link href={`/customers/${value}`} className="text-accent">
                        {shortId(value)}
                      </Link>
                    ) : (
                      cell(value)
                    )}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {event.rows.length > 8 && (
        <button onClick={() => setAll(!all)} className="w-full border-t border-border py-1.5 text-xs text-accent hover:bg-surface-2">
          {all ? "Show fewer" : `Show all ${event.rows.length} rows`}
        </button>
      )}
    </div>
  );
}

function ChartResult({ event }: { event: Extract<CopilotEvent, { type: "chart" }> }) {
  const data = event.rows.map((row) => {
    const point: Record<string, unknown> = { [event.x]: plainDate(String(row[0])) };
    event.y.forEach((key, i) => (point[key] = Number(row[i + 1])));
    return point;
  });
  const series = event.y.map((key) => ({ key, label: key.replace(/_/g, " ") }));
  const format = (v: number) => (Math.abs(v) >= 1000 ? compact(v) : String(Number(v.toPrecision(3))));
  return (
    <div className="rounded-lg border border-border bg-surface p-3">
      <div className="mb-2 text-xs font-medium">{event.title}</div>
      {event.kind === "line" ? (
        <TrendChart data={data} x={event.x} series={series} height={240} yFormat={format} />
      ) : (
        <BarsChart
          data={data}
          x={event.x}
          series={series}
          height={Math.max(220, data.length > 8 ? data.length * 26 : 240)}
          horizontal={data.length > 8}
          categoryWidth={140}
          yFormat={format}
          zeroLine
        />
      )}
    </div>
  );
}

function CustomerResult({ event }: { event: Extract<CopilotEvent, { type: "customer" }> }) {
  const c = event.customer;
  return (
    <div className="rounded-lg border border-border bg-surface p-3">
      <div className="flex flex-wrap items-center gap-2">
        <Link href={`/customers/${c.customer_id}`} className="font-mono text-xs font-medium text-accent">
          {shortId(c.customer_id)}
        </Link>
        <RiskBadge band={c.risk_band} />
        <Badge>{c.segment_name}</Badge>
      </div>
      <div className="mt-2 grid grid-cols-2 gap-2 text-xs sm:grid-cols-4">
        <div>
          <div className="text-muted">P(inactive 60d)</div>
          <div className="font-medium">{c.churn_probability === null ? "—" : pct(c.churn_probability)}</div>
        </div>
        <div>
          <div className="text-muted">Value 180d</div>
          <div className="font-medium">{units(c.clv_180d)}</div>
        </div>
        <div>
          <div className="text-muted">Visits, 12 mo</div>
          <div className="font-medium">{int(c.visits_365d)}</div>
        </div>
        <div>
          <div className="text-muted">Last purchase</div>
          <div className="font-medium">{c.recency_days === null ? "—" : `${int(c.recency_days)} days ago`}</div>
        </div>
      </div>
      {c.drivers.length > 0 && (
        <div className="mt-2 text-xs text-muted">
          Drivers:{" "}
          {c.drivers
            .map((d) => `${featureLabel(d.feature)} (${d.log_odds > 0 ? "raises" : "lowers"} risk)`)
            .join(", ")}
        </div>
      )}
    </div>
  );
}

const TOOL_TEXT: Record<string, (args: Record<string, unknown>) => string> = {
  run_sql: (a) => `Queried the data${a.purpose ? `: ${a.purpose}` : ""}`,
  create_chart: (a) => `Drew a chart${a.title ? `: ${a.title}` : ""}`,
  get_customer: (a) => `Opened customer ${shortId(String(a.customer_id ?? ""))}`,
  search_products: (a) => `Searched products for “${a.query}”`,
  similar_products: (a) => `Found products similar to ${a.article_id}`,
  plan_campaign: () => "Planned a campaign",
};

const TOOL_ICON: Record<string, typeof Wrench> = {
  run_sql: Database,
  create_chart: BarChart3,
  get_customer: User,
  search_products: Search,
  similar_products: Shirt,
  plan_campaign: Megaphone,
};

function Step({ icon: Icon, children, tone }: { icon: typeof Wrench; children: ReactNode; tone?: "bad" | "warn" }) {
  return (
    <div
      className={cn(
        "flex items-start gap-2 text-xs",
        tone === "bad" ? "text-bad" : tone === "warn" ? "text-warn" : "text-muted",
      )}
    >
      <Icon className="mt-0.5 size-3.5 shrink-0" aria-hidden />
      <span>{children}</span>
    </div>
  );
}

function EventView({ event }: { event: CopilotEvent }) {
  switch (event.type) {
    case "tool_call":
      return <Step icon={TOOL_ICON[event.name] ?? Wrench}>{(TOOL_TEXT[event.name] ?? (() => event.name))(event.args)}</Step>;
    case "thinking":
      return <p className="text-xs italic text-muted">{event.text}</p>;
    case "tool_error":
      return (
        <Step icon={AlertTriangle} tone="warn">
          {event.name} failed: {event.message}. The model was told and can correct it.
        </Step>
      );
    case "restart":
      return (
        <Step icon={RotateCcw} tone="warn">
          {event.message}
        </Step>
      );
    case "table":
      return <TableResult event={event} />;
    case "chart":
      return <ChartResult event={event} />;
    case "customer":
      return <CustomerResult event={event} />;
    case "products":
      return (
        <div>
          <div className="mb-2 text-xs font-medium">{event.title}</div>
          <ProductGrid dense>
            {event.items.slice(0, 6).map((hit) => (
              <ProductCard key={hit.article.article_id} article={hit.article} />
            ))}
          </ProductGrid>
        </div>
      );
    case "campaign":
      return (
        <div className="grid grid-cols-2 gap-2 rounded-lg border border-border bg-surface p-3 text-xs sm:grid-cols-4">
          {[
            ["Contacted", compact(event.plan.contacted)],
            ["Spend", units(event.plan.spend)],
            ["Expected saved", units(event.plan.expected_value_saved)],
            ["ROI", pct(event.plan.roi, 0)],
          ].map(([label, value]) => (
            <div key={label}>
              <div className="text-muted">{label}</div>
              <div className="font-medium">{value}</div>
            </div>
          ))}
        </div>
      );
    case "error":
      return (
        <Step icon={AlertTriangle} tone="bad">
          {event.message}
        </Step>
      );
    default:
      return null;
  }
}

function AssistantMessage({ message }: { message: Extract<Message, { role: "assistant" }> }) {
  const shown = message.events.filter((e) => e.type !== "answer" && e.type !== "done");
  return (
    <div className="flex gap-3">
      <span className="mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-lg bg-accent text-accent-fg">
        <Sparkles className="size-3.5" aria-hidden />
      </span>
      <div className="flex min-w-0 flex-1 flex-col gap-2.5">
        {shown.map((event, i) => (
          <EventView key={i} event={event} />
        ))}
        {message.state === "streaming" && !message.answer && (
          <div className="flex items-center gap-2 text-xs text-muted">
            <Spinner /> {shown.length ? "Working…" : "Thinking…"}
          </div>
        )}
        {message.answer && (
          <div className="prose-answer text-sm leading-relaxed">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.answer}</ReactMarkdown>
          </div>
        )}
        {message.state === "stopped" && <p className="text-xs text-muted">Stopped.</p>}
        {message.done && (
          <div className="text-[11px] text-muted">
            {message.done.model} · {message.done.seconds.toFixed(1)}s
            {message.done.usage.input_tokens
              ? ` · ${compact((message.done.usage.input_tokens ?? 0) + (message.done.usage.output_tokens ?? 0) + (message.done.usage.thinking_tokens ?? 0))} tokens`
              : ""}
          </div>
        )}
      </div>
    </div>
  );
}

export function CopilotView() {
  const params = useSearchParams();
  const [input, setInput] = useState(params.get("q") ?? "");
  const [messages, setMessages] = useState<Message[]>([]);
  const [busy, setBusy] = useState(false);
  const abort = useRef<AbortController | null>(null);
  const bottom = useRef<HTMLDivElement>(null);
  const status = useQuery({
    queryKey: ["copilot-status"],
    queryFn: () => getJson<CopilotStatus>("/api/copilot/status"),
  });

  useEffect(() => {
    bottom.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages]);
  useEffect(() => () => abort.current?.abort(), []);

  const updateLast = (change: (m: Extract<Message, { role: "assistant" }>) => Message) =>
    setMessages((all) => {
      const last = all[all.length - 1];
      return last?.role === "assistant" ? [...all.slice(0, -1), change(last)] : all;
    });

  async function ask(question: string) {
    const text = question.trim();
    if (!text || busy) return;
    // Earlier turns go back as plain text; the server caps history at 40 messages.
    const history = messages
      .flatMap<{ role: "user" | "assistant"; content: string }>((m) =>
        m.role === "user"
          ? [{ role: "user" as const, content: m.text }]
          : m.answer
            ? [{ role: "assistant" as const, content: m.answer.slice(0, 8000) }]
            : [],
      )
      .slice(-30);
    setMessages((all) => [...all, { role: "user", text }, { role: "assistant", events: [], state: "streaming" }]);
    setInput("");
    setBusy(true);
    const controller = new AbortController();
    abort.current = controller;
    try {
      const response = await fetch("/api/copilot/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json", Accept: "text/event-stream" },
        body: JSON.stringify({ messages: [...history, { role: "user", content: text }] }),
        signal: controller.signal,
      });
      if (!response.ok) throw await errorFrom(response);
      for await (const event of readEvents(response)) {
        updateLast((m) => ({
          ...m,
          events: [...m.events, event],
          answer: event.type === "answer" ? event.text : m.answer,
          done: event.type === "done" ? event : m.done,
        }));
      }
      updateLast((m) => ({ ...m, state: "done" }));
    } catch (error) {
      if (controller.signal.aborted) {
        updateLast((m) => ({ ...m, state: "stopped" }));
      } else {
        const message = error instanceof Error ? error.message : String(error);
        updateLast((m) => ({ ...m, state: "failed", events: [...m.events, { type: "error", message }] }));
      }
    } finally {
      setBusy(false);
      abort.current = null;
    }
  }

  if (status.error) return <ErrorState error={status.error} onRetry={() => status.refetch()} />;

  if (status.data && !status.data.configured) {
    return (
      <>
        <PageHeader title="Analyst copilot" />
        <Card className="max-w-2xl">
          <h2 className="font-semibold">Add a Gemini API key to turn the copilot on</h2>
          <ol className="mt-3 list-decimal space-y-1.5 pl-5 text-sm text-muted">
            <li>Create a key in Google AI Studio.</li>
            <li>
              Open the <code className="font-mono">.env</code> file in the project folder and put it after{" "}
              <code className="font-mono">GEMINI_API_KEY=</code>.
            </li>
            <li>Restart the API. The key stays on the server and is never sent to the browser.</li>
          </ol>
        </Card>
      </>
    );
  }

  return (
    <div className="flex min-h-[calc(100vh-7rem)] flex-col lg:min-h-[calc(100vh-4rem)]">
      <PageHeader
        title="Analyst copilot"
        description={
          <>
            Ask business questions in plain English. Gemini answers by calling tools: read-only SQL over{" "}
            {status.data?.tables.length ?? 5} tables, customer profiles, product search, and the campaign planner. Every
            number comes from a tool result shown below it.
          </>
        }
        actions={
          messages.length > 0 && (
            <Button size="sm" onClick={() => setMessages([])} disabled={busy}>
              <RotateCcw className="size-3.5" /> New chat
            </Button>
          )
        }
      />

      <div className="flex flex-1 flex-col gap-6">
        {messages.length === 0 ? (
          <div className="flex flex-col gap-4">
            <div className="flex items-center gap-3 text-sm text-muted">
              <span className="flex size-9 items-center justify-center rounded-xl border border-accent/30 bg-accent-soft text-accent">
                <Sparkles className="size-4" aria-hidden />
              </span>
              Start with one of these, or ask your own question below.
            </div>
            <div className="grid gap-3 sm:grid-cols-2">
              {SUGGESTIONS.map((suggestion) => (
                <button
                  key={suggestion}
                  onClick={() => ask(suggestion)}
                  className="group flex items-start justify-between gap-3 rounded-2xl border border-border bg-surface p-4 text-left text-sm leading-relaxed transition-colors hover:border-accent/50 hover:bg-surface-2"
                >
                  {suggestion}
                  <ArrowUp
                    className="mt-0.5 size-4 shrink-0 rotate-45 text-muted transition-colors group-hover:text-accent"
                    aria-hidden
                  />
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((message, i) =>
            message.role === "user" ? (
              <div key={i} className="ml-auto max-w-[85%] rounded-2xl rounded-br-md bg-accent-soft px-4 py-2.5 text-sm">
                {message.text}
              </div>
            ) : (
              <AssistantMessage key={i} message={message} />
            ),
          )
        )}
        <div ref={bottom} />
      </div>

      <form
        onSubmit={(event) => {
          event.preventDefault();
          ask(input);
        }}
        className="sticky bottom-0 mt-6 bg-bg pb-2 pt-2"
      >
        <div className="flex items-end gap-2 rounded-xl border border-border bg-surface p-2 focus-within:border-accent">
          <textarea
            value={input}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === "Enter" && !event.shiftKey) {
                event.preventDefault();
                ask(input);
              }
            }}
            rows={Math.min(5, Math.max(1, input.split("\n").length))}
            maxLength={8000}
            placeholder="Ask about sales, customers, products or campaigns…"
            className="flex-1 resize-none bg-transparent px-2 py-1.5 text-sm outline-none placeholder:text-muted"
            aria-label="Question"
          />
          {busy ? (
            <Button type="button" size="sm" onClick={() => abort.current?.abort()} aria-label="Stop">
              <Square className="size-3.5" /> Stop
            </Button>
          ) : (
            <Button type="submit" variant="primary" size="sm" disabled={!input.trim()} aria-label="Send">
              <ArrowUp className="size-4" />
            </Button>
          )}
        </div>
        <p className="mt-1.5 px-1 text-[11px] text-muted">
          {status.data ? `${status.data.model} · ` : ""}A free Gemini key allows about 20 requests per model per day, and
          each question uses 3 to 5. Check important answers against the tables shown.
        </p>
      </form>
    </div>
  );
}
