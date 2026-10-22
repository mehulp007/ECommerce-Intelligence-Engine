"use client";

import { useQuery } from "@tanstack/react-query";
import { ArrowDown, ArrowRight, BarChart3, Database, Orbit, Sparkles } from "lucide-react";
import dynamic from "next/dynamic";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { useEffect, useMemo, useRef, useState, useSyncExternalStore, type ReactNode } from "react";

import { TrendChart } from "@/components/charts";
import { ProductImage } from "@/components/product";
import { buttonClasses } from "@/components/ui";
import { getJson, postJson } from "@/lib/api";
import { compact, date, fixed, int, month, pct, units } from "@/lib/format";
import { loadGalaxy } from "@/lib/galaxy";
import { useDebounced } from "@/lib/hooks";
import {
  loadStoryCustomers,
  RISK_COLOURS,
  SCENES,
  sceneProgress,
  SOURCE_COLOURS,
  storyScroll,
} from "@/lib/story";
import transcript from "@/lib/story-copilot.json";
import type { CampaignPlan, ModelReports, Overview, SearchHit } from "@/lib/types";
import { cn } from "@/lib/utils";

const StoryScene = dynamic(() => import("@/components/story/story-scene"), { ssr: false });

const QUERY = "red floral summer dress";
const COST = 0.005;
const UPLIFT = 0.1;
const SOURCE_NAMES: Record<string, string> = {
  repurchase: "Buy again",
  popular: "Best sellers",
  age_popular: "Best sellers by age",
  cooc: "Bought together",
  als: "Similar shoppers (ALS)",
  style: "Similar style (FashionCLIP)",
};

const noSubscribe = () => () => {};
const isSmall = () => window.innerWidth < 768;
function subscribeMotion(listener: () => void) {
  const media = matchMedia("(prefers-reduced-motion: reduce)");
  media.addEventListener("change", listener);
  return () => media.removeEventListener("change", listener);
}
const reducedMotion = () => matchMedia("(prefers-reduced-motion: reduce)").matches;

function Card({ children, wide = false, className }: { children: ReactNode; wide?: boolean; className?: string }) {
  return (
    <div
      className={cn(
        "w-full rounded-3xl border border-white/10 bg-[#0b0d14]/60 p-6 shadow-[0_30px_80px_-30px_rgba(0,0,0,0.9)] backdrop-blur-md sm:p-7",
        wide ? "max-w-2xl" : "max-w-md",
        className,
      )}
    >
      {children}
    </div>
  );
}

function Eyebrow({ index, children }: { index: number; children: ReactNode }) {
  return (
    <div className="mb-3 flex items-center gap-2 font-mono text-[11px] uppercase tracking-[0.14em] text-accent">
      <span className="text-muted">{String(index).padStart(2, "0")}</span>
      {children}
    </div>
  );
}

function Headline({ children, large = false }: { children: ReactNode; large?: boolean }) {
  return (
    <h2
      className={cn(
        "font-[family-name:var(--font-serif)] leading-[1.05] tracking-tight text-fg",
        large ? "text-5xl sm:text-7xl" : "text-3xl sm:text-4xl",
      )}
    >
      {children}
    </h2>
  );
}

function Figure({ value, label }: { value: ReactNode; label: string }) {
  return (
    <div>
      <div className="font-mono text-2xl font-medium tracking-tight text-fg">{value}</div>
      <div className="mt-0.5 text-xs text-muted">{label}</div>
    </div>
  );
}

function Swatch({ colour, children }: { colour: string; children: ReactNode }) {
  return (
    <span className="flex items-center gap-1.5 text-xs text-muted">
      <span className="size-2 rounded-full" style={{ background: colour }} />
      {children}
    </span>
  );
}

/** The recorded copilot run, revealed step by step as the reader scrolls. */
function CopilotReplay({ t }: { t: number }) {
  const typed = transcript.question.slice(0, Math.round(Math.min(1, t / 0.22) * transcript.question.length));
  const chart = transcript.chart.rows.map(([month, value]) => ({ month, value }));
  const summary = transcript.answer
    .split("\n\n")
    .find((part) => !part.startsWith("#") && part.length > 80)
    ?.replace(/\*\*/g, "");
  return (
    <div className="font-mono text-[12px] leading-relaxed">
      <div className="text-fg">
        <span className="text-accent">you › </span>
        {typed}
        {t < 0.22 && <span className="animate-pulse">▍</span>}
      </div>
      <div className={cn("mt-3 hidden space-y-1 text-muted transition-opacity duration-500 sm:block", t > 0.26 ? "opacity-100" : "opacity-0")}>
        {transcript.steps.map((step, i) => (
          <div key={i} className="flex gap-2">
            <Database className="mt-0.5 size-3.5 shrink-0 text-accent" aria-hidden />
            <span>
              {step.tool}: {step.purpose}
            </span>
          </div>
        ))}
      </div>
      <pre
        className={cn(
          "mt-3 hidden max-h-28 overflow-hidden rounded-xl sm:block border border-white/10 bg-white/[0.03] p-3 text-[11px] text-muted transition-opacity duration-500 [mask-image:linear-gradient(180deg,black_70%,transparent)]",
          t > 0.36 ? "opacity-100" : "opacity-0",
        )}
      >
        {transcript.sql}
      </pre>
      <div className={cn("mt-3 transition-opacity duration-500", t > 0.5 ? "opacity-100" : "opacity-0")}>
        <div className="mb-1 font-sans text-xs text-muted">{transcript.chart.title}</div>
        <TrendChart
          data={chart}
          x="month"
          height={130}
          series={[{ key: "value", label: "Online share (%)" }]}
          yFormat={(v) => `${Math.round(v)}%`}
        />
      </div>
      <p className={cn("mt-3 font-sans text-sm leading-relaxed text-fg transition-opacity duration-500", t > 0.66 ? "opacity-100" : "opacity-0")}>
        {summary}
      </p>
    </div>
  );
}

export function StoryView() {
  const params = useSearchParams();
  const capture = params.get("capture") === "1";
  const small = useSyncExternalStore(noSubscribe, isSmall, () => false);
  const calm = useSyncExternalStore(subscribeMotion, reducedMotion, () => false);
  const sections = useRef<(HTMLElement | null)[]>([]);
  const [position, setPosition] = useState(0);
  const [budget, setBudget] = useState(300);

  const overview = useQuery({ queryKey: ["overview"], queryFn: () => getJson<Overview>("/api/overview") });
  const models = useQuery({ queryKey: ["models"], queryFn: () => getJson<ModelReports>("/api/models") });
  const galaxy = useQuery({
    queryKey: ["galaxy", small ? 4 : 2],
    queryFn: () => loadGalaxy(small ? 4 : 2),
    enabled: !calm,
    staleTime: Infinity,
  });
  const customers = useQuery({
    queryKey: ["story-customers", small ? 2 : 1],
    queryFn: () => loadStoryCustomers(small ? 2 : 1),
    enabled: !calm,
    staleTime: Infinity,
  });
  const search = useQuery({
    queryKey: ["search", QUERY, 50],
    queryFn: () => getJson<SearchHit[]>("/api/search", { q: QUERY, k: 50 }),
  });
  const plannedBudget = useDebounced(budget, 200);
  const plan = useQuery({
    queryKey: ["story-plan", plannedBudget],
    queryFn: () =>
      postJson<CampaignPlan>("/api/campaigns/plan", {
        budget: plannedBudget,
        cost_per_contact: COST,
        uplift: UPLIFT,
        top: 0,
      }),
  });

  // Track the reader: 0 at the top, SCENES at the end; each pinned section adds 1.
  useEffect(() => {
    let frame = 0;
    const update = () => {
      frame = 0;
      const height = window.innerHeight;
      const total = sections.current.reduce((sum, el) => {
        if (!el) return sum;
        const rect = el.getBoundingClientRect();
        const room = Math.max(1, rect.height - height);
        return sum + Math.min(1, Math.max(0, -rect.top / room));
      }, 0);
      storyScroll.position = total;
      setPosition(total);
    };
    const onScroll = () => {
      if (!frame) frame = requestAnimationFrame(update);
    };
    update();
    window.addEventListener("scroll", onScroll, { passive: true });
    window.addEventListener("resize", onScroll);
    return () => {
      window.removeEventListener("scroll", onScroll);
      window.removeEventListener("resize", onScroll);
      cancelAnimationFrame(frame);
    };
  }, []);

  const searchMatches = useMemo(() => {
    if (!galaxy.data || !search.data) return [];
    return search.data.map((hit) => galaxy.data.index.get(hit.article.article_id)).filter((i): i is number => i !== undefined);
  }, [galaxy.data, search.data]);

  // The sampled customers the planner would contact, by expected value saved.
  const ranked = useMemo(() => {
    const c = customers.data;
    if (!c) return [];
    return Array.from({ length: c.count }, (_, i) => i).sort((a, b) => c.churn[b] * c.clv[b] - c.churn[a] * c.clv[a]);
  }, [customers.data]);
  const targeted = useMemo(() => {
    const c = customers.data;
    if (!c || !plan.data) return [];
    const share = plan.data.contacted / c.meta.population;
    return ranked.slice(0, Math.round(share * c.count));
  }, [customers.data, plan.data, ranked]);

  // Share of the planner's picks (in the sample) that are high risk.
  const highShare = useMemo(() => {
    const c = customers.data;
    if (!c || !targeted.length) return null;
    return targeted.filter((i) => c.risk[i] === 2).length / targeted.length;
  }, [customers.data, targeted]);

  const scene = Math.min(SCENES - 1, Math.floor(position));
  const local = (i: number) => sceneProgress(position, i);
  const shown = (i: number) => calm || scene === i;
  const r = models.data?.recommender_metrics;
  const c = models.data?.customer_metrics;
  const kpis = overview.data?.kpis;
  const sourceSizes = r?.candidate_sizes;
  const typed = QUERY.slice(0, Math.round(Math.min(1, local(3) / 0.4) * QUERY.length));
  const closure = overview.data?.weekly.filter((w) => w.online_share >= 0.999) ?? [];

  const scrollTo = (i: number) => sections.current[i]?.scrollIntoView({ behavior: calm ? "auto" : "smooth" });

  const sectionClass = calm ? "relative py-16" : "relative h-[230vh]";
  const stickyClass = calm
    ? "flex px-5 sm:px-10 lg:px-16"
    : "sticky top-0 flex h-dvh items-end px-4 pb-6 pt-24 sm:items-center sm:px-10 sm:pb-0 lg:px-16";
  const reveal = (i: number) =>
    cn("transition-[opacity,transform] duration-700 ease-out", shown(i) ? "translate-y-0 opacity-100" : "pointer-events-none translate-y-6 opacity-0");

  return (
    <div className="relative min-h-dvh bg-bg text-fg">
      {!calm && (
        <div className="fixed inset-0" aria-hidden>
          <div className="absolute inset-0" style={{ background: "radial-gradient(ellipse at 50% 35%, #151b36 0%, #0b0d14 62%)" }} />
          {/* A soft scrim keeps the text readable over bright points. */}
          <div
            className="pointer-events-none absolute inset-0 z-10 hidden sm:block"
            style={{ background: "linear-gradient(90deg, rgba(11,13,20,0.84) 0%, rgba(11,13,20,0.36) 40%, transparent 62%)" }}
          />
          <div
            className="pointer-events-none absolute inset-0 z-10 sm:hidden"
            style={{ background: "linear-gradient(0deg, rgba(11,13,20,0.9) 0%, rgba(11,13,20,0.55) 55%, rgba(11,13,20,0.2) 100%)" }}
          />
          <div className="absolute inset-0">
            <StoryScene
              galaxy={galaxy.data}
              overview={overview.data}
              sources={sourceSizes ? Object.values(sourceSizes) : undefined}
              customers={customers.data}
              searchMatches={searchMatches}
              targeted={targeted}
              calm={calm}
            />
          </div>
        </div>
      )}

      <header className="fixed inset-x-0 top-0 z-30 border-b border-white/5 bg-bg/40 backdrop-blur-md">
        <div className="flex items-center justify-between px-4 py-3 sm:px-8">
          <Link href="/" className="flex items-center gap-2.5">
            <span className="flex size-7 items-center justify-center rounded-lg bg-accent text-accent-fg">
              <BarChart3 className="size-3.5" aria-hidden />
            </span>
            <span className="text-sm font-semibold tracking-tight">Intelligence Engine</span>
          </Link>
          <Link href="/" className={buttonClasses("primary", "sm")}>
            Open the app <ArrowRight className="size-3.5" />
          </Link>
        </div>
        <div className="h-px bg-white/5">
          <div className="h-px bg-accent transition-[width] duration-150" style={{ width: `${(position / SCENES) * 100}%` }} />
        </div>
      </header>

      {!calm && (
        <nav className="fixed right-4 top-1/2 z-30 hidden -translate-y-1/2 flex-col gap-2.5 lg:flex" aria-label="Story scenes">
          {Array.from({ length: SCENES }, (_, i) => (
            <button
              key={i}
              onClick={() => scrollTo(i)}
              aria-label={`Go to scene ${i + 1}`}
              aria-current={scene === i ? "step" : undefined}
              className={cn("size-2 rounded-full transition-all", scene === i ? "scale-125 bg-accent" : "bg-white/20 hover:bg-white/50")}
            />
          ))}
        </nav>
      )}

      <main className="relative z-10">
        {/* 1. Hero */}
        <section ref={(el) => { sections.current[0] = el; }} className={sectionClass}>
          <div className={cn(stickyClass, !calm && "sm:items-center")}>
            <div className={cn("max-w-3xl", reveal(0))}>
              <div className="mb-5 font-mono text-[11px] uppercase tracking-[0.18em] text-accent">E-Commerce Intelligence Engine</div>
              <Headline large>What will they buy next?</Headline>
              <p className="mt-6 max-w-xl text-base leading-relaxed text-muted sm:text-lg">
                {kpis ? `${compact(kpis.purchases)} real fashion purchases, ${compact(kpis.buyers)} customers, and ${int(overview.data?.kpis.articles_sold)} products sold` : "Millions of real fashion purchases"}
                . One local AI system that recommends, searches by meaning, predicts who is about to leave, and answers
                questions about it all.
              </p>
              <div className="mt-8 flex flex-wrap items-center gap-3">
                <button onClick={() => scrollTo(1)} className={buttonClasses("secondary")}>
                  <ArrowDown className="size-4" /> Scroll the story
                </button>
                <button onClick={() => scrollTo(6)} className={buttonClasses("ghost")}>
                  Skip to the results
                </button>
              </div>
              {!calm && <p className="mt-8 text-xs text-muted">Behind this text: every product, placed by how it looks.</p>}
            </div>
          </div>
        </section>

        {/* 2. Two years of shopping */}
        <section ref={(el) => { sections.current[1] = el; }} className={sectionClass}>
          <div className={stickyClass}>
            <Card className={reveal(1)}>
              <Eyebrow index={2}>The data</Eyebrow>
              <Headline>Two years of shopping, week by week</Headline>
              <p className="mt-4 text-sm leading-relaxed text-muted">
                Each particle is a slice of a week&apos;s revenue, split into online and in-store sales. Then, in spring 2020,
                the store stream goes dark: for {closure.length || 6} weeks from{" "}
                {closure[0] ? date(closure[0].week) : "23 Mar 2020"}, over 99.9% of items were bought online.
              </p>
              <div className="mt-5 grid grid-cols-3 gap-4">
                <Figure value={compact(kpis?.purchases)} label="items bought" />
                <Figure value={compact(kpis?.orders)} label="shopping visits" />
                <Figure value={pct(kpis?.online_share, 0)} label="bought online" />
              </div>
              <div className="mt-5 flex gap-4">
                <Swatch colour="#7c83fd">Online</Swatch>
                <Swatch colour="#2dd4bf">In store</Swatch>
                <Swatch colour="#f5b547">Store closures</Swatch>
              </div>
              {calm && overview.data && (
                <div className="mt-5">
                  <TrendChart
                    data={overview.data.weekly}
                    x="week"
                    xFormat={month}
                    height={160}
                    series={[{ key: "online_share", label: "Online share" }]}
                    yFormat={(v) => pct(v, 0)}
                  />
                </div>
              )}
            </Card>
          </div>
        </section>

        {/* 3. Recommendations */}
        <section ref={(el) => { sections.current[2] = el; }} className={sectionClass}>
          <div className={stickyClass}>
            <Card className={reveal(2)}>
              <Eyebrow index={3}>Recommendations</Eyebrow>
              <Headline>Six guesses, one ranker, twelve picks</Headline>
              <p className="mt-4 text-sm leading-relaxed text-muted">
                For every customer, six sources each propose products. About{" "}
                {r ? int(r.candidates_per_customer) : "133"} candidates survive deduplication; a LightGBM ranker scores them
                with {r?.ranker.features.length ?? 48} features and keeps the best 12, each with its reason.
              </p>
              <ul className="mt-4 grid grid-cols-2 gap-x-4 gap-y-1.5">
                {Object.entries(sourceSizes ?? {}).map(([name, size], i) => (
                  <li key={name}>
                    <Swatch colour={SOURCE_COLOURS[i % SOURCE_COLOURS.length]}>
                      {SOURCE_NAMES[name] ?? name} <span className="font-mono text-[10px]">· {size}</span>
                    </Swatch>
                  </li>
                ))}
              </ul>
              <div className="mt-5 grid grid-cols-2 gap-4">
                <Figure value={fixed(r?.results.ranker.map_at_12, 4)} label="MAP@12 on an unseen week" />
                <Figure
                  value={r ? `${(r.results.ranker.map_at_12 / r.results.popular.map_at_12).toFixed(1)}×` : "—"}
                  label="the best-seller baseline"
                />
              </div>
            </Card>
          </div>
        </section>

        {/* 4. Search */}
        <section ref={(el) => { sections.current[3] = el; }} className={sectionClass}>
          <div className={stickyClass}>
            <Card className={reveal(3)}>
              <Eyebrow index={4}>Visual search</Eyebrow>
              <Headline>Search by meaning, not keywords</Headline>
              <div className="mt-5 flex items-center gap-2 rounded-xl border border-white/10 bg-white/[0.04] px-3 py-2.5 font-mono text-sm">
                <Sparkles className="size-4 text-accent" aria-hidden />
                <span>{calm ? QUERY : typed}</span>
                {!calm && local(3) < 0.4 && <span className="animate-pulse text-accent">▍</span>}
              </div>
              <p className="mt-4 text-sm leading-relaxed text-muted">
                FashionCLIP reads every photo and description. The query lands where red floral dresses live in the
                Galaxy; the 50 closest products light up.
                {search.error ? " (Text search is off on this server, so the matches can't be shown.)" : ""}
              </p>
              {!capture && search.data && (calm || local(3) > 0.5) && (
                <div className="mt-4 grid grid-cols-4 gap-2">
                  {search.data.slice(0, 4).map((hit) => (
                    <ProductImage key={hit.article.article_id} article={hit.article} />
                  ))}
                </div>
              )}
              <div className="mt-5 grid grid-cols-2 gap-4">
                <Figure value={pct(models.data?.search_quality?.precision_at_10, 0)} label="precision@10, 12 test queries" />
                <Figure value={compact(105542)} label="products searchable" />
              </div>
            </Card>
          </div>
        </section>

        {/* 5. Customers */}
        <section ref={(el) => { sections.current[4] = el; }} className={sectionClass}>
          <div className={stickyClass}>
            <Card className={reveal(4)}>
              <Eyebrow index={5}>Customers</Eyebrow>
              <Headline>Who is about to leave?</Headline>
              <p className="mt-4 text-sm leading-relaxed text-muted">
                {customers.data ? int(customers.data.meta.count) : "30,000"} real active customers, sampled from{" "}
                {customers.data ? int(customers.data.meta.population) : "993,492"}. Further right means more visits,
                higher means more recent, deeper means more predicted value. Colour is the model&apos;s risk of no
                purchase in the next 60 days.
              </p>
              <div className="mt-4 flex gap-4">
                <Swatch colour={RISK_COLOURS[0]}>Low risk</Swatch>
                <Swatch colour={RISK_COLOURS[1]}>Medium</Swatch>
                <Swatch colour={RISK_COLOURS[2]}>High</Swatch>
              </div>
              <label className="mt-5 block">
                <div className="flex justify-between text-sm">
                  <span className="font-medium">Retention budget</span>
                  <span className="font-mono text-accent">{int(budget)} units</span>
                </div>
                <input
                  type="range"
                  min={50}
                  max={1000}
                  step={10}
                  value={budget}
                  onChange={(event) => setBudget(Number(event.target.value))}
                  className="mt-2 w-full"
                />
              </label>
              <div className="mt-4 grid grid-cols-3 gap-4">
                <Figure value={compact(plan.data?.contacted)} label="customers to contact" />
                <Figure value={units(plan.data?.expected_value_saved)} label="value saved (units)" />
                <Figure value={pct(plan.data?.roi, 0)} label="expected ROI" />
              </div>
              {highShare !== null && (
                <p className="mt-4 text-sm leading-relaxed text-fg">
                  Of the customers the planner would contact first, only {pct(highShare, 0)} are high risk. Expected
                  value saved weighs risk by what a customer is worth, so valuable customers at moderate risk come first.
                </p>
              )}
              <p className="mt-3 text-[11px] leading-relaxed text-muted">
                {calm ? "" : "Lit points are the ones the planner would contact first. "}Assumes {pct(UPLIFT, 0)} uplift and {COST} units per
                contact; the churn model scores {fixed(c?.churn.test_results.xgboost_tuned.roc_auc, 3)} ROC AUC on
                unseen months.
              </p>
            </Card>
          </div>
        </section>

        {/* 6. Copilot */}
        <section ref={(el) => { sections.current[5] = el; }} className={sectionClass}>
          <div className={stickyClass}>
            <Card wide className={reveal(5)}>
              <Eyebrow index={6}>Analyst copilot</Eyebrow>
              <Headline>Ask in plain English</Headline>
              <div className="mt-5 rounded-2xl border border-white/10 bg-black/40 p-4">
                <CopilotReplay t={calm ? 1 : local(5)} />
              </div>
              <p className="mt-3 text-[11px] text-muted">
                A recorded run on {date(transcript.recorded)} · {transcript.model} · {transcript.seconds} s ·{" "}
                {compact(transcript.tokens)} tokens. {transcript.verified} The SQL runs read-only in a sandbox.
              </p>
            </Card>
          </div>
        </section>

        {/* 7. Results */}
        <section ref={(el) => { sections.current[6] = el; }} className={calm ? "relative py-16" : "relative h-[160vh]"}>
          <div className={stickyClass}>
            <Card wide className={reveal(6)}>
              <Eyebrow index={7}>Results</Eyebrow>
              <Headline>Measured on data the models never saw</Headline>
              <div className="mt-6 grid grid-cols-2 gap-x-6 gap-y-5 sm:grid-cols-3">
                <Figure value={fixed(r?.results.ranker.map_at_12, 4)} label={`MAP@12 · best sellers ${fixed(r?.results.popular.map_at_12, 4)}`} />
                <Figure value={fixed(c?.churn.test_results.xgboost_tuned.roc_auc, 3)} label="inactivity ROC AUC" />
                <Figure value={fixed(c?.clv.holdout_results.bg_nbd_gamma_gamma?.spend.correlation, 3)} label="value forecast correlation" />
                <Figure value={pct(models.data?.search_quality?.precision_at_10, 0)} label="search precision@10" />
                <Figure value={compact(models.data?.data_quality?.transactions.rows)} label="transactions checked" />
                <Figure value={compact(c?.scores.active_customers)} label="customers scored" />
              </div>
              <div className="mt-8 flex flex-wrap gap-3">
                <Link href="/" className={buttonClasses("primary")}>
                  Open the app <ArrowRight className="size-4" />
                </Link>
                <Link href="/galaxy" className={buttonClasses("secondary")}>
                  <Orbit className="size-4" /> Explore the Galaxy
                </Link>
                <Link href="/models" className={buttonClasses("ghost")}>
                  See every metric
                </Link>
              </div>
            </Card>
          </div>
        </section>
      </main>
    </div>
  );
}
