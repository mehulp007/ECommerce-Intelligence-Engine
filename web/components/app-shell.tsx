"use client";

import { useQuery } from "@tanstack/react-query";
import {
  BarChart3,
  FlaskConical,
  LayoutDashboard,
  Megaphone,
  Orbit,
  Play,
  Menu,
  PieChart,
  Shirt,
  Sparkles,
  Users,
  X,
} from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState, type ReactNode } from "react";

import { getJson } from "@/lib/api";
import { date } from "@/lib/format";
import type { Health } from "@/lib/types";
import { cn } from "@/lib/utils";

const NAV = [
  {
    group: "Workspace",
    items: [
      { href: "/", label: "Overview", icon: LayoutDashboard },
      { href: "/customers", label: "Customer 360", icon: Users },
      { href: "/products", label: "Products", icon: Shirt },
      { href: "/galaxy", label: "Galaxy", icon: Orbit },
      { href: "/segments", label: "Segments", icon: PieChart },
      { href: "/campaigns", label: "Campaigns", icon: Megaphone },
    ],
  },
  {
    group: "Intelligence",
    items: [
      { href: "/copilot", label: "Copilot", icon: Sparkles },
      { href: "/models", label: "Model Lab", icon: FlaskConical },
    ],
  },
];

const FULL_BLEED = ["/galaxy"];

function ApiStatus() {
  const health = useQuery({
    queryKey: ["health"],
    queryFn: () => getJson<Health>("/api/health"),
    staleTime: 60_000,
    refetchInterval: 60_000,
  });
  const ok = health.isSuccess;
  return (
    <div
      className="flex items-start gap-2.5 rounded-xl border border-border bg-surface px-3 py-2.5 text-xs text-muted"
      title={health.error?.message}
    >
      <span className="relative mt-1 flex size-2 shrink-0" aria-hidden>
        {ok && <span className="absolute inset-0 animate-ping rounded-full bg-good/60 motion-reduce:hidden" />}
        <span
          className={cn(
            "relative size-2 rounded-full",
            health.isPending ? "bg-warn" : ok ? "bg-good" : "bg-bad",
          )}
        />
      </span>
      <span className="leading-relaxed">
        {health.isPending
          ? "Connecting to the API…"
          : ok
            ? `API ready · data to ${date(health.data.data_end)}`
            : "API offline: run eie serve run"}
      </span>
    </div>
  );
}

function NavLinks({ onNavigate }: { onNavigate?: () => void }) {
  const pathname = usePathname();
  return (
    <nav className="flex flex-col gap-5">
      {NAV.map(({ group, items }) => (
        <div key={group}>
          <div className="mb-1.5 px-3 text-[11px] font-medium uppercase tracking-[0.08em] text-muted">{group}</div>
          <div className="flex flex-col gap-0.5">
            {items.map(({ href, label, icon: Icon }) => {
              const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
              return (
                <Link
                  key={href}
                  href={href}
                  onClick={onNavigate}
                  aria-current={active ? "page" : undefined}
                  className={cn(
                    "relative flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm transition-colors",
                    active ? "bg-surface-2 font-medium text-fg" : "text-muted hover:bg-surface hover:text-fg",
                  )}
                >
                  {active && (
                    <span className="absolute inset-y-1.5 left-0 w-0.5 rounded-full bg-accent" aria-hidden />
                  )}
                  <Icon className={cn("size-4", active && "text-accent")} aria-hidden />
                  {label}
                </Link>
              );
            })}
          </div>
        </div>
      ))}
    </nav>
  );
}

function StoryLink({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <Link
      href="/story"
      onClick={onNavigate}
      className="group flex items-center gap-2.5 rounded-xl border border-accent/25 bg-accent-soft px-3 py-2.5 text-sm transition-colors hover:border-accent/50"
    >
      <span className="flex size-6 items-center justify-center rounded-md bg-accent text-accent-fg">
        <Play className="size-3 fill-current" aria-hidden />
      </span>
      <span className="leading-tight">
        <span className="block font-medium text-fg">Watch the story</span>
        <span className="block text-xs text-muted">A 2-minute scroll tour</span>
      </span>
    </Link>
  );
}

function Brand() {
  return (
    <Link href="/" className="flex items-center gap-2.5">
      <span className="flex size-8 items-center justify-center rounded-lg bg-accent text-accent-fg shadow-[0_0_24px_-6px_rgba(124,131,253,0.6)]">
        <BarChart3 className="size-4" aria-hidden />
      </span>
      <span className="leading-tight">
        <span className="block text-sm font-semibold tracking-tight">Intelligence Engine</span>
        <span className="block text-xs text-muted">H&amp;M fashion retail data</span>
      </span>
    </Link>
  );
}

export function AppShell({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);
  const pathname = usePathname();
  return (
    <div className="flex min-h-screen">
      <aside className="sticky top-0 hidden h-screen w-60 shrink-0 flex-col gap-7 border-r border-border bg-bg px-3 py-5 lg:flex">
        <div className="px-2">
          <Brand />
        </div>
        <StoryLink />
        <NavLinks />
        <div className="mt-auto">
          <ApiStatus />
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-30 flex items-center justify-between border-b border-border bg-bg/85 px-4 py-3 backdrop-blur lg:hidden">
          <Brand />
          <button
            onClick={() => setOpen(true)}
            className="rounded-lg p-2 text-muted hover:bg-surface-2 hover:text-fg"
            aria-label="Open navigation"
          >
            <Menu className="size-5" />
          </button>
        </header>

        {open && (
          <div className="fixed inset-0 z-40 lg:hidden" role="dialog" aria-modal>
            <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={() => setOpen(false)} />
            <div className="absolute inset-y-0 left-0 flex w-64 flex-col gap-7 border-r border-border bg-bg px-3 py-5">
              <div className="flex items-center justify-between px-2">
                <Brand />
                <button
                  onClick={() => setOpen(false)}
                  className="rounded-lg p-2 text-muted hover:bg-surface-2 hover:text-fg"
                  aria-label="Close navigation"
                >
                  <X className="size-5" />
                </button>
              </div>
              <StoryLink onNavigate={() => setOpen(false)} />
              <NavLinks onNavigate={() => setOpen(false)} />
              <div className="mt-auto">
                <ApiStatus />
              </div>
            </div>
          </div>
        )}

        {/* Full-screen pages, like the Galaxy, take the whole content area. */}
        {FULL_BLEED.some((path) => pathname.startsWith(path)) ? (
          <main className="relative min-w-0 flex-1">{children}</main>
        ) : (
          <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-6 sm:px-6 lg:px-10 lg:py-9">
            <div key={pathname} className="page-in">
              {children}
            </div>
          </main>
        )}
      </div>
    </div>
  );
}
