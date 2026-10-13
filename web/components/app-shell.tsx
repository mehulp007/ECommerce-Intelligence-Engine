"use client";

import { useQuery } from "@tanstack/react-query";
import {
  BarChart3,
  FlaskConical,
  LayoutDashboard,
  Megaphone,
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

import { ThemeToggle } from "@/components/theme";
import { getJson } from "@/lib/api";
import { date } from "@/lib/format";
import type { Health } from "@/lib/types";
import { cn } from "@/lib/utils";

const NAV = [
  { href: "/", label: "Overview", icon: LayoutDashboard },
  { href: "/customers", label: "Customer 360", icon: Users },
  { href: "/products", label: "Products", icon: Shirt },
  { href: "/segments", label: "Segments", icon: PieChart },
  { href: "/campaigns", label: "Campaigns", icon: Megaphone },
  { href: "/copilot", label: "Copilot", icon: Sparkles },
  { href: "/models", label: "Model Lab", icon: FlaskConical },
];

function ApiStatus() {
  const health = useQuery({
    queryKey: ["health"],
    queryFn: () => getJson<Health>("/api/health"),
    staleTime: 60_000,
    refetchInterval: 60_000,
  });
  const ok = health.isSuccess;
  return (
    <div className="flex items-start gap-2 text-xs text-muted" title={health.error?.message}>
      <span
        className={cn(
          "mt-1 size-2 shrink-0 rounded-full",
          health.isPending ? "bg-warn" : ok ? "bg-good" : "bg-bad",
        )}
        aria-hidden
      />
      <span>
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
    <nav className="flex flex-col gap-0.5">
      {NAV.map(({ href, label, icon: Icon }) => {
        const active = href === "/" ? pathname === "/" : pathname.startsWith(href);
        return (
          <Link
            key={href}
            href={href}
            onClick={onNavigate}
            className={cn(
              "flex items-center gap-2.5 rounded-lg px-3 py-2 text-sm transition-colors",
              active ? "bg-accent-soft font-medium text-accent" : "text-muted hover:bg-surface-2 hover:text-fg",
            )}
          >
            <Icon className="size-4" aria-hidden />
            {label}
          </Link>
        );
      })}
    </nav>
  );
}

function Brand() {
  return (
    <Link href="/" className="flex items-center gap-2.5">
      <span className="flex size-8 items-center justify-center rounded-lg bg-accent text-accent-fg">
        <BarChart3 className="size-4" aria-hidden />
      </span>
      <span className="leading-tight">
        <span className="block text-sm font-semibold">Intelligence Engine</span>
        <span className="block text-xs text-muted">H&amp;M fashion retail data</span>
      </span>
    </Link>
  );
}

export function AppShell({ children }: { children: ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="flex min-h-screen">
      <aside className="sticky top-0 hidden h-screen w-60 shrink-0 flex-col gap-6 border-r border-border bg-surface p-4 lg:flex">
        <Brand />
        <NavLinks />
        <div className="mt-auto flex flex-col gap-3">
          <ApiStatus />
          <ThemeToggle />
        </div>
      </aside>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-30 flex items-center justify-between border-b border-border bg-surface/90 px-4 py-3 backdrop-blur lg:hidden">
          <Brand />
          <button
            onClick={() => setOpen(true)}
            className="rounded-lg p-2 hover:bg-surface-2"
            aria-label="Open navigation"
          >
            <Menu className="size-5" />
          </button>
        </header>

        {open && (
          <div className="fixed inset-0 z-40 lg:hidden" role="dialog" aria-modal>
            <div className="absolute inset-0 bg-black/40" onClick={() => setOpen(false)} />
            <div className="absolute inset-y-0 left-0 flex w-64 flex-col gap-6 bg-surface p-4">
              <div className="flex items-center justify-between">
                <Brand />
                <button
                  onClick={() => setOpen(false)}
                  className="rounded-lg p-2 hover:bg-surface-2"
                  aria-label="Close navigation"
                >
                  <X className="size-5" />
                </button>
              </div>
              <NavLinks onNavigate={() => setOpen(false)} />
              <div className="mt-auto flex flex-col gap-3">
                <ApiStatus />
                <ThemeToggle />
              </div>
            </div>
          </div>
        )}

        <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-6 sm:px-6 lg:px-8 lg:py-8">{children}</main>
      </div>
    </div>
  );
}
