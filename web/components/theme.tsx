"use client";

import { Monitor, Moon, Sun } from "lucide-react";
import { useSyncExternalStore } from "react";

type Theme = "light" | "dark" | "system";
const KEY = "theme";
const listeners = new Set<() => void>();

/** Runs before the page paints so the first frame already has the right theme. */
export const themeScript = `(() => {
  try {
    const saved = localStorage.getItem("${KEY}");
    const dark = saved === "dark" || (saved !== "light" && matchMedia("(prefers-color-scheme: dark)").matches);
    document.documentElement.classList.toggle("dark", dark);
  } catch {}
})();`;

function read(): Theme {
  try {
    const saved = localStorage.getItem(KEY);
    return saved === "light" || saved === "dark" ? saved : "system";
  } catch {
    return "system";
  }
}

function apply(theme: Theme) {
  const dark =
    theme === "dark" || (theme === "system" && matchMedia("(prefers-color-scheme: dark)").matches);
  document.documentElement.classList.toggle("dark", dark);
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  const media = matchMedia("(prefers-color-scheme: dark)");
  const onSystem = () => {
    if (read() === "system") apply("system");
  };
  media.addEventListener("change", onSystem);
  return () => {
    listeners.delete(listener);
    media.removeEventListener("change", onSystem);
  };
}

const NEXT: Record<Theme, Theme> = { system: "light", light: "dark", dark: "system" };
const ICONS = { light: Sun, dark: Moon, system: Monitor };

export function ThemeToggle() {
  const theme = useSyncExternalStore(subscribe, read, () => "system" as Theme);
  const Icon = ICONS[theme];
  return (
    <button
      onClick={() => {
        const next = NEXT[theme];
        try {
          if (next === "system") localStorage.removeItem(KEY);
          else localStorage.setItem(KEY, next);
        } catch {
          // Storage can be blocked; the theme still applies for this page.
        }
        apply(next);
        listeners.forEach((listener) => listener());
      }}
      className="inline-flex h-8 items-center gap-1.5 rounded-lg px-2 text-xs text-muted hover:bg-surface-2 hover:text-fg"
      title={`Theme: ${theme} (click to change)`}
    >
      <Icon className="size-4" aria-hidden />
      <span className="capitalize">{theme}</span>
    </button>
  );
}
