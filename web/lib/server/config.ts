import "server-only";

import { readFileSync } from "node:fs";
import path from "node:path";

/** Read one variable from the repository's .env (one folder above web/). */
function fromRootEnv(name: string): string | undefined {
  try {
    const text = readFileSync(path.join(process.cwd(), "..", ".env"), "utf8");
    for (const line of text.split(/\r?\n/)) {
      const match = line.match(/^\s*([A-Z0-9_]+)\s*=\s*(.*)\s*$/);
      if (match && match[1] === name) {
        return match[2].replace(/^(['"])(.*)\1$/, "$2") || undefined;
      }
    }
  } catch {
    // No .env file: rely on the process environment.
  }
  return undefined;
}

export function apiUrl(): string {
  return (process.env.EIE_API_URL || "http://127.0.0.1:8000").replace(/\/+$/, "");
}

/** The API's bearer key. It stays on the server and never reaches the browser. */
export function apiKey(): string {
  return process.env.EIE_API_KEY || fromRootEnv("EIE_API_KEY") || "";
}
