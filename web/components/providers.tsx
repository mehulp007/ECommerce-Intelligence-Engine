"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useState, type ReactNode } from "react";

import { ApiError } from "@/lib/api";

export function Providers({ children }: { children: ReactNode }) {
  const [client] = useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            // The data is a historical snapshot, so results rarely change.
            staleTime: 10 * 60 * 1000,
            refetchOnWindowFocus: false,
            retry: (count, error) =>
              count < 1 && !(error instanceof ApiError && error.status < 500),
          },
        },
      }),
  );
  return <QueryClientProvider client={client}>{children}</QueryClientProvider>;
}
