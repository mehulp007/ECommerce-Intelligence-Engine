import type { Metadata } from "next";
import { Suspense } from "react";

import { CopilotView } from "@/components/views/copilot";

export const metadata: Metadata = { title: "Analyst copilot" };

export default function Page() {
  return (
    <Suspense>
      <CopilotView />
    </Suspense>
  );
}
