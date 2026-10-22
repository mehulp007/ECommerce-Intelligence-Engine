import type { Metadata } from "next";
import { Suspense } from "react";

import { CampaignsView } from "@/components/views/campaigns";

export const metadata: Metadata = { title: "Campaign simulator" };

export default function Page() {
  return (
    <Suspense>
      <CampaignsView />
    </Suspense>
  );
}
