import type { Metadata } from "next";
import { Suspense } from "react";

import { GalaxyView } from "@/components/views/galaxy";

export const metadata: Metadata = { title: "Product Galaxy" };

export default function Page() {
  return (
    <Suspense>
      <GalaxyView />
    </Suspense>
  );
}
