import type { Metadata } from "next";

import { SegmentsView } from "@/components/views/segments";

export const metadata: Metadata = { title: "Segments" };

export default function Page() {
  return <SegmentsView />;
}
