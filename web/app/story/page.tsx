import type { Metadata } from "next";
import { Instrument_Serif } from "next/font/google";
import { Suspense } from "react";

import { StoryView } from "@/components/story/story-view";

const serif = Instrument_Serif({ variable: "--font-serif", weight: "400", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "The story",
  description: "A scroll-through tour of the engine: the data, recommendations, visual search, churn, and the copilot.",
};

export default function Page() {
  return (
    <div className={serif.variable}>
      <Suspense>
        <StoryView />
      </Suspense>
    </div>
  );
}
