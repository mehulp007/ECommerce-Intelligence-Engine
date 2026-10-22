import type { Metadata } from "next";
import { Suspense } from "react";

import { CustomersView } from "@/components/views/customers";

export const metadata: Metadata = { title: "Customer 360" };

export default function Page() {
  return (
    <Suspense>
      <CustomersView />
    </Suspense>
  );
}
