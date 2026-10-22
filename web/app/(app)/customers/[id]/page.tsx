import type { Metadata } from "next";
import { notFound } from "next/navigation";

import { CustomerDetailView } from "@/components/views/customer-detail";

export const metadata: Metadata = { title: "Customer" };

export default async function Page(props: PageProps<"/customers/[id]">) {
  const { id } = await props.params;
  if (!/^[0-9a-f]{64}$/.test(id)) notFound();
  return <CustomerDetailView id={id} />;
}
