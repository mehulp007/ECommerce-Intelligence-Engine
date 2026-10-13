import type { Metadata } from "next";
import { notFound } from "next/navigation";

import { ProductDetailView } from "@/components/views/product-detail";

export const metadata: Metadata = { title: "Product" };

export default async function Page(props: PageProps<"/products/[id]">) {
  const { id } = await props.params;
  if (!/^\d{10}$/.test(id)) notFound();
  return <ProductDetailView id={id} />;
}
