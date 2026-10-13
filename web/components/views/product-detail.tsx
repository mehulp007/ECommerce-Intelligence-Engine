"use client";

import { useQuery } from "@tanstack/react-query";
import { ArrowLeft } from "lucide-react";
import Link from "next/link";

import { BarsChart } from "@/components/charts";
import { ProductCard, ProductGrid, ProductImage } from "@/components/product";
import { Badge, Card, CardTitle, EmptyState, ErrorState, Skeleton } from "@/components/ui";
import { ApiError, getJson } from "@/lib/api";
import { compact, date, int, pct, units } from "@/lib/format";
import type { ArticleDetail, SearchHit } from "@/lib/types";

export function ProductDetailView({ id }: { id: string }) {
  const product = useQuery({
    queryKey: ["product", id],
    queryFn: () => getJson<ArticleDetail>(`/api/products/${id}`),
  });
  const similar = useQuery({
    queryKey: ["similar", id],
    queryFn: () => getJson<SearchHit[]>(`/api/products/${id}/similar`, { k: 12 }),
    enabled: product.isSuccess,
  });

  const back = (
    <Link href="/products" className="mb-4 inline-flex items-center gap-1 text-sm text-muted hover:text-fg">
      <ArrowLeft className="size-4" /> Products
    </Link>
  );

  if (product.error) {
    return (
      <>
        {back}
        {product.error instanceof ApiError && product.error.status === 404 ? (
          <EmptyState>No product has this ID.</EmptyState>
        ) : (
          <ErrorState error={product.error} onRetry={() => product.refetch()} />
        )}
      </>
    );
  }

  const p = product.data;
  // The last 12 weeks before the end of the data, oldest first.
  const weeks = p?.weekly_sales_12w.map((sold, i, all) => ({ week: i === all.length - 1 ? "Last" : `-${all.length - 1 - i}w`, sold })) ?? [];

  return (
    <>
      {back}
      <div className="grid gap-6 md:grid-cols-[minmax(0,320px)_1fr]">
        {p ? <ProductImage article={p} priority className="w-full" /> : <Skeleton className="aspect-[2/3]" />}
        <div className="min-w-0">
          {p ? (
            <>
              <div className="text-xs text-muted">Article {p.article_id}</div>
              <h1 className="mt-1 text-2xl font-semibold tracking-tight">{p.prod_name}</h1>
              <div className="mt-2 flex flex-wrap gap-1.5">
                {[
                  ...new Set(
                    [p.index_group_name, p.product_group_name, p.product_type_name, p.colour_group_name, p.department_name].filter(
                      (tag): tag is string => Boolean(tag),
                    ),
                  ),
                ].map((tag) => (
                    <Badge key={tag}>{tag}</Badge>
                  ))}
              </div>
              {p.detail_desc && <p className="mt-4 max-w-2xl text-sm leading-relaxed text-muted">{p.detail_desc}</p>}
              <dl className="mt-5 grid grid-cols-2 gap-x-6 gap-y-3 text-sm sm:grid-cols-4">
                {[
                  ["Sold, all time", int(p.sales_total)],
                  ["Sold, last week", int(p.sales_1w)],
                  ["Sold, last 4 weeks", int(p.sales_4w)],
                  ["Popularity rank (4 weeks)", p.popularity_rank_4w ? `#${int(p.popularity_rank_4w)}` : "—"],
                  ["Customers", compact(p.buyers_total)],
                  ["Median price", `${units(p.median_price)} units`],
                  ["Bought online", pct(p.online_share, 0)],
                  ["Sold between", `${date(p.first_sale)} – ${date(p.last_sale)}`],
                ].map(([label, value]) => (
                  <div key={label}>
                    <dt className="text-xs text-muted">{label}</dt>
                    <dd className="mt-0.5 font-medium">{value}</dd>
                  </div>
                ))}
              </dl>
              <Card className="mt-5">
                <CardTitle hint="Items sold per week, up to 22 Sep 2020">Last 12 weeks</CardTitle>
                <BarsChart data={weeks} x="week" height={160} series={[{ key: "sold", label: "Sold" }]} />
              </Card>
            </>
          ) : (
            <div className="flex flex-col gap-3">
              <Skeleton className="h-8 w-2/3" />
              <Skeleton className="h-24" />
              <Skeleton className="h-40" />
            </div>
          )}
        </div>
      </div>

      <Card className="mt-6">
        <CardTitle hint="Nearest neighbours in FashionCLIP's joint image and text space">Looks similar</CardTitle>
        {similar.error ? (
          <ErrorState error={similar.error} />
        ) : similar.data ? (
          <ProductGrid dense>
            {similar.data.map((hit) => (
              <ProductCard
                key={hit.article.article_id}
                article={hit.article}
                footer={<div className="mt-1 text-xs text-muted">Similarity {hit.score.toFixed(3)}</div>}
              />
            ))}
          </ProductGrid>
        ) : (
          <Skeleton className="h-64" />
        )}
      </Card>
    </>
  );
}
