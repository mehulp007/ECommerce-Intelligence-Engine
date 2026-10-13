"use client";

import { keepPreviousData, useQuery } from "@tanstack/react-query";
import { ChevronLeft, ChevronRight, Search, Sparkles } from "lucide-react";
import { useRouter, useSearchParams } from "next/navigation";
import { useState } from "react";

import { ProductCard, ProductGrid } from "@/components/product";
import { Button, Card, EmptyState, ErrorState, Input, PageHeader, Segmented, Select, Skeleton } from "@/components/ui";
import { getJson } from "@/lib/api";
import { int } from "@/lib/format";
import type { ArticlePage, Facets, SearchHit } from "@/lib/types";

const PAGE = 24;
const EXAMPLES = [
  "red floral summer dress",
  "black leather ankle boots",
  "kids dinosaur t-shirt",
  "striped swimsuit",
  "chunky knitted cardigan",
  "white sneakers",
];

type Mode = "browse" | "describe";

function SkeletonGrid() {
  return (
    <ProductGrid>
      {Array.from({ length: 10 }, (_, i) => (
        <Skeleton key={i} className="aspect-[2/3.6]" />
      ))}
    </ProductGrid>
  );
}

function Describe({ query, onQuery }: { query: string; onQuery: (q: string) => void }) {
  const [text, setText] = useState(query);
  const results = useQuery({
    queryKey: ["search", query],
    queryFn: () => getJson<SearchHit[]>("/api/search", { q: query, k: 30 }),
    enabled: query.length >= 2,
  });
  return (
    <>
      <Card className="mb-4">
        <form
          className="flex gap-2"
          onSubmit={(event) => {
            event.preventDefault();
            onQuery(text.trim());
          }}
        >
          <div className="relative flex-1">
            <Sparkles className="pointer-events-none absolute left-3 top-2.5 size-4 text-accent" aria-hidden />
            <Input
              value={text}
              onChange={(event) => setText(event.target.value)}
              placeholder="Describe what you are looking for…"
              className="pl-9"
              maxLength={120}
              aria-label="Describe a product"
            />
          </div>
          <Button variant="primary" type="submit" disabled={text.trim().length < 2}>
            Search
          </Button>
        </form>
        <div className="mt-3 flex flex-wrap gap-1.5">
          {EXAMPLES.map((example) => (
            <button
              key={example}
              onClick={() => {
                setText(example);
                onQuery(example);
              }}
              className="rounded-full border border-border px-2.5 py-1 text-xs text-muted hover:border-accent hover:text-accent"
            >
              {example}
            </button>
          ))}
        </div>
        <p className="mt-3 text-xs text-muted">
          FashionCLIP embeds your words and every product photo and description in one space, then returns the closest
          products. Nothing here matches keywords. The first search can take a few seconds while the model loads.
        </p>
      </Card>
      {query.length < 2 ? (
        <EmptyState>Type a description or pick an example.</EmptyState>
      ) : results.error ? (
        <ErrorState error={results.error} onRetry={() => results.refetch()} />
      ) : results.isPending ? (
        <SkeletonGrid />
      ) : (
        <ProductGrid>
          {results.data.map((hit, i) => (
            <ProductCard
              key={hit.article.article_id}
              article={hit.article}
              priority={i < 10}
              footer={<div className="mt-1 text-xs text-muted">Similarity {hit.score.toFixed(3)}</div>}
            />
          ))}
        </ProductGrid>
      )}
    </>
  );
}

function Browse() {
  const router = useRouter();
  const params = useSearchParams();
  const filters = {
    index_group: params.get("index_group") ?? "",
    product_group: params.get("product_group") ?? "",
    colour: params.get("colour") ?? "",
    q: params.get("q") ?? "",
    sort: params.get("sort") ?? "popular",
  };
  const page = Math.max(0, Number(params.get("page") ?? 0));
  const [text, setText] = useState(filters.q);

  const update = (changes: Record<string, string>) => {
    const next = new URLSearchParams(params);
    for (const [key, value] of Object.entries(changes)) {
      if (value) next.set(key, value);
      else next.delete(key);
    }
    if (!("page" in changes)) next.delete("page");
    router.replace(`/products?${next.toString()}`, { scroll: false });
  };

  const facets = useQuery({ queryKey: ["facets"], queryFn: () => getJson<Facets>("/api/products/facets") });
  const results = useQuery({
    queryKey: ["browse", filters, page],
    queryFn: () => getJson<ArticlePage>("/api/products", { ...filters, limit: PAGE, offset: page * PAGE }),
    placeholderData: keepPreviousData,
  });
  const pages = results.data ? Math.ceil(results.data.total / PAGE) : 0;

  return (
    <>
      <Card className="mb-4">
        <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-[1.4fr_1fr_1fr_1fr_auto]">
          <form
            className="relative sm:col-span-2 lg:col-span-1"
            onSubmit={(event) => {
              event.preventDefault();
              update({ q: text.trim() });
            }}
          >
            <Search className="pointer-events-none absolute left-3 top-2.5 size-4 text-muted" aria-hidden />
            <Input
              value={text}
              onChange={(event) => setText(event.target.value)}
              onBlur={() => text.trim() !== filters.q && update({ q: text.trim() })}
              placeholder="Filter by product name"
              className="pl-9"
              maxLength={80}
              aria-label="Product name"
            />
          </form>
          <Select value={filters.index_group} onChange={(e) => update({ index_group: e.target.value })} aria-label="Index group">
            <option value="">All index groups</option>
            {facets.data?.index_groups.map((v) => <option key={v}>{v}</option>)}
          </Select>
          <Select value={filters.product_group} onChange={(e) => update({ product_group: e.target.value })} aria-label="Product group">
            <option value="">All product groups</option>
            {facets.data?.product_groups.map((v) => <option key={v}>{v}</option>)}
          </Select>
          <Select value={filters.colour} onChange={(e) => update({ colour: e.target.value })} aria-label="Colour">
            <option value="">All colours</option>
            {facets.data?.colours.map((v) => <option key={v}>{v}</option>)}
          </Select>
          <Select value={filters.sort} onChange={(e) => update({ sort: e.target.value })} aria-label="Sort">
            <option value="popular">Best selling (4 weeks)</option>
            <option value="newest">Newest</option>
            <option value="price_low">Price: low to high</option>
            <option value="price_high">Price: high to low</option>
          </Select>
        </div>
      </Card>

      {results.error ? (
        <ErrorState error={results.error} onRetry={() => results.refetch()} />
      ) : !results.data ? (
        <SkeletonGrid />
      ) : results.data.total === 0 ? (
        <EmptyState>No products match these filters.</EmptyState>
      ) : (
        <>
          <div className="mb-3 text-sm text-muted">{int(results.data.total)} products</div>
          <div className={results.isPlaceholderData ? "opacity-60 transition-opacity" : undefined}>
            <ProductGrid>
              {results.data.items.map((article, i) => (
                <ProductCard key={article.article_id} article={article} priority={i < 10} />
              ))}
            </ProductGrid>
          </div>
          <div className="mt-6 flex items-center justify-center gap-3">
            <Button size="sm" disabled={page === 0} onClick={() => update({ page: String(page - 1) })}>
              <ChevronLeft className="size-4" /> Previous
            </Button>
            <span className="text-sm tabular-nums text-muted">
              Page {page + 1} of {int(pages)}
            </span>
            <Button size="sm" disabled={page + 1 >= pages} onClick={() => update({ page: String(page + 1) })}>
              Next <ChevronRight className="size-4" />
            </Button>
          </div>
        </>
      )}
    </>
  );
}

export function ProductsView() {
  const router = useRouter();
  const params = useSearchParams();
  const mode: Mode = params.get("mode") === "describe" ? "describe" : "browse";
  const query = params.get("search") ?? "";

  return (
    <>
      <PageHeader
        title="Products"
        description="Browse 105k articles, or describe what you want in plain words and let FashionCLIP find it from the photos."
        actions={
          <Segmented<Mode>
            label="Search mode"
            value={mode}
            onChange={(next) => router.replace(next === "describe" ? "/products?mode=describe" : "/products")}
            options={[
              { value: "browse", label: "Browse catalog" },
              { value: "describe", label: "Describe it" },
            ]}
          />
        }
      />
      {mode === "describe" ? (
        <Describe
          key={query}
          query={query}
          onQuery={(q) => router.replace(`/products?mode=describe&search=${encodeURIComponent(q)}`, { scroll: false })}
        />
      ) : (
        <Browse />
      )}
    </>
  );
}
