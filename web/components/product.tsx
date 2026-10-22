"use client";

import { ImageOff } from "lucide-react";
import Link from "next/link";
import { useState, type ReactNode } from "react";

import { Badge } from "@/components/ui";
import { imageUrl } from "@/lib/api";
import { compact } from "@/lib/format";
import { REASONS, RISK } from "@/lib/labels";
import type { Article, Reason, RiskBand } from "@/lib/types";
import { cn } from "@/lib/utils";

export function ProductImage({
  article,
  className,
  priority = false,
}: {
  article: Pick<Article, "article_id" | "has_image" | "prod_name">;
  className?: string;
  priority?: boolean;
}) {
  const [failed, setFailed] = useState(false);
  const show = article.has_image && !failed;
  return (
    <div className={cn("relative aspect-[2/3] overflow-hidden rounded-xl bg-surface-2", className)}>
      {show ? (
        // Thumbnails are already small WebP files, so they are served as they are.
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={imageUrl(article.article_id)}
          alt={article.prod_name ?? "Product photo"}
          loading={priority ? "eager" : "lazy"}
          decoding="async"
          onError={() => setFailed(true)}
          className="size-full object-cover"
        />
      ) : (
        <div className="flex size-full flex-col items-center justify-center gap-1 text-xs text-muted">
          <ImageOff className="size-5" aria-hidden />
          No photo
        </div>
      )}
    </div>
  );
}

export function ProductCard({
  article,
  footer,
  badge,
  priority = false,
}: {
  article: Article;
  footer?: ReactNode;
  badge?: ReactNode;
  /** Load the photo immediately, for cards visible on first paint. */
  priority?: boolean;
}) {
  return (
    <Link
      href={`/products/${article.article_id}`}
      className="group flex flex-col gap-2 rounded-2xl border border-border bg-surface p-2 transition-[border-color,transform] duration-200 hover:-translate-y-0.5 hover:border-accent/50 motion-reduce:hover:translate-y-0"
    >
      <div className="relative">
        <ProductImage article={article} priority={priority} />
        {badge && <div className="absolute left-1.5 top-1.5">{badge}</div>}
      </div>
      <div className="min-w-0 px-1 pb-1">
        <div className="truncate text-sm font-medium group-hover:text-accent">{article.prod_name ?? article.article_id}</div>
        <div className="truncate text-xs text-muted">
          {[article.colour_group_name, article.product_type_name].filter(Boolean).join(" · ")}
        </div>
        {footer ?? (
          <div className="mt-1 text-xs text-muted">
            {article.sales_4w ? `${compact(article.sales_4w)} sold in last 4 weeks` : "No sales in last 4 weeks"}
          </div>
        )}
      </div>
    </Link>
  );
}

export function ProductGrid({ children, dense = false }: { children: ReactNode; dense?: boolean }) {
  return (
    <div
      className={cn(
        "grid gap-3",
        dense
          ? "grid-cols-2 sm:grid-cols-3 md:grid-cols-4 xl:grid-cols-6"
          : "grid-cols-2 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5",
      )}
    >
      {children}
    </div>
  );
}

export function ReasonBadge({ reason }: { reason: Reason }) {
  const info = REASONS[reason];
  return (
    <Badge className="border border-white/10 bg-black/70 text-fg backdrop-blur" title={info.hint}>
      {info.label}
    </Badge>
  );
}

export function RiskBadge({ band }: { band: RiskBand }) {
  const info = RISK[band];
  return <Badge className={info.className}>{info.label}</Badge>;
}
