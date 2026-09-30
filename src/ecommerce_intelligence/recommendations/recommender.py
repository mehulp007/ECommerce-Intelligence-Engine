"""BM25 recommendations with strict seen-item filtering and popularity fallback."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from implicit.nearest_neighbours import BM25Recommender
from scipy.sparse import csr_matrix, load_npz, save_npz

from ecommerce_intelligence.recommendations.interactions import InteractionData

NEIGHBORS = 50
RECOMMENDATION_COUNT = 5


@dataclass
class ProductRecommender:
    model: BM25Recommender
    user_items: csr_matrix
    customer_ids: list[str]
    stock_codes: list[str]
    catalog: pd.DataFrame

    def __post_init__(self) -> None:
        if self.user_items.shape != (len(self.customer_ids), len(self.stock_codes)):
            raise ValueError("Recommendation matrix and mappings have different shapes")
        if len(self.catalog) != len(self.stock_codes):
            raise ValueError("Catalog and product mapping have different lengths")
        if (
            self.catalog["item_id"].tolist() != list(range(len(self.stock_codes)))
            or self.catalog["stock_code"].tolist() != self.stock_codes
        ):
            raise ValueError("Catalog rows and product indices are not aligned")
        self.customer_index = {
            customer: index for index, customer in enumerate(self.customer_ids)
        }
        self.popular_item_ids = (
            self.catalog.sort_values("popularity_rank")["item_id"].astype(int).tolist()
        )
        self.catalog_by_item = self.catalog.set_index("item_id")

    @classmethod
    def fit(cls, interactions: InteractionData) -> ProductRecommender:
        model = BM25Recommender(K=NEIGHBORS, K1=1.2, B=0.75, num_threads=4)
        model.fit(interactions.matrix, show_progress=False)
        return cls(
            model=model,
            user_items=interactions.matrix,
            customer_ids=interactions.customer_ids,
            stock_codes=interactions.stock_codes,
            catalog=interactions.catalog,
        )

    def seen_item_ids(self, customer_id: str) -> set[int]:
        user_index = self.customer_index.get(str(customer_id))
        if user_index is None:
            return set()
        return {int(item) for item in self.user_items[user_index].indices}

    def popular_only_item_ids(self, customer_id: str, k: int = 5) -> list[int]:
        seen = self.seen_item_ids(customer_id)
        return [item for item in self.popular_item_ids if item not in seen][:k]

    def recommend_item_ids(
        self, customer_id: str, k: int = RECOMMENDATION_COUNT
    ) -> list[tuple[int, float | None, str]]:
        if k < 1:
            raise ValueError("Recommendation count must be positive")
        customer_id = str(customer_id)
        user_index = self.customer_index.get(customer_id)
        seen = self.seen_item_ids(customer_id)
        results: list[tuple[int, float | None, str]] = []
        selected: set[int] = set()

        if user_index is not None and len(seen) < len(self.stock_codes):
            item_ids, scores = self.model.recommend(
                user_index,
                self.user_items[user_index],
                N=min(max(k * 3, k), len(self.stock_codes)),
                filter_already_liked_items=True,
            )
            for item_id, score in zip(item_ids, scores, strict=True):
                item_id = int(item_id)
                if (
                    item_id < 0
                    or item_id in seen
                    or item_id in selected
                    or not np.isfinite(score)
                    or score <= 0
                ):
                    continue
                results.append((item_id, float(score), "bm25"))
                selected.add(item_id)
                if len(results) == k:
                    break

        for item_id in self.popular_item_ids:
            if len(results) == k:
                break
            if item_id in seen or item_id in selected:
                continue
            results.append((item_id, None, "popular_fallback"))
            selected.add(item_id)
        return results

    def recommend(self, customer_id: str, k: int = 5) -> list[dict]:
        output = []
        for item_id, score, source in self.recommend_item_ids(customer_id, k=k):
            product = self.catalog_by_item.loc[item_id]
            output.append(
                {
                    "stock_code": str(product["stock_code"]),
                    "description": str(product["description"]),
                    "median_unit_price": float(product["median_unit_price"]),
                    "score": score,
                    "source": source,
                }
            )
        return output

    def save(self, artifact_dir: Path) -> None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        self.model.save(str(artifact_dir / "bm25_recommender.npz"))
        save_npz(artifact_dir / "recommender_user_items.npz", self.user_items)
        (artifact_dir / "recommender_mappings.json").write_text(
            json.dumps(
                {"customer_ids": self.customer_ids, "stock_codes": self.stock_codes},
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        self.catalog.to_parquet(artifact_dir / "product_catalog.parquet", index=False)

    @classmethod
    def load(cls, artifact_dir: Path) -> ProductRecommender:
        mappings = json.loads(
            (artifact_dir / "recommender_mappings.json").read_text(encoding="utf-8")
        )
        return cls(
            model=BM25Recommender.load(str(artifact_dir / "bm25_recommender.npz")),
            user_items=load_npz(artifact_dir / "recommender_user_items.npz").tocsr(),
            customer_ids=mappings["customer_ids"],
            stock_codes=mappings["stock_codes"],
            catalog=pd.read_parquet(artifact_dir / "product_catalog.parquet"),
        )
