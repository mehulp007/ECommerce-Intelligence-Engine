"""Visual style: customer profiles built from the embeddings of recent purchases.

Article embeddings come from a pretrained image model, so they carry no
information about future purchases; profiles use purchases before `as_of` only.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import duckdb
import faiss
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from ecommerce_intelligence.paths import ProjectPaths
from ecommerce_intelligence.recsys import config
from ecommerce_intelligence.search.embeddings import ArticleEmbeddings
from ecommerce_intelligence.search.encoder import normalize

RECENCY_DECAY = 0.9
SIMILARITY_CHUNK = 200_000


@dataclass
class StyleModel:
    items: np.ndarray
    profiles: np.ndarray = field(default=None, repr=False)
    customers: np.ndarray = field(default=None, repr=False)

    @classmethod
    def load(cls, paths: ProjectPaths, con: duckdb.DuckDBPyConnection) -> StyleModel:
        embeddings = ArticleEmbeddings.load(paths.embeddings)
        article_ids = con.execute(
            "SELECT article_id FROM article_map ORDER BY article_idx"
        ).fetchnumpy()["article_id"]
        if not np.array_equal(np.asarray(article_ids), embeddings.article_ids):
            raise ValueError("Embeddings and articles are out of sync; rerun embed")
        return cls(normalize(embeddings.item))

    def prepare(self, con: duckdb.DuckDBPyConnection, as_of: str) -> None:
        """Profile every target with recent purchases as a weighted mean vector."""
        history = con.execute(f"""
            SELECT customer_idx, article_idx,
                   row_number() OVER (
                       PARTITION BY customer_idx
                       ORDER BY last_date DESC, article_idx
                   ) AS recency
            FROM (
                SELECT customer_idx, article_idx, max(purchase_date) AS last_date
                FROM tx SEMI JOIN targets USING (customer_idx)
                WHERE purchase_date >= DATE '{as_of}' - INTERVAL
                        {config.REPURCHASE_DAYS} DAY
                  AND purchase_date < DATE '{as_of}'
                GROUP BY customer_idx, article_idx
            )
            QUALIFY recency <= {config.STYLE_SEEDS}
            """).fetchnumpy()
        customers = np.unique(history["customer_idx"]).astype(np.int32)
        self.customers = customers
        if not len(customers):
            self.profiles = np.zeros((0, self.items.shape[1]), dtype=np.float32)
            return
        rows = np.searchsorted(customers, history["customer_idx"])
        weights = RECENCY_DECAY ** (history["recency"].astype(np.float32) - 1)
        mixer = csr_matrix(
            (weights, (rows, history["article_idx"])),
            shape=(len(customers), len(self.items)),
        )
        self.profiles = normalize(mixer @ self.items)

    def candidates(self, con: duckdb.DuckDBPyConnection, as_of: str) -> None:
        """Recently sold articles closest to each customer's style profile."""
        active = con.execute(f"""
            SELECT DISTINCT article_idx FROM tx
            WHERE purchase_date >= DATE '{as_of}' - INTERVAL
                    {config.STYLE_ACTIVE_DAYS} DAY
              AND purchase_date < DATE '{as_of}'
            ORDER BY article_idx
            """).fetchnumpy()["article_idx"]
        frame = pd.DataFrame(
            {
                "customer_idx": pd.Series(dtype="int32"),
                "article_idx": pd.Series(dtype="int32"),
                "style_score": pd.Series(dtype="float32"),
                "style_rank": pd.Series(dtype="int16"),
            }
        )
        if len(active) and len(self.profiles):
            index = faiss.IndexFlatIP(self.items.shape[1])
            index.add(np.ascontiguousarray(self.items[active]))
            k = min(config.STYLE_CANDIDATES, len(active))
            scores, rows = index.search(np.ascontiguousarray(self.profiles), k)
            frame = pd.DataFrame(
                {
                    "customer_idx": np.repeat(self.customers, k),
                    "article_idx": active[rows.ravel()].astype(np.int32),
                    "style_score": scores.ravel().astype(np.float32),
                    "style_rank": np.tile(
                        np.arange(1, k + 1, dtype=np.int16), len(self.customers)
                    ),
                }
            )
        con.register("style_frame", frame)
        con.execute("CREATE OR REPLACE TABLE cand_style AS SELECT * FROM style_frame")
        con.unregister("style_frame")

    def similarity(
        self, customer_idx: np.ndarray, article_idx: np.ndarray
    ) -> np.ndarray:
        """Cosine similarity between profile and article; NaN without a profile."""
        customer_idx = np.asarray(customer_idx)
        result_rows = np.searchsorted(self.customers, customer_idx)
        result_rows = np.minimum(result_rows, max(len(self.customers) - 1, 0))
        found = (
            self.customers[result_rows] == customer_idx
            if len(self.customers)
            else np.zeros(len(customer_idx), dtype=bool)
        )
        rows = np.where(found, result_rows, -1)
        result = np.full(len(rows), np.nan, dtype=np.float32)
        known = np.flatnonzero(rows >= 0)
        for start in range(0, len(known), SIMILARITY_CHUNK):
            part = known[start : start + SIMILARITY_CHUNK]
            result[part] = np.einsum(
                "ij,ij->i", self.profiles[rows[part]], self.items[article_idx[part]]
            )
        return result
