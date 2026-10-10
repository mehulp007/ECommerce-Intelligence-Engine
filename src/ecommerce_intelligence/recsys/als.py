"""Implicit-feedback matrix factorization (ALS) over recent purchases."""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import numpy as np
from implicit.cpu.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix
from threadpoolctl import threadpool_limits

from ecommerce_intelligence.recsys import config

FACTORS = 64
ITERATIONS = 15
REGULARIZATION = 0.05
ALPHA = 20.0


@dataclass
class AlsModel:
    model: AlternatingLeastSquares
    user_items: csr_matrix

    @classmethod
    def fit(
        cls,
        con: duckdb.DuckDBPyConnection,
        as_of: str,
        *,
        days: int = config.ALS_DAYS,
        seed: int = 42,
    ) -> AlsModel:
        users, items, counts = con.execute(f"""
                SELECT customer_idx, article_idx, count(*)::FLOAT AS purchases
                FROM tx
                WHERE purchase_date >= DATE '{as_of}' - INTERVAL {days} DAY
                  AND purchase_date < DATE '{as_of}'
                GROUP BY ALL
                """).fetchnumpy().values()
        shape = (
            con.execute("SELECT count(*) FROM customer_map").fetchone()[0],
            con.execute("SELECT count(*) FROM article_map").fetchone()[0],
        )
        # Repeat purchases raise confidence with diminishing returns.
        user_items = csr_matrix(
            (np.log1p(counts).astype(np.float32), (users, items)), shape=shape
        )
        # implicit parallelizes itself; a second BLAS thread pool slows it down.
        with threadpool_limits(1, "blas"):
            model = AlternatingLeastSquares(
                factors=FACTORS,
                regularization=REGULARIZATION,
                alpha=ALPHA,
                iterations=ITERATIONS,
                random_state=seed,
                num_threads=0,
            )
            model.fit(user_items, show_progress=False)
        return cls(model, user_items)

    def has_history(self, users: np.ndarray) -> np.ndarray:
        return np.diff(self.user_items.indptr)[users] > 0

    def recommend(self, users: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
        with threadpool_limits(1, "blas"):
            return self.model.recommend(
                users,
                self.user_items[users],
                N=n,
                filter_already_liked_items=False,
            )
