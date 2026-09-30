from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

pytest.importorskip("implicit")

from ecommerce_intelligence.recommendations.interactions import (
    build_interactions,
    merchandise_purchases,
    split_purchases,
)
from ecommerce_intelligence.recommendations.recommender import ProductRecommender


def purchase(
    customer: str,
    code: str,
    invoice: str,
    date: str,
    *,
    quantity: int = 1,
) -> dict:
    return {
        "customer_id": customer,
        "stock_code": code,
        "description": f"Product {code}",
        "invoice_no": invoice,
        "invoice_date": pd.Timestamp(date),
        "quantity": quantity,
        "unit_price": 5.0,
        "line_total": 5.0 * quantity,
    }


def test_interactions_use_only_prior_merchandise_and_distinct_invoices() -> None:
    rows = pd.DataFrame(
        [
            purchase("A", "10001", "1", "2011-10-01"),
            purchase("A", "10001", "1", "2011-10-01"),
            purchase("A", "10001", "2", "2011-10-20", quantity=3),
            purchase("B", "10002", "3", "2011-10-25"),
            purchase("A", "10002", "4", "2011-11-01"),
            purchase("B", "POST", "5", "2011-10-25"),
        ]
    )
    merchandise = merchandise_purchases(rows)
    train, holdout = split_purchases(
        merchandise, observation_end=pd.Timestamp("2011-12-09")
    )
    interactions = build_interactions(train)

    assert len(merchandise) == 5
    assert len(train) == 4
    assert len(holdout) == 1
    assert interactions.customer_ids == ["A", "B"]
    assert interactions.stock_codes == ["10001", "10002"]
    assert interactions.matrix[0, 0] == 2
    assert interactions.matrix[0, 1] == 0
    assert interactions.pair_count == 2

    changed_holdout = pd.concat(
        [merchandise, pd.DataFrame([purchase("A", "10001", "6", "2011-11-15")])],
        ignore_index=True,
    )
    unchanged_train, _ = split_purchases(
        changed_holdout, observation_end=pd.Timestamp("2011-12-09")
    )
    assert (build_interactions(unchanged_train).matrix != interactions.matrix).nnz == 0


def test_incomplete_holdout_is_rejected() -> None:
    rows = merchandise_purchases(
        pd.DataFrame(
            [
                purchase("A", "10001", "1", "2011-10-01"),
                purchase("A", "10002", "2", "2011-11-01"),
            ]
        )
    )
    with pytest.raises(ValueError, match="complete holdout"):
        split_purchases(rows, observation_end=pd.Timestamp("2011-11-15"))


class FakeModel:
    def recommend(self, *_args, **_kwargs):
        return np.array([0, 2]), np.array([9.0, 4.0])


def test_seen_products_are_excluded_and_fallback_handles_new_customers() -> None:
    catalog = pd.DataFrame(
        {
            "item_id": range(5),
            "stock_code": [str(i) for i in range(5)],
            "description": [f"Product {i}" for i in range(5)],
            "median_unit_price": [5.0] * 5,
            "popularity_rank": range(1, 6),
        }
    )
    recommender = ProductRecommender(
        model=FakeModel(),
        user_items=csr_matrix([[1, 1, 0, 0, 0]]),
        customer_ids=["A"],
        stock_codes=[str(i) for i in range(5)],
        catalog=catalog,
    )

    known = recommender.recommend_item_ids("A", k=3)
    assert [item for item, _, _ in known] == [2, 3, 4]
    assert [source for _, _, source in known] == [
        "bm25",
        "popular_fallback",
        "popular_fallback",
    ]
    assert recommender.popular_only_item_ids("A", k=3) == [2, 3, 4]

    unknown = recommender.recommend("NEW", k=3)
    assert [item["stock_code"] for item in unknown] == ["0", "1", "2"]
    assert {item["source"] for item in unknown} == {"popular_fallback"}
