from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("implicit")

from ecommerce_intelligence.recommendations.recommender import ProductRecommender

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "recommender_metrics.json"


@pytest.mark.skipif(
    not REPORT.exists()
    or not (
        ROOT / "data" / "processed" / "recommender_holdout_predictions.parquet"
    ).exists(),
    reason="Run recommender training first",
)
def test_saved_recommender_and_holdout_results() -> None:
    metrics = json.loads(REPORT.read_text(encoding="utf-8"))["evaluation"]
    recommender = ProductRecommender.load(ROOT / "artifacts" / "serving")
    predictions = pd.read_parquet(
        ROOT / "data" / "processed" / "recommender_holdout_predictions.parquet"
    )

    assert recommender.user_items.shape == (
        len(recommender.customer_ids),
        len(recommender.stock_codes),
    )
    assert recommender.catalog["last_sold"].lt("2011-11-01").all()
    assert len(predictions) == metrics["eligible_users"] * 5
    assert predictions["heldout_hit"].sum() == metrics["bm25"]["total_hits"]
    assert (
        predictions["stock_code"].nunique()
        == metrics["bm25"]["unique_products_recommended"]
    )
    assert predictions.groupby("customer_id")[
        "heldout_hit"
    ].any().mean() == pytest.approx(metrics["bm25"]["hit_rate_at_5"])

    for customer_id in predictions["customer_id"].drop_duplicates().head(20):
        ranked = recommender.recommend_item_ids(customer_id)
        assert len(ranked) == 5
        assert not {item for item, _, _ in ranked} & recommender.seen_item_ids(
            customer_id
        )
        saved = predictions.loc[predictions["customer_id"].eq(customer_id)]
        assert [recommender.stock_codes[item] for item, _, _ in ranked] == saved[
            "stock_code"
        ].tolist()

    cold = recommender.recommend("NEW_CUSTOMER_DEMO")
    assert len(cold) == 5
    assert {product["source"] for product in cold} == {"popular_fallback"}
