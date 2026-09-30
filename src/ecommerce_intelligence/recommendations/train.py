"""Fit and evaluate a frozen November 2011 BM25 item-to-item recommender."""

from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from ecommerce_intelligence.data.source import sha256_file
from ecommerce_intelligence.recommendations.evaluation import evaluate_recommender
from ecommerce_intelligence.recommendations.interactions import (
    CUTOFF,
    HOLDOUT_END,
    MAX_INVOICE_WEIGHT,
    MERCHANDISE_CODE_PATTERN,
    build_interactions,
    merchandise_purchases,
    split_purchases,
)
from ecommerce_intelligence.recommendations.recommender import (
    NEIGHBORS,
    ProductRecommender,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def run(*, project_root: Path) -> dict:
    purchase_path = project_root / "data" / "interim" / "purchases.parquet"
    quality_path = project_root / "reports" / "data_quality.json"
    if not purchase_path.is_file() or not quality_path.is_file():
        raise FileNotFoundError("Run the data pipeline first")
    quality = json.loads(quality_path.read_text(encoding="utf-8"))
    purchases = pd.read_parquet(purchase_path)
    merchandise = merchandise_purchases(purchases)
    train, holdout = split_purchases(
        merchandise,
        observation_end=pd.Timestamp(quality["source_last_date"]),
    )
    interactions = build_interactions(train)
    with threadpool_limits(limits=1, user_api="blas"):
        recommender = ProductRecommender.fit(interactions)
    metrics, predictions, examples = evaluate_recommender(recommender, train, holdout)

    artifacts = project_root / "artifacts" / "serving"
    processed = project_root / "data" / "processed"
    reports = project_root / "reports"
    recommender.save(artifacts)
    processed.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(
        processed / "recommender_holdout_predictions.parquet", index=False
    )
    _write_json(processed / "recommendation_examples.json", examples)

    restored = ProductRecommender.load(artifacts)
    for customer_id in (
        next(iter(recommender.customer_ids)),
        next(iter(predictions["customer_id"])),
        "NEW_CUSTOMER_DEMO",
    ):
        original = recommender.recommend_item_ids(customer_id)
        reloaded = restored.recommend_item_ids(customer_id)
        if [item for item, _, _ in original] != [item for item, _, _ in reloaded]:
            raise ValueError("Saved recommender changed product ranking")
        if not np.allclose(
            [score or 0.0 for _, score, _ in original],
            [score or 0.0 for _, score, _ in reloaded],
            atol=1e-6,
        ):
            raise ValueError("Saved recommender changed ranking scores")

    schema = {
        "algorithm": "Implicit BM25Recommender item-to-item",
        "implicit_version": version("implicit"),
        "model_as_of": CUTOFF.date().isoformat(),
        "holdout_end_exclusive": HOLDOUT_END.date().isoformat(),
        "catalog_policy": f"stock_code matches {MERCHANDISE_CODE_PATTERN}; nonempty description",
        "interaction_weight": f"distinct customer-product invoice count capped at {MAX_INVOICE_WEIGHT}",
        "neighbors_per_item": NEIGHBORS,
        "seen_item_policy": "exclude every product bought before model_as_of",
        "fallback": "unique-buyer popularity from pre-cutoff purchases only",
        "purchase_data_sha256": sha256_file(purchase_path),
    }
    _write_json(artifacts / "recommender_schema.json", schema)
    report = {
        **schema,
        "valid_purchase_lines": len(purchases),
        "merchandise_purchase_lines": len(merchandise),
        "non_merchandise_lines_excluded": len(purchases) - len(merchandise),
        "train_purchase_lines": len(train),
        "holdout_purchase_lines": len(holdout),
        "train_customers": len(interactions.customer_ids),
        "catalog_products": len(interactions.stock_codes),
        "interaction_pairs": interactions.pair_count,
        "evaluation": metrics,
    }
    _write_json(reports / "recommender_metrics.json", report)
    print(
        f"BM25 Recall@5 {metrics['bm25']['recall_at_5']:.4f}; "
        f"catalog coverage {metrics['bm25']['catalog_coverage_at_5']:.4f} "
        f"on {metrics['eligible_users']:,} eligible November customers."
    )
    return report


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    run(project_root=project_root)


if __name__ == "__main__":
    main()
