"""November 2011 next-new-product evaluation with a frozen October catalog."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ecommerce_intelligence.recommendations.recommender import ProductRecommender


def evaluate_recommender(
    recommender: ProductRecommender,
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    *,
    k: int = 5,
) -> tuple[dict, pd.DataFrame, dict]:
    train_pairs = train.loc[:, ["customer_id", "stock_code"]].drop_duplicates()
    future_pairs = holdout.loc[:, ["customer_id", "stock_code"]].drop_duplicates()
    novel = future_pairs.merge(
        train_pairs.assign(previously_purchased=True),
        on=["customer_id", "stock_code"],
        how="left",
    )
    novel = novel.loc[novel["previously_purchased"].isna()].drop(
        columns="previously_purchased"
    )
    known_customer = novel["customer_id"].isin(recommender.customer_index)
    known_product = novel["stock_code"].isin(recommender.stock_codes)
    relevant = novel.loc[known_customer & known_product]
    truth = relevant.groupby("customer_id")["stock_code"].agg(set).to_dict()
    cold_relevant = novel.loc[~known_customer & known_product]
    cold_truth = cold_relevant.groupby("customer_id")["stock_code"].agg(set).to_dict()
    if not truth:
        raise ValueError(
            "No existing customers bought a new catalog product in holdout"
        )

    rows = []
    bm25_hits = 0
    popular_hits = 0
    target_total = 0
    bm25_hit_users = 0
    popular_hit_users = 0
    bm25_recall = []
    popular_recall = []
    recommended_items = set()
    popular_items = set()
    fallback_slots = 0
    successful_example = None

    for customer_id in sorted(truth):
        targets = truth[customer_id]
        seen = recommender.seen_item_ids(customer_id)
        ranked = recommender.recommend_item_ids(customer_id, k=k)
        baseline = recommender.popular_only_item_ids(customer_id, k=k)
        predicted = [item_id for item_id, _, _ in ranked]
        if len(predicted) != k or len(predicted) != len(set(predicted)):
            raise ValueError("Recommendations are incomplete or duplicated")
        if any(item in seen for item in predicted + baseline):
            raise ValueError("A previously purchased product was recommended")

        target_total += len(targets)
        hits = 0
        for rank, (item_id, score, source) in enumerate(ranked, 1):
            stock_code = recommender.stock_codes[item_id]
            hit = stock_code in targets
            hits += int(hit)
            fallback_slots += int(source == "popular_fallback")
            recommended_items.add(item_id)
            rows.append(
                {
                    "customer_id": customer_id,
                    "rank": rank,
                    "stock_code": stock_code,
                    "score": score,
                    "source": source,
                    "heldout_hit": hit,
                }
            )
        baseline_hits = sum(
            recommender.stock_codes[item_id] in targets for item_id in baseline
        )
        popular_items.update(baseline)
        bm25_hits += hits
        popular_hits += baseline_hits
        bm25_hit_users += int(hits > 0)
        popular_hit_users += int(baseline_hits > 0)
        bm25_recall.append(hits / len(targets))
        popular_recall.append(baseline_hits / len(targets))
        if successful_example is None and hits:
            successful_example = {
                "customer_id": customer_id,
                "relevant_new_products": sorted(targets),
                "recommendations": recommender.recommend(customer_id, k=k),
                "hit_count": hits,
            }

    users = len(truth)
    catalog_size = len(recommender.stock_codes)
    predictions = pd.DataFrame(rows)
    cold_recall = []
    cold_hit_users = 0
    cold_example = None
    for customer_id in sorted(cold_truth):
        targets = cold_truth[customer_id]
        recommended = recommender.popular_only_item_ids(customer_id, k=k)
        hits = sum(recommender.stock_codes[item] in targets for item in recommended)
        cold_recall.append(hits / len(targets))
        cold_hit_users += int(hits > 0)
        if cold_example is None and hits:
            cold_example = {
                "customer_id": customer_id,
                "relevant_new_products": sorted(targets),
                "recommendations": recommender.recommend(customer_id, k=k),
                "hit_count": hits,
            }
    metrics = {
        "k": k,
        "eligible_users": users,
        "eligible_target_pairs": target_total,
        "eligible_target_products": int(relevant["stock_code"].nunique()),
        "train_customer_product_pairs": len(train_pairs),
        "holdout_customer_product_pairs": len(future_pairs),
        "novel_holdout_pairs": len(novel),
        "holdout_new_customers": int(
            future_pairs.loc[
                ~future_pairs["customer_id"].isin(recommender.customer_index),
                "customer_id",
            ].nunique()
        ),
        "novel_pairs_with_cold_products": int((~known_product).sum()),
        "bm25": {
            "recall_at_5": float(np.mean(bm25_recall)),
            "micro_recall_at_5": bm25_hits / target_total,
            "hit_rate_at_5": bm25_hit_users / users,
            "catalog_coverage_at_5": len(recommended_items) / catalog_size,
            "unique_products_recommended": len(recommended_items),
            "fallback_slot_rate": fallback_slots / (users * k),
            "total_hits": bm25_hits,
        },
        "popular_baseline": {
            "recall_at_5": float(np.mean(popular_recall)),
            "micro_recall_at_5": popular_hits / target_total,
            "hit_rate_at_5": popular_hit_users / users,
            "catalog_coverage_at_5": len(popular_items) / catalog_size,
            "unique_products_recommended": len(popular_items),
            "total_hits": popular_hits,
        },
        "cold_start_fallback": {
            "eligible_new_customers": len(cold_truth),
            "recall_at_5": float(np.mean(cold_recall)) if cold_recall else None,
            "hit_rate_at_5": cold_hit_users / len(cold_truth) if cold_truth else None,
        },
    }
    examples = {
        "known_customer_with_holdout_hit": successful_example,
        "new_customer_with_holdout_hit": cold_example,
        "new_customer_popular_fallback": recommender.recommend(
            "NEW_CUSTOMER_DEMO", k=k
        ),
    }
    return metrics, predictions, examples
