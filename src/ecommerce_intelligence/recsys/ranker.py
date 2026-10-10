"""Stage two: a LightGBM LambdaRank model that orders each customer's candidates."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from ecommerce_intelligence.recsys.features import CATEGORICAL_FEATURES

PARAMS = {
    "objective": "lambdarank",
    "metric": "map",
    "eval_at": [12],
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_data_in_leaf": 200,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambdarank_truncation_level": 24,
    "verbosity": -1,
    "num_threads": 0,
    "seed": 42,
    "deterministic": True,
    "force_row_wise": True,
}
MAX_ROUNDS = 1000
EARLY_STOPPING = 50


def with_positives(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep customers with at least one purchased candidate; others carry no signal."""
    has_positive = frame.groupby("customer_idx")["label"].transform("max") > 0
    return frame.loc[has_positive]


def _dataset(frame: pd.DataFrame, features: list[str], **kwargs) -> lgb.Dataset:
    frame = frame.sort_values(["customer_idx", "article_idx"], kind="stable")
    groups = frame.groupby("customer_idx", sort=False).size().to_numpy()
    return lgb.Dataset(
        frame[features],
        label=frame["label"].to_numpy(),
        group=groups,
        categorical_feature=[c for c in CATEGORICAL_FEATURES if c in features],
        free_raw_data=True,
        **kwargs,
    )


def fit(
    train: pd.DataFrame,
    features: list[str],
    *,
    valid: pd.DataFrame | None = None,
    rounds: int = MAX_ROUNDS,
) -> lgb.Booster:
    """Train on labelled candidates; early-stop on `valid` when provided."""
    train_set = _dataset(with_positives(train), features)
    if valid is None:
        return lgb.train(PARAMS, train_set, num_boost_round=rounds)
    valid_set = _dataset(with_positives(valid), features, reference=train_set)
    return lgb.train(
        PARAMS,
        train_set,
        num_boost_round=rounds,
        valid_sets=[valid_set],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False)],
    )


def top_k(frame: pd.DataFrame, scores: np.ndarray, k: int) -> pd.DataFrame:
    """Return the k highest-scoring rows per customer with a 1-based rank."""
    ranked = frame.assign(score=scores).sort_values(
        ["customer_idx", "score", "article_idx"], ascending=[True, False, True]
    )
    ranked["rank"] = ranked.groupby("customer_idx").cumcount() + 1
    return ranked.loc[ranked["rank"] <= k]
