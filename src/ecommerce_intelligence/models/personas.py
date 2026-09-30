"""Four RFM personas fitted on the last training-month cohort."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

RFM_FEATURES = ("recency_days", "frequency_180d", "monetary_180d")
PERSONA_COUNT = 4
RANDOM_STATE = 42


def fit_personas(train: pd.DataFrame) -> tuple[Pipeline, pd.DataFrame, dict]:
    last_month = train["snapshot_date"].max()
    cohort = train.loc[train["snapshot_date"].eq(last_month)].copy()
    if len(cohort) < PERSONA_COUNT:
        raise ValueError("Not enough customers to fit four personas")
    if cohort["customer_id"].duplicated().any():
        raise ValueError("Persona cohort contains repeated customers")

    pipeline = Pipeline(
        steps=[
            ("log1p", FunctionTransformer(np.log1p, validate=True)),
            ("scale", StandardScaler()),
            (
                "kmeans",
                KMeans(n_clusters=PERSONA_COUNT, n_init=20, random_state=RANDOM_STATE),
            ),
        ]
    )
    features = cohort.loc[:, list(RFM_FEATURES)]
    cohort["persona_id"] = pipeline.fit_predict(features)
    profile = cohort.groupby("persona_id", sort=True).agg(
        customers=("customer_id", "nunique"),
        median_recency_days=("recency_days", "median"),
        median_frequency_180d=("frequency_180d", "median"),
        median_monetary_180d=("monetary_180d", "median"),
        mean_monetary_180d=("monetary_180d", "mean"),
        inactivity_rate_30d=("inactive_next_30d", "mean"),
    )
    profile["share"] = profile["customers"] / len(cohort)
    profile["persona_name"] = profile.index.map(_name_personas(profile))
    profile = profile.reset_index()

    transformed = pipeline[:-1].transform(features)
    metrics = {
        "silhouette_score": float(
            silhouette_score(
                transformed,
                cohort["persona_id"],
                sample_size=2000,
                random_state=RANDOM_STATE,
            )
        ),
        "davies_bouldin_score": float(
            davies_bouldin_score(transformed, cohort["persona_id"])
        ),
        "inertia": float(pipeline.named_steps["kmeans"].inertia_),
        "fit_customers": len(cohort),
    }
    return pipeline, profile, metrics


def _name_personas(profile: pd.DataFrame) -> dict[int, str]:
    remaining = {int(value) for value in profile.index}
    recency = int(profile.loc[list(remaining), "median_recency_days"].idxmax())
    remaining.remove(recency)
    value = int(profile.loc[list(remaining), "median_monetary_180d"].idxmax())
    remaining.remove(value)
    frequency = int(profile.loc[list(remaining), "median_frequency_180d"].idxmax())
    remaining.remove(frequency)
    occasional = remaining.pop()
    return {
        recency: "At-risk",
        value: "High-value",
        frequency: "Repeat buyers",
        occasional: "Occasional",
    }
