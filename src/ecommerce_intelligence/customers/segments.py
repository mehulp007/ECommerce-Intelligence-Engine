"""Behavioural and visual-style customer segments with data-driven names."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from ecommerce_intelligence.customers.features import INDEX_GROUPS

BEHAVIOUR = [
    "recency_days",
    "visits_365d",
    "spend_365d",
    "avg_visit_spend",
    "online_share",
    *INDEX_GROUPS.values(),
    "mean_price",
    "age",
]
LOG_COLUMNS = {"recency_days", "visits_365d", "spend_365d", "avg_visit_spend"}
STYLE_COMPONENTS = 8
STYLE_WEIGHT = 0.5
CANDIDATE_K = (5, 6, 7, 8)
FIT_SAMPLE = 200_000
SILHOUETTE_SAMPLE = 20_000
SEED = 42
MIN_CATEGORY_SHARE = 0.3
CATEGORY_LABELS = {
    "share_ladieswear": "womenswear",
    "share_divided": "Divided (young fashion)",
    "share_menswear": "menswear",
    "share_kids": "kids & family",
    "share_sport": "sportswear",
}


@dataclass
class SegmentModel:
    scaler: StandardScaler
    pca: PCA
    kmeans: KMeans
    medians: dict
    names: dict
    silhouette: dict

    def _matrix(self, frame: pd.DataFrame, style: np.ndarray) -> np.ndarray:
        return _matrix(frame, style, self.scaler, self.pca, self.medians)

    def predict(self, frame: pd.DataFrame, style: np.ndarray) -> np.ndarray:
        return self.kmeans.predict(self._matrix(frame, style))


def _behaviour(frame: pd.DataFrame, medians: dict) -> np.ndarray:
    values = frame[BEHAVIOUR].astype(np.float64).fillna(medians)
    for column in LOG_COLUMNS:
        values[column] = np.log1p(values[column].clip(lower=0))
    return values.to_numpy()


def _matrix(frame, style, scaler, pca, medians) -> np.ndarray:
    behaviour = scaler.transform(_behaviour(frame, medians))
    taste = pca.transform(style) / np.sqrt(pca.explained_variance_[:1])
    return np.hstack([behaviour, STYLE_WEIGHT * taste]).astype(np.float32)


def name_segments(frame: pd.DataFrame, labels: np.ndarray) -> dict[int, str]:
    """Describe each cluster by value tier, dominant category, and channel."""
    data = frame.assign(segment=labels)
    overall = data[BEHAVIOUR].median(numeric_only=True)
    share_means = data[list(CATEGORY_LABELS)].mean()
    names: dict[int, str] = {}
    for segment, group in data.groupby("segment"):
        spend_ratio = group["spend_365d"].median() / max(overall["spend_365d"], 1e-9)
        if group["recency_days"].median() > 180:
            value = "Lapsing"
        elif spend_ratio >= 2:
            value = "High-value"
        elif spend_ratio <= 0.75:
            value = "Occasional"
        else:
            value = "Regular"
        # A category names a segment only if it is over-represented and also a
        # large part of what the segment buys; a lift on a tiny share is noise.
        shares = group[list(CATEGORY_LABELS)].mean()
        lift = (shares / share_means).sort_values()
        top = lift.index[-1]
        category = (
            CATEGORY_LABELS[top]
            if lift.iloc[-1] >= 1.3 and shares[top] >= MIN_CATEGORY_SHARE
            else "multi-category"
        )
        online = group["online_share"].mean()
        channel = "online" if online >= 0.8 else "in-store" if online <= 0.3 else ""
        words = [value, category, channel, "shoppers"]
        names[int(segment)] = " ".join(w for w in words if w)
    seen: dict[str, int] = {}
    for segment in sorted(names):
        base = names[segment]
        seen[base] = seen.get(base, 0) + 1
        if seen[base] > 1:
            names[segment] = f"{base} ({seen[base]})"
    return names


def fit_segments(
    frame: pd.DataFrame, style: np.ndarray, *, log=print
) -> tuple[SegmentModel, np.ndarray]:
    rng = np.random.default_rng(SEED)
    sample = rng.choice(len(frame), size=min(FIT_SAMPLE, len(frame)), replace=False)
    medians = frame[BEHAVIOUR].median(numeric_only=True).to_dict()
    scaler = StandardScaler().fit(_behaviour(frame.iloc[sample], medians))
    pca = PCA(n_components=STYLE_COMPONENTS, random_state=SEED).fit(style[sample])
    matrix = _matrix(frame.iloc[sample], style[sample], scaler, pca, medians)
    scored = rng.choice(
        len(sample), size=min(SILHOUETTE_SAMPLE, len(sample)), replace=False
    )
    best, silhouettes = None, {}
    for k in CANDIDATE_K:
        kmeans = KMeans(n_clusters=k, n_init=5, random_state=SEED).fit(matrix)
        silhouettes[k] = float(
            silhouette_score(matrix[scored], kmeans.labels_[scored], random_state=SEED)
        )
        log(f"  k={k}: silhouette {silhouettes[k]:.3f}")
        if best is None or silhouettes[k] > silhouettes[best[0]]:
            best = (k, kmeans)
    labels = best[1].predict(_matrix(frame, style, scaler, pca, medians))
    names = name_segments(frame, labels)
    return SegmentModel(scaler, pca, best[1], medians, names, silhouettes), labels
