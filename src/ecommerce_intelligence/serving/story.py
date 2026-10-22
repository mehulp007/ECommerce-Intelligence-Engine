"""A small, anonymous sample of active customers for the story page.

Each customer becomes a point: x is visits in the last 12 months (log scale),
y is how recently they bought (recent at the top), and z is predicted 180-day
value (log scale), all scaled to [-1, 1]. Customer IDs are not kept.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

SAMPLE_SIZE = 30_000
SEED = 7
POINTS_FILE = "story_customers.bin"
META_FILE = "story_customers.json"
RISK_BANDS = ["low", "medium", "high"]
BYTES_PER_CUSTOMER = 22


def _unit(values: np.ndarray, top: float) -> np.ndarray:
    """Scale non-negative values to [-1, 1], clipping above `top`."""
    return np.clip(values / max(top, 1e-9), 0.0, 1.0) * 2.0 - 1.0


def build_story_sample(scores: pd.DataFrame, out: Path) -> dict:
    """Write the sample from the customer scores; returns its metadata."""
    active = scores.loc[scores["churn_probability"].notna()]
    n = min(SAMPLE_SIZE, len(active))
    sample = active.sample(n=n, random_state=SEED).reset_index(drop=True)
    rng = np.random.default_rng(SEED)

    # Visit counts are whole numbers; a little jitter avoids flat sheets of points.
    visits = np.log1p(
        sample["visits_365d"].to_numpy(float) + rng.uniform(-0.45, 0.45, n)
    )
    value = np.log1p(sample["clv_180d"].to_numpy(float))
    recency = sample["recency_days"].to_numpy(float)
    positions = np.column_stack(
        [
            _unit(visits, float(np.quantile(visits, 0.995))),
            -_unit(recency, 365.0),
            _unit(value, float(np.quantile(value, 0.995))),
        ]
    ).astype("<f4")
    churn = sample["churn_probability"].to_numpy("<f4")
    clv = sample["clv_180d"].to_numpy("<f4")
    segment = sample["segment_id"].to_numpy(np.uint8)
    risk = sample["risk_band"].map({b: i for i, b in enumerate(RISK_BANDS)})
    risk = risk.to_numpy(np.uint8)

    out.mkdir(parents=True, exist_ok=True)
    with open(out / POINTS_FILE, "wb") as f:
        for array in (positions, churn, clv, segment, risk):
            f.write(array.tobytes())
    names = (
        active.drop_duplicates("segment_id")
        .set_index("segment_id")["segment_name"]
        .sort_index()
    )
    meta = {
        "count": n,
        "population": len(active),
        "seed": SEED,
        "axes": {
            "x": "visits in the last 12 months (log scale)",
            "y": "days since last purchase (recent at the top)",
            "z": "predicted 180-day value (log scale)",
        },
        "segments": {str(int(k)): v for k, v in names.items()},
        "risk_bands": RISK_BANDS,
        "fields": [
            {"name": "position", "dtype": "float32", "offset": 0, "length": n * 3},
            {
                "name": "churn_probability",
                "dtype": "float32",
                "offset": n * 12,
                "length": n,
            },
            {"name": "clv_180d", "dtype": "float32", "offset": n * 16, "length": n},
            {"name": "segment_id", "dtype": "uint8", "offset": n * 20, "length": n},
            {"name": "risk_band", "dtype": "uint8", "offset": n * 21, "length": n},
        ],
        "bytes": n * BYTES_PER_CUSTOMER,
    }
    (out / META_FILE).write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return meta
