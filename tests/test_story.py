from __future__ import annotations

import json

import numpy as np
import pandas as pd

from ecommerce_intelligence.serving.story import (
    BYTES_PER_CUSTOMER,
    META_FILE,
    POINTS_FILE,
    RISK_BANDS,
    build_story_sample,
)


def _read(directory):
    meta = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
    raw = (directory / POINTS_FILE).read_bytes()
    n = meta["count"]
    fields = {f["name"]: f for f in meta["fields"]}
    return (
        meta,
        raw,
        {
            "position": np.frombuffer(raw, "<f4", n * 3, 0).reshape(n, 3),
            "churn": np.frombuffer(
                raw, "<f4", n, fields["churn_probability"]["offset"]
            ),
            "clv": np.frombuffer(raw, "<f4", n, fields["clv_180d"]["offset"]),
            "segment": np.frombuffer(raw, np.uint8, n, fields["segment_id"]["offset"]),
            "risk": np.frombuffer(raw, np.uint8, n, fields["risk_band"]["offset"]),
        },
    )


def test_sample_holds_only_active_customers_and_no_ids(world) -> None:
    out = world.root / "data" / "serving"
    scores = pd.read_parquet(out / "customer_scores.parquet")
    active = scores.loc[scores["churn_probability"].notna()]
    meta, raw, cols = _read(out)
    assert meta["count"] == min(30_000, len(active)) and meta["population"] == len(
        active
    )
    assert len(raw) == meta["bytes"] == meta["count"] * BYTES_PER_CUSTOMER
    assert np.all(np.abs(cols["position"]) <= 1.0 + 1e-6)
    # Every sampled point is a real active customer's probability and value.
    pairs = set(
        zip(
            active["churn_probability"].round(5),
            active["clv_180d"].round(5),
            strict=True,
        )
    )
    sampled = zip(cols["churn"].round(5), cols["clv"].round(5), strict=True)
    assert all((float(p), float(v)) in pairs for p, v in sampled)
    assert set(cols["risk"]) <= set(range(len(RISK_BANDS)))
    assert {str(s) for s in cols["segment"]} <= set(meta["segments"])
    for customer_id in active["customer_id"].head(50):
        assert customer_id.encode() not in raw


def test_axes_put_recent_frequent_valuable_customers_high(tmp_path) -> None:
    scores = pd.DataFrame(
        {
            "customer_id": ["a", "b"],
            "churn_probability": [0.2, 0.9],
            "risk_band": ["low", "high"],
            "segment_id": [0, 1],
            "segment_name": ["Loyal", "Occasional"],
            "visits_365d": [40.0, 1.0],
            "recency_days": [2.0, 360.0],
            "clv_180d": [3.0, 0.01],
        }
    )
    build_story_sample(scores, tmp_path)
    _, _, cols = _read(tmp_path)
    loyal, occasional = cols["position"]
    order = list(cols["risk"])
    loyal, occasional = (loyal, occasional) if order[0] == 0 else (occasional, loyal)
    assert loyal[0] > occasional[0]  # More visits
    assert loyal[1] > occasional[1]  # More recent
    assert loyal[2] > occasional[2]  # More value
