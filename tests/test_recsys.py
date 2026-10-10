from __future__ import annotations

import json
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from ecommerce_intelligence.paths import ProjectPaths
from ecommerce_intelligence.recsys import config
from ecommerce_intelligence.recsys.als import AlsModel
from ecommerce_intelligence.recsys.candidates import build_candidates
from ecommerce_intelligence.recsys.data import open_context, week_labels
from ecommerce_intelligence.recsys.features import FEATURES, build_features
from ecommerce_intelligence.recsys.metrics import (
    average_precision_at_k,
    map_at_k,
    recall_at_k,
)
from ecommerce_intelligence.recsys.pipeline import run


def test_average_precision_matches_competition_definition() -> None:
    assert average_precision_at_k(["a", "b", "c"], {"a", "c"}) == pytest.approx(
        (1 / 1 + 2 / 3) / 2
    )
    assert average_precision_at_k(["x", "a"], {"a"}) == 0.5
    assert average_precision_at_k(["a", "a"], {"a", "b"}) == 0.5  # No double credit.
    assert average_precision_at_k([], {"a"}) == 0.0
    many = {str(i) for i in range(20)}
    perfect = [str(i) for i in range(12)]
    assert average_precision_at_k(perfect, many) == 1.0  # Normalized by min(n, 12).


def test_map_scores_missing_predictions_as_zero() -> None:
    actuals = {1: {"a"}, 2: {"b"}}
    assert map_at_k({1: ["a"]}, actuals) == 0.5
    assert recall_at_k({1: ["a"]}, actuals) == 0.5


def _write_tables(root) -> ProjectPaths:
    paths = ProjectPaths(root)
    paths.parquet.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    customers = [f"c{i:03d}" for i in range(120)]
    articles = [f"{i:010d}" for i in range(100_000_000, 100_000_040)]
    pd.DataFrame(
        {
            "customer_id": customers,
            "age": rng.integers(18, 70, len(customers)).astype("int16"),
        }
    ).to_parquet(paths.table("customers"))
    pd.DataFrame(
        {
            "article_id": articles,
            "product_code": [a[:7] for a in articles],
            "product_type_no": np.arange(40) % 5,
            "graphical_appearance_no": 1,
            "colour_group_code": np.arange(40) % 7,
            "perceived_colour_master_id": 1,
            "department_no": np.arange(40) % 3,
            "index_group_no": np.arange(40) % 2,
            "section_no": 1,
            "garment_group_no": np.arange(40) % 4,
        }
    ).to_parquet(paths.table("articles"))
    start = date(2020, 6, 1)
    end = date(2020, 9, 23)
    days = (end - start).days
    rows = 6000
    # Each customer favours a few articles so collaborative signals exist.
    favourite = rng.integers(0, 40, size=(len(customers), 4))
    customer_ix = rng.integers(0, len(customers), rows)
    pick = np.where(
        rng.random(rows) < 0.6,
        favourite[customer_ix, rng.integers(0, 4, rows)],
        rng.integers(0, 40, rows),
    )
    pd.DataFrame(
        {
            "purchase_date": [
                start + timedelta(days=int(d)) for d in rng.integers(0, days, rows)
            ],
            "customer_id": [customers[i] for i in customer_ix],
            "article_id": [articles[i] for i in pick],
            "price": rng.uniform(0.01, 0.1, rows),
            "sales_channel_id": rng.integers(1, 3, rows).astype("int8"),
        }
    ).to_parquet(paths.table("transactions"))
    return paths


def test_features_use_only_purchases_before_as_of(tmp_path) -> None:
    as_of = "2020-09-16"
    con = open_context(_write_tables(tmp_path))
    con.register("t", pd.DataFrame({"customer_idx": np.arange(120, dtype=np.int32)}))
    con.execute("CREATE TABLE targets AS SELECT customer_idx FROM t")
    als = AlsModel.fit(con, as_of)
    build_candidates(con, as_of, als)
    before = build_features(con, as_of)

    # Purchases on or after the as-of date must not change anything.
    con.execute(f"""
        INSERT INTO tx
        SELECT customer_idx, (article_idx + 1) % 40, DATE '{as_of}' + INTERVAL 2 DAY,
               price, online
        FROM tx LIMIT 2000
        """)
    assert (AlsModel.fit(con, as_of).user_items != als.user_items).nnz == 0
    build_candidates(con, as_of, als)
    after = build_features(con, as_of)

    columns = ["customer_idx", "article_idx", *FEATURES]
    pd.testing.assert_frame_equal(before[columns], after[columns])
    assert not set(FEATURES) & {"customer_idx", "article_idx", "label"}


def test_week_labels_cover_exactly_seven_days(tmp_path) -> None:
    con = open_context(_write_tables(tmp_path))
    labels = week_labels(con, "2020-09-16")
    expected = con.execute("""
        SELECT count(DISTINCT (customer_idx, article_idx)) FROM tx
        WHERE purchase_date BETWEEN DATE '2020-09-16' AND DATE '2020-09-22'
        """).fetchone()[0]
    assert sum(len(v) for v in labels.values()) == expected


def test_pipeline_end_to_end_on_synthetic_data(tmp_path, monkeypatch) -> None:
    from ecommerce_intelligence.recsys import pipeline

    monkeypatch.setattr(pipeline, "INFERENCE_CHUNK", 50)
    paths = _write_tables(tmp_path)
    report = run(paths, ablations=True, score_all=True, log=lambda _: None)

    ranker_map = report["results"]["ranker"]["map_at_12"]
    assert 0 <= ranker_map <= 1
    assert set(report["results"]) >= {"ranker", "popular", "als"}
    assert "without_als" in report["ablations"]
    assert report["candidate_recall"]["all"] >= report["candidate_recall"]["popular"]

    recs = pd.read_parquet(paths.root / "data" / "serving" / "recommendations.parquet")
    assert recs["customer_id"].nunique() == 120
    assert recs.groupby("customer_id").size().max() == config.K
    assert not recs.duplicated(["customer_id", "article_id"]).any()
    assert set(recs["reason"]) <= set(config.REASONS.values())
    saved = json.loads((paths.reports / "recommender_metrics.json").read_text())
    assert saved["scoring"]["customers"] == 120
    assert (paths.artifacts / "recsys" / "ranker.txt").is_file()
