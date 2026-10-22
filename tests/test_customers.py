from __future__ import annotations

import hashlib
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from ecommerce_intelligence.customers import campaign
from ecommerce_intelligence.customers.clv import (
    BetaGeoModel,
    GammaGammaModel,
    LifetimeValueModel,
)
from ecommerce_intelligence.customers.features import (
    FEATURES,
    TARGET,
    add_index_groups,
    build_snapshot,
)
from ecommerce_intelligence.customers.pipeline import run
from ecommerce_intelligence.paths import ProjectPaths
from ecommerce_intelligence.recsys.data import open_context
from ecommerce_intelligence.search.embeddings import ArticleEmbeddings
from ecommerce_intelligence.search.encoder import normalize

GROUPS = ["Ladieswear", "Divided", "Menswear", "Baby/Children", "Sport"]


def _write(root, customers: int = 400) -> ProjectPaths:
    """Two years of synthetic shoppers with different rates and lifetimes."""
    paths = ProjectPaths(root)
    paths.parquet.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(3)
    # Real customer IDs are 64-character hex hashes.
    ids = [hashlib.sha256(f"c{i}".encode()).hexdigest() for i in range(customers)]
    articles = [f"{i:010d}" for i in range(200_000_000, 200_000_060)]
    pd.DataFrame(
        {
            "customer_id": ids,
            "fashion_news_subscriber": rng.random(customers) < 0.4,
            "active": rng.random(customers) < 0.4,
            "club_member_status": "active",
            "fashion_news_frequency": "none",
            "age": rng.integers(18, 70, customers).astype("int16"),
            "postal_code": "p",
        }
    ).to_parquet(paths.table("customers"))
    pd.DataFrame(
        {
            "article_id": articles,
            "product_code": [a[:7] for a in articles],
            "product_type_no": np.arange(60) % 6,
            "graphical_appearance_no": 1,
            "colour_group_code": np.arange(60) % 7,
            "perceived_colour_master_id": 1,
            "department_no": np.arange(60) % 4,
            "index_group_no": np.arange(60) % 5,
            "index_group_name": [GROUPS[i % 5] for i in range(60)],
            "section_no": 1,
            "garment_group_no": np.arange(60) % 3,
            "prod_name": [f"Item {i}" for i in range(60)],
            "product_type_name": [
                ["Dress", "Trousers", "Top"][i % 3] for i in range(60)
            ],
            "product_group_name": [
                ["Garment Full body", "Garment Lower body", "Garment Upper body"][i % 3]
                for i in range(60)
            ],
            "colour_group_name": [["Black", "White", "Red"][i % 3] for i in range(60)],
            "department_name": "Dept",
            "section_name": "Section",
            "garment_group_name": "Jersey",
            "detail_desc": "A synthetic product.",
        }
    ).to_parquet(paths.table("articles"))
    start, end = date(2018, 9, 20), date(2020, 9, 22)
    rows = []
    for c in range(customers):
        first = start + timedelta(days=int(rng.integers(0, 500)))
        last = first + timedelta(days=int(rng.exponential(250)))
        rate = rng.gamma(2.0, 1 / 40)  # Visits per day.
        day = first
        while day <= min(last, end):
            for _ in range(rng.integers(1, 4)):
                rows.append(
                    (
                        day,
                        ids[c],
                        articles[rng.integers(0, 60)],
                        rng.uniform(0.01, 0.08),
                        int(rng.integers(1, 3)),
                    )
                )
            day += timedelta(days=max(1, int(rng.exponential(1 / rate))))
    pd.DataFrame(
        rows,
        columns=[
            "purchase_date",
            "customer_id",
            "article_id",
            "price",
            "sales_channel_id",
        ],
    ).astype({"sales_channel_id": "int8"}).to_parquet(paths.table("transactions"))
    ArticleEmbeddings(
        article_ids=np.array(articles),
        image=normalize(rng.normal(size=(60, 16))),
        text=normalize(rng.normal(size=(60, 16))),
        has_image=np.ones(60, dtype=bool),
    ).save(paths.embeddings, {"model": "synthetic", "dim": 16})
    return paths


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory) -> ProjectPaths:
    return _write(tmp_path_factory.mktemp("customers"))


def test_snapshot_features_ignore_the_future_and_label_looks_forward(synthetic) -> None:
    con = open_context(synthetic)
    add_index_groups(con, synthetic)
    as_of = "2020-02-01"
    before = build_snapshot(con, as_of)
    con.execute(f"""
        INSERT INTO tx SELECT customer_idx, article_idx,
               DATE '{as_of}' + INTERVAL 10 DAY, price, online
        FROM tx WHERE purchase_date < DATE '{as_of}' LIMIT 300
        """)
    after = build_snapshot(con, as_of)
    pd.testing.assert_frame_equal(
        before[["customer_idx", *FEATURES]], after[["customer_idx", *FEATURES]]
    )
    # Customers who received a purchase 10 days later can no longer be inactive.
    changed = before[TARGET].to_numpy() != after[TARGET].to_numpy()
    assert changed.any() and (after.loc[changed, TARGET] == 0).all()
    assert before["recency_days"].min() >= 1 and before["recency_days"].max() <= 365
    shares = before[[c for c in FEATURES if c.startswith("share_")]].sum(axis=1)
    assert np.allclose(shares, 1.0, atol=1e-5)


def test_snapshot_label_matches_hand_computation(synthetic) -> None:
    con = open_context(synthetic)
    add_index_groups(con, synthetic)
    frame = build_snapshot(con, "2020-01-01")
    expected = con.execute("""
        SELECT customer_idx FROM tx
        WHERE purchase_date >= DATE '2020-01-01' AND purchase_date < DATE '2020-03-01'
        """).fetchnumpy()["customer_idx"]
    active_later = set(expected.tolist())
    assert (
        frame[TARGET] == (~frame["customer_idx"].isin(active_later)).astype(int)
    ).all()


def test_lifetime_value_behaves_sensibly(synthetic) -> None:
    from ecommerce_intelligence.customers.clv import rfm_summary

    con = open_context(synthetic)
    summary = rfm_summary(con, "2020-03-26")
    model = LifetimeValueModel.fit(summary)
    # A customer whose last visit was recent is more likely alive than one
    # with the same history who has been silent for a year.
    recent = model.visits.p_alive([10], [100], [102])
    silent = model.visits.p_alive([10], [50], [102])
    assert recent[0] > silent[0]
    assert model.visits.p_alive([0], [0], [50])[0] == 1.0
    more = model.visits.expected_visits(26, [20], [100], [102])
    fewer = model.visits.expected_visits(26, [2], [100], [102])
    assert more[0] > fewer[0] > 0
    # Gamma-Gamma shrinks a customer's mean spend toward the population mean.
    spend = model.spend
    population = spend.p * spend.v / (spend.q - 1)
    estimate = spend.expected_spend([1], [population * 3])[0]
    assert population < estimate < population * 3
    predicted = model.predict(summary)
    assert predicted["clv"].notna().all() and (predicted["clv"] >= 0).all()


@pytest.mark.parametrize("a, b", [(0.8, 2.4), (2.0, 5.0)])
def test_beta_geo_expectation_matches_simulation(a, b) -> None:
    """The closed form agrees with simulating the BG/NBD process, also for a < 1."""
    model = BetaGeoModel(r=0.6, alpha=4.0, a=a, b=b)
    rng = np.random.default_rng(0)
    n, horizon = 200_000, 26.0
    rate = rng.gamma(model.r, 1 / model.alpha, n)
    dropout = rng.beta(a, b, n)
    counts, clock, alive = np.zeros(n), np.zeros(n), np.ones(n, dtype=bool)
    while alive.any():
        clock = np.where(alive, clock + rng.exponential(1 / rate), clock)
        bought = alive & (clock <= horizon)
        counts += bought
        alive = bought & (rng.random(n) > dropout)
    expected = model.expected_visits(horizon, [0], [0], [1e-9])[0]
    assert counts.mean() == pytest.approx(expected, rel=0.02)


def test_beta_geo_rejects_singular_dropout() -> None:
    with pytest.raises(ValueError):
        BetaGeoModel(1.0, 1.0, 1.0, 1.0).expected_visits(10, [1], [5], [10])
    assert GammaGammaModel(2.0, 3.0, 1.0).expected_spend([0], [5.0])[
        0
    ] == pytest.approx(1.0)


def test_campaign_targets_highest_expected_value_within_budget() -> None:
    scores = pd.DataFrame(
        {
            "churn_probability": [0.9, 0.5, 0.8, np.nan, 0.1],
            "clv_180d": [1.0, 4.0, 0.1, 9.0, 1.0],
        }
    )
    result = campaign.plan(scores, budget=0.25, cost_per_contact=0.1, uplift=0.5)
    # Values: 0.45, 1.0, 0.04, (excluded), 0.05 -> best two within budget.
    assert result.contacted == 2
    assert list(result.customers.index) == [1, 0]
    assert result.expected_value_saved == pytest.approx(1.45)
    assert result.roi == pytest.approx((1.45 - 0.2) / 0.2)
    # Only customers worth more than a contact are chosen by default.
    generous = campaign.plan(scores, budget=10, cost_per_contact=0.1, uplift=0.5)
    assert generous.contacted == 2
    everyone = campaign.plan(
        scores, budget=10, cost_per_contact=0.1, uplift=0.5, profitable_only=False
    )
    assert everyone.contacted == 4
    curve = campaign.roi_curve(scores, cost_per_contact=0.1, uplift=0.5, points=4)
    assert curve["expected_value_saved"].is_monotonic_increasing
    with pytest.raises(ValueError):
        campaign.plan(scores, budget=1, cost_per_contact=0, uplift=0.5)


def test_segment_names_ignore_lifts_on_tiny_shares() -> None:
    from ecommerce_intelligence.customers.segments import BEHAVIOUR, name_segments

    rows = []
    for size, ladies, kids, spend in ((80, 0.95, 0.01, 0.2), (20, 0.70, 0.05, 2.0)):
        for _ in range(size):
            row = dict.fromkeys(BEHAVIOUR, 0.0)
            row.update(
                share_ladieswear=ladies,
                share_kids=kids,
                share_divided=1 - ladies - kids,
                spend_365d=spend,
                recency_days=30,
                online_share=0.5,
            )
            rows.append(row)
    frame = pd.DataFrame(rows)
    names = name_segments(frame, np.repeat([0, 1], [80, 20]))
    # Segment 1 has about 2.8x the average kids share, but kids are only 5% of it.
    assert "kids" not in names[1]
    assert names[1].startswith("High-value")


def test_pipeline_end_to_end(synthetic) -> None:
    report = run(synthetic, trials=2, log=lambda _: None)

    tuned = report["churn"]["test_results"]["xgboost_tuned"]
    assert 0 <= tuned["roc_auc"] <= 1
    assert {"logistic_recency_only", "xgboost_default"} <= set(
        report["churn"]["test_results"]
    )
    assert report["clv"]["holdout_results"]["bg_nbd_gamma_gamma"]["spend"]["mae"] >= 0
    assert report["segments"]["chosen_k"] in (5, 6, 7, 8)

    scores = pd.read_parquet(
        synthetic.root / "data" / "serving" / "customer_scores.parquet"
    )
    assert len(scores) == 400 and scores["customer_id"].is_unique
    active = scores["churn_probability"].notna()
    assert scores.loc[active, "churn_probability"].between(0, 1).all()
    assert (scores.loc[~active, "segment_id"] == -1).all()
    first = scores.loc[active].iloc[0]
    assert {first[f"driver{i}_feature"] for i in (1, 2, 3)} <= set(FEATURES)
    assert abs(first["driver1_log_odds"]) >= abs(first["driver2_log_odds"])
    assert set(scores.loc[active, "risk_band"]) <= {"low", "medium", "high"}
    assert (synthetic.artifacts / "customers" / "churn_xgboost.json").is_file()


def test_fits_recover_from_a_line_search_stop_but_still_report_failure(
    monkeypatch,
) -> None:
    from types import SimpleNamespace

    from ecommerce_intelligence.customers import clv

    rng = np.random.default_rng(3)
    summary = pd.DataFrame(
        {"frequency": rng.integers(1, 8, 400), "monetary": rng.gamma(4, 0.01, 400)}
    )
    real = clv.minimize
    calls = []

    def flaky(objective, x0, args, method, bounds):
        calls.append(method)
        result = real(objective, x0=x0, args=args, method=method, bounds=bounds)
        if method == "L-BFGS-B":  # Simulate ABNORMAL_TERMINATION_IN_LNSRCH.
            return SimpleNamespace(x=result.x, success=False, message="ABNORMAL")
        return result

    monkeypatch.setattr(clv, "minimize", flaky)
    model = clv.GammaGammaModel.fit(summary)
    assert calls == ["L-BFGS-B", "Powell"]
    assert np.isfinite([model.p, model.q, model.v]).all()

    def broken(objective, x0, args, method, bounds):
        return SimpleNamespace(x=np.asarray(x0), success=False, message="diverged")

    monkeypatch.setattr(clv, "minimize", broken)
    with pytest.raises(RuntimeError, match="Gamma-Gamma fit failed: diverged"):
        clv.GammaGammaModel.fit(summary)
