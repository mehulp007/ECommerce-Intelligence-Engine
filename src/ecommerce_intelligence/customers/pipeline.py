"""Train and evaluate churn, lifetime value, and segments; score every customer."""

from __future__ import annotations

import gc
import json
import time
from collections.abc import Callable
from importlib.metadata import version

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ecommerce_intelligence.customers import campaign, churn
from ecommerce_intelligence.customers.clv import (
    HORIZON_DAYS,
    LifetimeValueModel,
    rfm_summary,
)
from ecommerce_intelligence.customers.features import (
    FEATURES,
    TARGET,
    add_index_groups,
    build_snapshot,
)
from ecommerce_intelligence.customers.segments import fit_segments
from ecommerce_intelligence.data.duck import sql_path
from ecommerce_intelligence.evaluation.classification import classification_metrics
from ecommerce_intelligence.paths import ProjectPaths
from ecommerce_intelligence.recsys import config as recsys_config
from ecommerce_intelligence.recsys.data import open_context
from ecommerce_intelligence.recsys.style import StyleModel

SCORING_AS_OF = recsys_config.INFERENCE_AS_OF.isoformat()
CLV_CALIBRATION_END = "2020-03-26"
CLV_HOLDOUT_DAYS = 181  # 2020-03-26 through 2020-09-22, the last day of data.
FINAL_TRAIN = churn.TRAIN_SNAPSHOTS + churn.CALIBRATION_SNAPSHOTS + ("2020-05-01",)
FINAL_CALIBRATION = ("2020-06-01", "2020-07-01")
RISK_BANDS = ((0.5, "low"), (0.75, "medium"), (1.01, "high"))
DEFAULT_CAMPAIGN = {"cost_per_contact": 0.005, "uplift": 0.10, "budget": 500.0}

Log = Callable[[str], None]


def _platt(raw_calibration: np.ndarray, labels: np.ndarray) -> Callable:
    calibrator = LogisticRegression().fit(churn.logit(raw_calibration), labels)
    return lambda raw: calibrator.predict_proba(churn.logit(raw))[:, 1]


def _rounded(metrics: dict) -> dict:
    return {k: round(v, 5) if isinstance(v, float) else v for k, v in metrics.items()}


def _calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 10) -> list[dict]:
    edges = np.quantile(p, np.linspace(0, 1, bins + 1))
    index = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, bins - 1)
    return [
        {
            "bin": int(b),
            "mean_predicted": round(float(p[index == b].mean()), 4),
            "observed_rate": round(float(y[index == b].mean()), 4),
            "customers": int((index == b).sum()),
        }
        for b in range(bins)
        if (index == b).any()
    ]


def _bands(probability: np.ndarray) -> np.ndarray:
    limits = np.array([limit for limit, _ in RISK_BANDS])
    names = np.array([name for _, name in RISK_BANDS])
    return names[np.searchsorted(limits, probability, side="right")]


DRIVERS = 3
DRIVER_COLUMNS = [
    f"driver{rank}_{part}"
    for rank in range(1, DRIVERS + 1)
    for part in ("feature", "value", "log_odds")
]


def _drivers(model: churn.CalibratedChurnModel, frame: pd.DataFrame) -> pd.DataFrame:
    """The three features that moved each customer's raw log-odds the most."""
    contributions = model.contributions(frame)[:, :-1]
    order = np.argsort(-np.abs(contributions), axis=1)[:, :DRIVERS]
    values = frame[FEATURES].to_numpy()
    names = np.array(FEATURES)
    columns = {}
    for rank in range(DRIVERS):
        picked = order[:, rank : rank + 1]
        columns[f"driver{rank + 1}_feature"] = names[picked[:, 0]]
        columns[f"driver{rank + 1}_value"] = np.take_along_axis(values, picked, 1)[:, 0]
        columns[f"driver{rank + 1}_log_odds"] = np.take_along_axis(
            contributions, picked, 1
        )[:, 0]
    return pd.DataFrame(columns, index=frame.index)[DRIVER_COLUMNS]


def evaluate_churn(
    snapshots: dict[str, pd.DataFrame], *, trials: int, log: Log
) -> tuple[dict, dict]:
    train = churn.sampled(pd.concat([snapshots[d] for d in churn.TRAIN_SNAPSHOTS]))
    calibration = pd.concat([snapshots[d] for d in churn.CALIBRATION_SNAPSHOTS])
    test = pd.concat([snapshots[d] for d in churn.TEST_SNAPSHOTS], ignore_index=True)
    y_test = test[TARGET].to_numpy()

    params = churn.tune(
        train.loc[train["customer_idx"] % 10 == 0],
        calibration.loc[calibration["customer_idx"] % 5 == 0],
        trials=trials,
        log=log,
    )
    tuned = churn.fit_calibrated(params, train, calibration)
    default = churn.fit_calibrated(
        {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.1}, train, calibration
    )

    probabilities = {
        "xgboost_tuned": tuned.predict(test),
        "xgboost_default": default.predict(test),
    }
    for name, features in (
        ("logistic_all_features", None),
        ("logistic_recency_only", ["recency_days"]),
    ):
        model = churn.logistic_baseline(features).fit(train, train[TARGET])
        to_probability = _platt(
            model.predict_proba(calibration)[:, 1], calibration[TARGET]
        )
        probabilities[name] = to_probability(model.predict_proba(test)[:, 1])

    results = {
        name: _rounded(classification_metrics(y_test, p))
        for name, p in probabilities.items()
    }
    results["xgboost_tuned_uncalibrated"] = _rounded(
        classification_metrics(y_test, tuned.raw(test))
    )
    for name, value in results.items():
        log(
            f"  {name:28s} ROC AUC {value['roc_auc']:.4f}  "
            f"AP {value['average_precision']:.4f}  Brier {value['brier']:.4f}"
        )

    best = probabilities["xgboost_tuned"]
    by_month = {}
    for date, group in test.groupby("snapshot_date"):
        rows = group.index.to_numpy()
        by_month[date.date().isoformat()] = _rounded(
            classification_metrics(y_test[rows], best[rows])
        )

    sample = test.sample(n=min(100_000, len(test)), random_state=churn.SEED)
    importance = pd.Series(
        np.abs(tuned.contributions(sample)[:, :-1]).mean(axis=0), index=FEATURES
    ).sort_values(ascending=False)
    report = {
        "target": f"no purchase in the {60} days after the snapshot (inactivity proxy)",
        "population": "customers with a purchase in the 365 days before the snapshot",
        "split": {
            "train_snapshots": list(churn.TRAIN_SNAPSHOTS),
            "calibration_snapshots": list(churn.CALIBRATION_SNAPSHOTS),
            "test_snapshots": list(churn.TEST_SNAPSHOTS),
            "train_rows_sampled": len(train),
            "calibration_rows": len(calibration),
            "test_rows": len(test),
        },
        "prevalence_by_snapshot": {
            date: {
                "customers": len(frame),
                "inactive_rate": round(float(frame[TARGET].mean()), 4),
            }
            for date, frame in snapshots.items()
        },
        "tuned_params": params,
        "test_results": results,
        "test_by_month": by_month,
        "calibration_deciles": _calibration_table(y_test, best),
        "mean_abs_log_odds_contribution": {
            k: round(float(v), 4) for k, v in importance.items()
        },
    }
    return report, params


def evaluate_clv(con, *, log: Log) -> dict:
    summary = rfm_summary(con, CLV_CALIBRATION_END)
    model = LifetimeValueModel.fit(summary)
    predicted = model.predict(summary, days=CLV_HOLDOUT_DAYS)
    actual = con.execute(f"""
        SELECT customer_idx, count(DISTINCT purchase_date) AS visits,
               sum(price) AS spend
        FROM tx WHERE purchase_date >= DATE '{CLV_CALIBRATION_END}'
        GROUP BY customer_idx
        """).df()
    previous = con.execute(f"""
        SELECT customer_idx, count(DISTINCT purchase_date) AS prev_visits,
               sum(price) AS prev_spend
        FROM tx
        WHERE purchase_date >= DATE '{CLV_CALIBRATION_END}'
                                - INTERVAL {CLV_HOLDOUT_DAYS} DAY
          AND purchase_date < DATE '{CLV_CALIBRATION_END}'
        GROUP BY customer_idx
        """).df()
    frame = (
        summary.merge(predicted, on="customer_idx")
        .merge(actual, on="customer_idx", how="left")
        .merge(previous, on="customer_idx", how="left")
        .fillna({"visits": 0, "spend": 0, "prev_visits": 0, "prev_spend": 0})
    )
    weeks = CLV_HOLDOUT_DAYS / 7
    lifetime_rate = (frame["frequency"] + 1) / np.maximum(frame["age"], 1)
    candidates = {
        "bg_nbd_gamma_gamma": (frame["expected_visits"], frame["clv"]),
        "previous_period": (frame["prev_visits"], frame["prev_spend"]),
        "lifetime_average_rate": (
            lifetime_rate * weeks,
            lifetime_rate * weeks * frame["monetary"],
        ),
    }

    def score(pred, truth):
        pred, truth = np.asarray(pred, float), np.asarray(truth, float)
        return {
            "mae": round(float(np.mean(np.abs(pred - truth))), 5),
            "rmse": round(float(np.sqrt(np.mean((pred - truth) ** 2))), 5),
            "correlation": round(float(np.corrcoef(pred, truth)[0, 1]), 4),
            "total_predicted": round(float(pred.sum()), 1),
            "total_actual": round(float(truth.sum()), 1),
        }

    results = {
        name: {"visits": score(v, frame["visits"]), "spend": score(s, frame["spend"])}
        for name, (v, s) in candidates.items()
    }
    for name, value in results.items():
        spend = value["spend"]
        log(
            f"  {name:24s} spend MAE {spend['mae']:.4f}  "
            f"corr {spend['correlation']:.3f}  total {spend['total_predicted']:,.0f} "
            f"vs actual {spend['total_actual']:,.0f}"
        )
    deciles = pd.qcut(frame["clv"].rank(method="first"), 10, labels=False)
    return {
        "calibration_end": CLV_CALIBRATION_END,
        "holdout_days": CLV_HOLDOUT_DAYS,
        "customers": len(frame),
        "params": model.params(),
        "holdout_results": results,
        "spend_by_predicted_decile": [
            {
                "decile": int(d),
                "mean_predicted": round(float(g["clv"].mean()), 4),
                "mean_actual": round(float(g["spend"].mean()), 4),
            }
            for d, g in frame.groupby(deciles)
        ],
    }


def run(
    paths: ProjectPaths, *, trials: int = churn.TUNING_TRIALS, log: Log = print
) -> dict:
    started = time.perf_counter()
    timings: dict[str, float] = {}

    def lap(name: str, since: float) -> float:
        timings[name] = round(time.perf_counter() - since, 1)
        log(f"{name}: {timings[name]:.1f}s")
        return time.perf_counter()

    tick = time.perf_counter()
    con = open_context(paths)
    add_index_groups(con, paths)
    dates = churn.TRAIN_SNAPSHOTS + churn.CALIBRATION_SNAPSHOTS + churn.TEST_SNAPSHOTS
    snapshots = {}
    for date in dates:
        snapshots[date] = build_snapshot(con, date)
        tick = lap(f"snapshot {date}: {len(snapshots[date]):,} active customers", tick)

    churn_report, params = evaluate_churn(snapshots, trials=trials, log=log)
    tick = lap("churn evaluation", tick)

    final_model = churn.fit_calibrated(
        params,
        churn.sampled(pd.concat([snapshots[d] for d in FINAL_TRAIN])),
        pd.concat([snapshots[d] for d in FINAL_CALIBRATION]),
    )
    snapshots.clear()
    gc.collect()
    tick = lap("final churn model", tick)

    clv_report = evaluate_clv(con, log=log)
    tick = lap("lifetime value holdout", tick)

    current = build_snapshot(con, SCORING_AS_OF, label=False)
    current["churn_probability"] = final_model.predict(current)
    current["risk_band"] = _bands(current["churn_probability"].to_numpy())
    current[DRIVER_COLUMNS] = _drivers(final_model, current)
    tick = lap(f"score churn for {len(current):,} active customers", tick)

    summary = rfm_summary(con, SCORING_AS_OF)
    clv_model = LifetimeValueModel.fit(summary)
    value = clv_model.predict(summary, days=HORIZON_DAYS)
    tick = lap("lifetime value", tick)

    style = StyleModel.load(paths, con)
    con.register("active_frame", current[["customer_idx"]])
    con.execute(
        "CREATE OR REPLACE TABLE targets AS SELECT customer_idx FROM active_frame"
    )
    con.unregister("active_frame")
    style.prepare(con, SCORING_AS_OF, days=365)
    rows = np.searchsorted(style.customers, current["customer_idx"].to_numpy())
    if not np.array_equal(style.customers[rows], current["customer_idx"].to_numpy()):
        raise ValueError("Every active customer needs a style profile")
    segment_model, labels = fit_segments(current, style.profiles[rows], log=log)
    current["segment_id"] = labels.astype(np.int16)
    current["segment_name"] = current["segment_id"].map(segment_model.names)
    tick = lap("segments", tick)

    scores = _write_scores(con, paths, current, value)
    tick = lap("write customer scores", tick)

    active_scores = scores.dropna(subset=["churn_probability"])
    plan = campaign.plan(active_scores, **DEFAULT_CAMPAIGN)
    curve = campaign.roi_curve(
        active_scores,
        cost_per_contact=DEFAULT_CAMPAIGN["cost_per_contact"],
        uplift=DEFAULT_CAMPAIGN["uplift"],
        points=20,
    )

    artifacts = paths.artifacts / "customers"
    artifacts.mkdir(parents=True, exist_ok=True)
    final_model.model.save_model(artifacts / "churn_xgboost.json")
    joblib.dump(final_model.calibrator, artifacts / "churn_calibrator.joblib")
    joblib.dump(segment_model, artifacts / "segments.joblib")
    (artifacts / "metadata.json").write_text(
        json.dumps(
            {
                "features": FEATURES,
                "churn_params": params,
                "clv_params": clv_model.params(),
                "segment_names": segment_model.names,
                "scoring_as_of": SCORING_AS_OF,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    segment_profile = (
        current.groupby(["segment_id", "segment_name"])
        .agg(
            customers=("customer_idx", "size"),
            median_spend_365d=("spend_365d", "median"),
            median_visits_365d=("visits_365d", "median"),
            mean_online_share=("online_share", "mean"),
            mean_churn_probability=("churn_probability", "mean"),
        )
        .reset_index()
    )
    report = {
        "scoring_as_of": SCORING_AS_OF,
        "churn": churn_report,
        "clv": clv_report,
        "segments": {
            "silhouette_by_k": segment_model.silhouette,
            "chosen_k": int(segment_model.kmeans.n_clusters),
            "profiles": json.loads(segment_profile.round(4).to_json(orient="records")),
        },
        "scores": {
            "customers": len(scores),
            "active_customers": int(scores["churn_probability"].notna().sum()),
            "risk_bands": scores["risk_band"].value_counts().to_dict(),
            "mean_churn_probability": round(
                float(active_scores["churn_probability"].mean()), 4
            ),
            "total_clv_180d": round(float(scores["clv_180d"].sum()), 1),
        },
        "campaign_example": {
            **DEFAULT_CAMPAIGN,
            "contacted": plan.contacted,
            "spend": round(plan.spend, 2),
            "expected_value_saved": round(plan.expected_value_saved, 2),
            "expected_net": round(plan.expected_net, 2),
            "roi": round(plan.roi, 4),
            "curve": json.loads(curve.round(3).to_json(orient="records")),
        },
        "timings_seconds": timings,
        "total_seconds": round(time.perf_counter() - started, 1),
        "package_versions": {
            n: version(n) for n in ("xgboost", "optuna", "scikit-learn", "scipy")
        },
    }
    paths.reports.mkdir(parents=True, exist_ok=True)
    (paths.reports / "customer_metrics.json").write_text(
        json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8"
    )
    con.close()
    return report


def _write_scores(
    con, paths: ProjectPaths, current: pd.DataFrame, value: pd.DataFrame
) -> pd.DataFrame:
    columns = [
        "customer_idx",
        "churn_probability",
        "risk_band",
        *DRIVER_COLUMNS,
        "segment_id",
        "segment_name",
        "recency_days",
        "visits_365d",
        "spend_365d",
    ]
    con.register("active_scores", current[columns])
    con.register("value_scores", value)
    target = paths.root / "data" / "serving" / "customer_scores.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"""
        COPY (
            SELECT m.customer_id,
                   a.churn_probability::FLOAT AS churn_probability,
                   coalesce(a.risk_band, 'lapsed') AS risk_band,
                   {", ".join(f"a.{c}" for c in DRIVER_COLUMNS)},
                   coalesce(a.segment_id, -1)::SMALLINT AS segment_id,
                   coalesce(a.segment_name, 'Lapsed (no purchase in 12 months)')
                       AS segment_name,
                   a.recency_days, a.visits_365d, a.spend_365d,
                   v.p_alive::FLOAT AS p_alive,
                   v.expected_visits::FLOAT AS expected_visits_180d,
                   v.expected_visit_spend::FLOAT AS expected_visit_spend,
                   v.clv::FLOAT AS clv_180d
            FROM customer_map AS m
            LEFT JOIN active_scores AS a USING (customer_idx)
            LEFT JOIN value_scores AS v USING (customer_idx)
            ORDER BY m.customer_id
        ) TO {sql_path(target)} (FORMAT parquet, COMPRESSION zstd)
        """)
    con.unregister("active_scores")
    con.unregister("value_scores")
    return pd.read_parquet(target)
