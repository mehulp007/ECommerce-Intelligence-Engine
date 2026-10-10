"""Evaluate the recommender on the held-out week, then score every customer."""

from __future__ import annotations

import gc
import json
import time
from collections.abc import Callable
from datetime import date
from importlib.metadata import version

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

from ecommerce_intelligence.data.duck import sql_path
from ecommerce_intelligence.paths import ProjectPaths
from ecommerce_intelligence.recsys import config, ranker
from ecommerce_intelligence.recsys.als import AlsModel
from ecommerce_intelligence.recsys.candidates import build_candidates
from ecommerce_intelligence.recsys.data import open_context, week_labels
from ecommerce_intelligence.recsys.features import (
    FEATURE_GROUPS,
    FEATURES,
    build_features,
)
from ecommerce_intelligence.recsys.metrics import map_at_k, recall_at_k

RANK_COLUMNS = {
    "repurchase": "repurchase_rank",
    "popular": "popular_rank",
    "age_popular": "age_popular_rank",
    "cooc": "cooc_rank",
    "als": "als_rank",
}
SOURCE_COLUMNS = {
    "repurchase": ["repurchase_rank"],
    "popular": ["popular_rank"],
    "age_popular": ["age_popular_rank"],
    "cooc": ["cooc_score", "cooc_rank"],
    "als": ["als_score", "als_rank"],
}
INFERENCE_CHUNK = 50_000

Log = Callable[[str], None]


def _set_targets(con: duckdb.DuckDBPyConnection, customers: np.ndarray) -> None:
    con.register("target_frame", pd.DataFrame({"customer_idx": customers}))
    con.execute(
        "CREATE OR REPLACE TABLE targets AS "
        "SELECT DISTINCT customer_idx::INTEGER AS customer_idx FROM target_frame"
    )
    con.unregister("target_frame")


def _label_frame(labels: dict[int, set[int]]) -> pd.DataFrame:
    customers = np.repeat(
        np.fromiter(labels, dtype=np.int32), [len(v) for v in labels.values()]
    )
    articles = np.fromiter(
        (item for items in labels.values() for item in items), dtype=np.int32
    )
    return pd.DataFrame(
        {"customer_idx": customers, "article_idx": articles, "label": np.int8(1)}
    )


def prepare_week(
    con: duckdb.DuckDBPyConnection,
    as_of: date,
    customers: np.ndarray,
    labels: dict[int, set[int]] | None,
    als: AlsModel,
) -> pd.DataFrame:
    _set_targets(con, customers)
    build_candidates(con, as_of.isoformat(), als)
    frame = build_features(con, as_of.isoformat())
    if labels is not None:
        frame = frame.merge(
            _label_frame(labels), on=["customer_idx", "article_idx"], how="left"
        )
        frame["label"] = frame["label"].fillna(0).astype(np.int8)
    return frame


def _predictions(top: pd.DataFrame) -> dict[int, list[int]]:
    return top.groupby("customer_idx")["article_idx"].agg(list).to_dict()


def _ranked_by(frame: pd.DataFrame, column: str, fill: pd.Series | None = None):
    """Baseline lists: sort one source's rank, optionally topped up by popularity."""
    picked = (
        frame.dropna(subset=[column])
        .sort_values(["customer_idx", column])
        .groupby("customer_idx")["article_idx"]
        .agg(list)
        .to_dict()
    )
    if fill is None:
        return picked
    filler = fill.tolist()
    customers = frame["customer_idx"].unique()
    result = {}
    for customer in customers:
        items = picked.get(customer, [])[: config.K]
        for item in filler:
            if len(items) == config.K:
                break
            if item not in items:
                items.append(item)
        result[customer] = items
    return result


def _ablate(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    others = [RANK_COLUMNS[s] for s in config.SOURCES if s != source]
    out = frame.loc[frame[others].notna().any(axis=1)].copy()
    out[SOURCE_COLUMNS[source]] = np.nan
    out["n_sources"] = out[list(RANK_COLUMNS.values())].notna().sum(axis=1)
    out["n_sources"] = out["n_sources"].astype(np.float32)
    return out


def _evaluate_model(
    model: lgb.Booster, frame: pd.DataFrame, features: list[str], labels: dict
) -> dict:
    top = ranker.top_k(frame, model.predict(frame[features]), config.K)
    predictions = _predictions(top)
    return {
        "map_at_12": map_at_k(predictions, labels, config.K),
        "recall_at_12": recall_at_k(predictions, labels, config.K),
    }


def _reason(frame: pd.DataFrame) -> pd.Series:
    reason = pd.Series("trending", index=frame.index, dtype="object")
    for source in reversed(list(config.REASONS)):
        reason[frame[RANK_COLUMNS[source]].notna()] = config.REASONS[source]
    return reason


def run(
    paths: ProjectPaths,
    *,
    ablations: bool = True,
    score_all: bool = True,
    log: Log = print,
) -> dict:
    started = time.perf_counter()
    timings: dict[str, float] = {}

    def lap(name: str, since: float) -> float:
        timings[name] = round(time.perf_counter() - since, 1)
        log(f"{name}: {timings[name]:.1f}s")
        return time.perf_counter()

    tick = time.perf_counter()
    con = open_context(paths)
    validation = config.VALIDATION_START
    train_dates = config.training_as_of_dates(
        validation - config.WEEK, config.TRAINING_WEEKS
    )
    tick = lap("load", tick)

    frames: dict[date, pd.DataFrame] = {}
    labels: dict[date, dict[int, set[int]]] = {}
    for as_of in [validation, *train_dates]:
        labels[as_of] = week_labels(con, as_of.isoformat())
        customers = np.fromiter(labels[as_of], dtype=np.int32)
        als = AlsModel.fit(con, as_of.isoformat())
        frame = prepare_week(con, as_of, customers, labels[as_of], als)
        # Training weeks only need customers with a purchased candidate; the
        # held-out week stays complete so every buyer counts in evaluation.
        frames[as_of] = frame if as_of == validation else ranker.with_positives(frame)
        del als, frame
        gc.collect()
        tick = lap(f"week {as_of}: {len(frames[as_of]):,} candidate rows", tick)

    val_frame, val_labels = frames[validation], labels[validation]
    candidate_recall = {
        source: float(
            val_frame.loc[val_frame[column].notna(), "label"].sum()
            / sum(len(v) for v in val_labels.values())
        )
        for source, column in RANK_COLUMNS.items()
    }
    candidate_recall["all"] = float(
        val_frame["label"].sum() / sum(len(v) for v in val_labels.values())
    )
    candidates_per_customer = round(len(val_frame) / len(val_labels), 1)

    # Early stopping uses the most recent training week, never the held-out week.
    newest, *older = train_dates
    stopper = ranker.fit(
        pd.concat([frames[d] for d in older]), FEATURES, valid=frames[newest]
    )
    best_rounds = max(stopper.best_iteration, 50)
    del stopper
    tick = lap(f"early stopping ({best_rounds} rounds)", tick)

    train_frame = pd.concat([frames[d] for d in train_dates], ignore_index=True)
    model = ranker.fit(train_frame, FEATURES, rounds=best_rounds)
    tick = lap("ranker training", tick)

    results = {"ranker": _evaluate_model(model, val_frame, FEATURES, val_labels)}
    popular = (
        val_frame.dropna(subset=["popular_rank"])
        .drop_duplicates("article_idx")
        .sort_values("popular_rank")["article_idx"]
    )
    baselines = {
        "popular": {c: popular.head(config.K).tolist() for c in val_labels},
        "repurchase_then_popular": _ranked_by(val_frame, "repurchase_rank", popular),
        "cooc_then_popular": _ranked_by(val_frame, "cooc_rank", popular),
        "als": _ranked_by(val_frame, "als_rank"),
    }
    for name, predictions in baselines.items():
        results[name] = {
            "map_at_12": map_at_k(predictions, val_labels, config.K),
            "recall_at_12": recall_at_k(predictions, val_labels, config.K),
        }
    for name, value in results.items():
        log(f"  {name:24s} MAP@12 {value['map_at_12']:.5f}")
    tick = lap("evaluation", tick)

    importance = pd.Series(
        model.feature_importance("gain"), index=model.feature_name()
    ).sort_values(ascending=False)

    ablation_results = {}
    if ablations:
        for source in config.SOURCES:
            model_a = ranker.fit(
                pd.concat([_ablate(frames[d], source) for d in train_dates]),
                FEATURES,
                rounds=best_rounds,
            )
            ablation_results[f"without_{source}"] = _evaluate_model(
                model_a, _ablate(val_frame, source), FEATURES, val_labels
            )
            log(f"  without {source}: {ablation_results[f'without_{source}']}")
        for group, columns in FEATURE_GROUPS.items():
            if group == "source":
                continue
            kept = [f for f in FEATURES if f not in columns]
            model_a = ranker.fit(train_frame, kept, rounds=best_rounds)
            ablation_results[f"without_{group}_features"] = _evaluate_model(
                model_a, val_frame, kept, val_labels
            )
            name = f"without_{group}_features"
            log(f"  {name}: {ablation_results[name]}")
        tick = lap("ablations", tick)

    del train_frame
    frames[validation] = ranker.with_positives(val_frame)
    del val_frame
    gc.collect()

    # The production model also learns from the held-out week.
    final_dates = [validation, *train_dates[:-1]]
    final_model = ranker.fit(
        pd.concat([frames[d] for d in final_dates], ignore_index=True),
        FEATURES,
        rounds=best_rounds,
    )
    artifact_dir = paths.artifacts / "recsys"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    final_model.save_model(str(artifact_dir / "ranker.txt"))
    tick = lap("final model", tick)
    frames.clear()
    gc.collect()

    scored = {}
    if score_all:
        scored = score_customers(con, paths, final_model, log=log)
        tick = lap("score all customers", tick)

    report = {
        "as_of": {
            "validation_week": [
                validation.isoformat(),
                (validation + config.WEEK).isoformat(),
            ],
            "training_label_weeks": [d.isoformat() for d in train_dates],
            "inference_as_of": config.INFERENCE_AS_OF.isoformat(),
        },
        "validation_customers": len(val_labels),
        "validation_pairs": int(sum(len(v) for v in val_labels.values())),
        "candidates_per_customer": candidates_per_customer,
        "candidate_recall": candidate_recall,
        "results": results,
        "ablations": ablation_results,
        "ranker": {
            "rounds": best_rounds,
            "params": {k: v for k, v in ranker.PARAMS.items() if k != "eval_at"},
            "features": FEATURES,
            "top_features_by_gain": {
                k: round(float(v), 1) for k, v in importance.head(15).items()
            },
        },
        "candidate_sizes": {
            "repurchase": config.REPURCHASE_CANDIDATES,
            "popular": config.POPULAR_CANDIDATES,
            "age_popular": config.AGE_POPULAR_CANDIDATES,
            "cooc": config.COOC_CANDIDATES,
            "als": config.ALS_CANDIDATES,
        },
        "scoring": scored,
        "timings_seconds": timings,
        "total_seconds": round(time.perf_counter() - started, 1),
        "package_versions": {
            name: version(name) for name in ("lightgbm", "implicit", "duckdb")
        },
    }
    paths.reports.mkdir(parents=True, exist_ok=True)
    (paths.reports / "recommender_metrics.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (artifact_dir / "metadata.json").write_text(
        json.dumps(
            {
                "features": FEATURES,
                "rounds": best_rounds,
                "inference_as_of": config.INFERENCE_AS_OF.isoformat(),
                "validation_map_at_12": results["ranker"]["map_at_12"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    con.close()
    return report


def score_customers(
    con: duckdb.DuckDBPyConnection,
    paths: ProjectPaths,
    model: lgb.Booster,
    *,
    log: Log = print,
) -> dict:
    """Write the top 12 articles for every customer as of the inference date."""
    as_of = config.INFERENCE_AS_OF
    als = AlsModel.fit(con, as_of.isoformat())
    customers = con.execute(
        "SELECT customer_idx FROM customer_map ORDER BY customer_idx"
    ).fetchnumpy()["customer_idx"]
    con.execute("""
        CREATE OR REPLACE TABLE recommendations (
            customer_idx INTEGER, rank TINYINT, article_idx INTEGER,
            score FLOAT, reason VARCHAR
        )
        """)
    for start in range(0, len(customers), INFERENCE_CHUNK):
        chunk = customers[start : start + INFERENCE_CHUNK]
        frame = prepare_week(con, as_of, chunk, None, als)
        top = ranker.top_k(frame, model.predict(frame[FEATURES]), config.K)
        top = top.assign(reason=_reason(top))[
            ["customer_idx", "rank", "article_idx", "score", "reason"]
        ]
        con.register("top_frame", top)
        con.execute("INSERT INTO recommendations SELECT * FROM top_frame")
        con.unregister("top_frame")
        log(f"  scored {min(start + INFERENCE_CHUNK, len(customers)):,} customers")
        del frame, top
        gc.collect()

    target = paths.root / "data" / "serving" / "recommendations.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"""
        COPY (
            SELECT c.customer_id, r.rank, a.article_id, r.score, r.reason
            FROM recommendations AS r
            JOIN customer_map AS c USING (customer_idx)
            JOIN article_map AS a USING (article_idx)
            ORDER BY c.customer_id, r.rank
        ) TO {sql_path(target)} (FORMAT parquet, COMPRESSION zstd)
        """)
    summary = con.execute("""
        SELECT count(DISTINCT customer_idx) AS customers, count(*) AS rows,
               count(DISTINCT article_idx) AS distinct_articles
        FROM recommendations
        """).fetchone()
    reasons = dict(
        con.execute(
            "SELECT reason, count(*) FROM recommendations GROUP BY ALL ORDER BY 2 DESC"
        ).fetchall()
    )
    total = summary[1]
    return {
        "customers": summary[0],
        "rows": total,
        "catalog_coverage": round(
            summary[2] / con.execute("SELECT count(*) FROM article_map").fetchone()[0],
            4,
        ),
        "reason_share": {k: round(v / total, 4) for k, v in reasons.items()},
        "output": str(target.relative_to(paths.root)),
    }
