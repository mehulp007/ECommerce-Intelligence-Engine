"""Train personas and inactivity models on historical customer snapshots."""

from __future__ import annotations

import json
import os
from importlib.metadata import version
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from ecommerce_intelligence.data.source import sha256_file
from ecommerce_intelligence.evaluation.classification import classification_metrics
from ecommerce_intelligence.models.churn import (
    XGB_PARAMETERS,
    calibrated_probabilities,
    fit_churn,
)
from ecommerce_intelligence.models.personas import RFM_FEATURES, fit_personas
from ecommerce_intelligence.models.temporal import (
    CALIBRATION_MONTHS,
    MODEL_FEATURES,
    TARGET,
    TEST_MONTHS,
    TRAIN_MONTHS,
    split_snapshots,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _evaluate(
    frame: pd.DataFrame, model: XGBClassifier, calibrator
) -> tuple[dict, pd.DataFrame]:
    features = frame.loc[:, list(MODEL_FEATURES)]
    raw = model.predict_proba(features)[:, 1]
    calibrated = calibrated_probabilities(model, calibrator, features)
    result = frame.loc[:, ["customer_id", "snapshot_date", TARGET]].copy()
    result["raw_probability"] = raw
    result["calibrated_probability"] = calibrated
    metrics = {
        "raw": classification_metrics(frame[TARGET].to_numpy(), raw),
        "calibrated": classification_metrics(frame[TARGET].to_numpy(), calibrated),
    }
    return metrics, result


def run(*, project_root: Path) -> dict:
    matplotlib_cache = project_root / ".matplotlib"
    matplotlib_cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
    from ecommerce_intelligence.models.explain import generate_explanations

    snapshot_path = project_root / "data" / "processed" / "customer_snapshots.parquet"
    if not snapshot_path.is_file():
        raise FileNotFoundError("Run the data pipeline before training")
    snapshots = pd.read_parquet(snapshot_path)
    split = split_snapshots(snapshots)

    artifacts = project_root / "artifacts" / "serving"
    reports = project_root / "reports"
    processed = project_root / "data" / "processed"
    artifacts.mkdir(parents=True, exist_ok=True)
    reports.mkdir(parents=True, exist_ok=True)

    personas, persona_profile, persona_metrics = fit_personas(split.train)
    persona_profile.to_csv(reports / "persona_profiles.csv", index=False)
    joblib.dump(personas, artifacts / "rfm_personas.joblib")
    persona_names = {
        int(row.persona_id): row.persona_name
        for row in persona_profile.itertuples(index=False)
    }
    _write_json(artifacts / "persona_names.json", persona_names)

    model, calibrator = fit_churn(split)
    model.save_model(artifacts / "churn_xgboost.json")
    joblib.dump(calibrator, artifacts / "churn_calibrator.joblib")
    schema = {
        "model_features": list(MODEL_FEATURES),
        "rfm_features": list(RFM_FEATURES),
        "target": TARGET,
        "positive_class": "no qualifying purchase in the next 30 days",
        "history_days": 180,
        "label_days": 30,
        "train_months": TRAIN_MONTHS,
        "calibration_months": CALIBRATION_MONTHS,
        "test_months": TEST_MONTHS,
        "calibration": "sigmoid fit on September raw XGBoost log-odds",
    }
    _write_json(artifacts / "feature_schema.json", schema)

    calibration_metrics, _ = _evaluate(split.calibration, model, calibrator)
    test_metrics, test_predictions = _evaluate(split.test, model, calibrator)
    test_predictions.to_parquet(processed / "heldout_predictions.parquet", index=False)
    monthly = {}
    for date, group in test_predictions.groupby("snapshot_date", sort=True):
        monthly[date.date().isoformat()] = classification_metrics(
            group[TARGET].to_numpy(), group["calibrated_probability"].to_numpy()
        )

    # Reload the serving artifacts and check the resulting probabilities.
    reloaded_model = XGBClassifier()
    reloaded_model.load_model(artifacts / "churn_xgboost.json")
    reloaded_calibrator = joblib.load(artifacts / "churn_calibrator.joblib")
    reloaded_personas = joblib.load(artifacts / "rfm_personas.joblib")
    probe = split.test.head(20)
    if not np.allclose(
        calibrated_probabilities(model, calibrator, probe),
        calibrated_probabilities(reloaded_model, reloaded_calibrator, probe),
        atol=1e-8,
    ):
        raise ValueError("Serialized churn artifacts changed predictions")
    if not np.array_equal(
        personas.predict(probe.loc[:, list(RFM_FEATURES)]),
        reloaded_personas.predict(probe.loc[:, list(RFM_FEATURES)]),
    ):
        raise ValueError("Serialized persona artifact changed assignments")

    shap_global, shap_summary = generate_explanations(
        model,
        calibrator,
        split.test,
        report_dir=reports,
        processed_dir=processed,
    )
    data_hash = sha256_file(snapshot_path)
    package_versions = {
        name: version(name)
        for name in ("numpy", "pandas", "scikit-learn", "xgboost", "shap", "mlflow")
    }

    database_uri = f"sqlite:///{(project_root / 'mlflow.db').as_posix()}"
    mlflow.set_tracking_uri(database_uri)
    experiment_name = "ecommerce-intelligence-milestone2"
    if mlflow.get_experiment_by_name(experiment_name) is None:
        mlflow.create_experiment(
            experiment_name, artifact_location=(project_root / "mlruns").as_uri()
        )
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="rfm-personas-k4") as persona_run:
        mlflow.log_params(
            {
                "source_snapshot_sha256": data_hash,
                "fit_month": TRAIN_MONTHS[-1],
                "rfm_features": ",".join(RFM_FEATURES),
                "clusters": 4,
                "random_state": 42,
            }
        )
        mlflow.log_metrics(persona_metrics)
        for path in (
            artifacts / "rfm_personas.joblib",
            artifacts / "persona_names.json",
            reports / "persona_profiles.csv",
        ):
            mlflow.log_artifact(str(path))

    with mlflow.start_run(run_name="xgboost-churn-calibrated") as churn_run:
        mlflow.log_params(
            {
                **XGB_PARAMETERS,
                "source_snapshot_sha256": data_hash,
                "model_features": ",".join(MODEL_FEATURES),
                "train_months": ",".join(TRAIN_MONTHS),
                "calibration_months": ",".join(CALIBRATION_MONTHS),
                "test_months": ",".join(TEST_MONTHS),
                "calibration": "sigmoid_logistic_regression",
                "decision_threshold": 0.5,
            }
        )
        mlflow.log_metrics(
            {
                f"test_{key}": value
                for key, value in test_metrics["calibrated"].items()
                if key not in {"rows", "tn", "fp", "fn", "tp"}
            }
        )
        mlflow.log_metrics(
            {
                f"test_raw_{key}": test_metrics["raw"][key]
                for key in ("roc_auc", "average_precision", "brier", "log_loss")
            }
        )
        for path in (
            artifacts / "churn_xgboost.json",
            artifacts / "churn_calibrator.joblib",
            artifacts / "feature_schema.json",
            reports / "shap_global.csv",
            reports / "shap_global.png",
            reports / "shap_high_risk_example.png",
        ):
            mlflow.log_artifact(str(path))

    report = {
        "data_sha256": data_hash,
        "package_versions": package_versions,
        "split": {
            "train_months": TRAIN_MONTHS,
            "calibration_months": CALIBRATION_MONTHS,
            "test_months": TEST_MONTHS,
            "train_rows": len(split.train),
            "calibration_rows": len(split.calibration),
            "test_rows": len(split.test),
            "train_prevalence": float(split.train[TARGET].mean()),
            "calibration_prevalence": float(split.calibration[TARGET].mean()),
            "test_prevalence": float(split.test[TARGET].mean()),
        },
        "personas": {
            **persona_metrics,
            "names": persona_names,
            "mlflow_run_id": persona_run.info.run_id,
        },
        "calibration_metrics": calibration_metrics,
        "heldout_metrics": test_metrics,
        "heldout_by_month": monthly,
        "shap": {
            **shap_summary,
            "global_ranking": shap_global["feature"].tolist(),
        },
        "churn_mlflow_run_id": churn_run.info.run_id,
    }
    _write_json(reports / "milestone2_metrics.json", report)
    print(
        f"Held-out ROC AUC {test_metrics['calibrated']['roc_auc']:.3f}; "
        f"average precision {test_metrics['calibrated']['average_precision']:.3f}; "
        f"Brier {test_metrics['calibrated']['brier']:.3f} "
        f"on {len(split.test):,} customer-months."
    )
    return report


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    run(project_root=project_root)


if __name__ == "__main__":
    main()
