from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

joblib = pytest.importorskip("joblib")
XGBClassifier = pytest.importorskip("xgboost").XGBClassifier

from ecommerce_intelligence.evaluation.classification import classification_metrics
from ecommerce_intelligence.models.churn import calibrated_probabilities
from ecommerce_intelligence.models.temporal import (
    MODEL_FEATURES,
    TARGET,
    split_snapshots,
)

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "reports" / "milestone2_metrics.json"


@pytest.mark.skipif(
    not METRICS.exists()
    or not (ROOT / "data" / "processed" / "heldout_predictions.parquet").exists(),
    reason="Run model training first",
)
def test_saved_holdout_metrics_and_serving_artifacts() -> None:
    report = json.loads(METRICS.read_text(encoding="utf-8"))
    predictions = pd.read_parquet(
        ROOT / "data" / "processed" / "heldout_predictions.parquet"
    )
    snapshots = pd.read_parquet(
        ROOT / "data" / "processed" / "customer_snapshots.parquet"
    )
    test = split_snapshots(snapshots).test
    schema = json.loads(
        (ROOT / "artifacts" / "serving" / "feature_schema.json").read_text()
    )

    assert schema["model_features"] == list(MODEL_FEATURES)
    assert TARGET not in schema["model_features"]
    assert len(predictions) == len(test) == report["split"]["test_rows"]
    assert set(predictions["snapshot_date"].dt.strftime("%Y-%m-%d")) == {
        "2011-10-01",
        "2011-11-01",
    }
    recalculated = classification_metrics(
        predictions[TARGET].to_numpy(), predictions["calibrated_probability"].to_numpy()
    )
    assert recalculated["roc_auc"] == pytest.approx(
        report["heldout_metrics"]["calibrated"]["roc_auc"]
    )
    assert recalculated["brier"] == pytest.approx(
        report["heldout_metrics"]["calibrated"]["brier"]
    )

    model = XGBClassifier()
    model.load_model(ROOT / "artifacts" / "serving" / "churn_xgboost.json")
    calibrator = joblib.load(ROOT / "artifacts" / "serving" / "churn_calibrator.joblib")
    probe = test.head(20)
    persisted = predictions.merge(
        probe[["customer_id", "snapshot_date"]],
        on=["customer_id", "snapshot_date"],
    )
    assert np.allclose(
        calibrated_probabilities(model, calibrator, probe),
        persisted["calibrated_probability"].to_numpy(),
        atol=1e-8,
    )
