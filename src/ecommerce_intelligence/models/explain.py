"""Held-out Tree SHAP explanations for the uncalibrated XGBoost margin."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier

from ecommerce_intelligence.models.churn import calibrated_probabilities
from ecommerce_intelligence.models.temporal import MODEL_FEATURES

SHAP_SAMPLE_SIZE = 512
RANDOM_STATE = 42


def generate_explanations(
    model: XGBClassifier,
    calibrator,
    test: pd.DataFrame,
    *,
    report_dir: Path,
    processed_dir: Path,
) -> tuple[pd.DataFrame, dict]:
    sample = test.sample(n=min(SHAP_SAMPLE_SIZE, len(test)), random_state=RANDOM_STATE)
    sample = sample.reset_index(drop=True)
    features = sample.loc[:, list(MODEL_FEATURES)]
    explanation = shap.TreeExplainer(model, model_output="raw")(features)
    values = np.asarray(explanation.values)
    if values.shape != (len(sample), len(MODEL_FEATURES)):
        raise ValueError(f"Unexpected SHAP value shape: {values.shape}")

    raw_margin = model.predict(features, output_margin=True)
    reconstructed = np.asarray(explanation.base_values) + values.sum(axis=1)
    if not np.allclose(raw_margin, reconstructed, atol=1e-3):
        raise ValueError("SHAP values do not reconstruct the XGBoost margin")

    raw_probability = model.predict_proba(features)[:, 1]
    calibrated_probability = calibrated_probabilities(model, calibrator, features)
    local = sample.loc[:, ["customer_id", "snapshot_date", "inactive_next_30d"]].copy()
    local["base_margin"] = explanation.base_values
    local["raw_margin"] = raw_margin
    local["raw_probability"] = raw_probability
    local["calibrated_probability"] = calibrated_probability
    for index, column in enumerate(MODEL_FEATURES):
        local[column] = features[column].to_numpy()
        local[f"shap_{column}"] = values[:, index]
    processed_dir.mkdir(parents=True, exist_ok=True)
    local.to_parquet(processed_dir / "shap_heldout_sample.parquet", index=False)

    global_importance = pd.DataFrame(
        {
            "feature": MODEL_FEATURES,
            "mean_absolute_shap": np.abs(values).mean(axis=0),
            "mean_shap": values.mean(axis=0),
        }
    ).sort_values("mean_absolute_shap", ascending=False)
    report_dir.mkdir(parents=True, exist_ok=True)
    global_importance.to_csv(report_dir / "shap_global.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ordered = global_importance.iloc[::-1]
    ax.barh(ordered["feature"], ordered["mean_absolute_shap"], color="#4F6D8A")
    ax.set_xlabel("Mean absolute SHAP value (XGBoost log-odds)")
    ax.set_title("Held-out churn drivers")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(report_dir / "shap_global.png", dpi=160)
    plt.close(fig)

    high_risk_index = int(np.argmax(calibrated_probability))
    shap.plots.waterfall(explanation[high_risk_index], max_display=8, show=False)
    plt.tight_layout()
    plt.savefig(report_dir / "shap_high_risk_example.png", dpi=160, bbox_inches="tight")
    plt.close("all")
    return global_importance, {
        "sample_rows": len(sample),
        "max_additivity_error": float(np.max(np.abs(raw_margin - reconstructed))),
        "explained_output": "uncalibrated XGBoost log-odds",
        "top_feature": str(global_importance.iloc[0]["feature"]),
    }
