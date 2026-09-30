"""Fixed XGBoost churn model and separate forward-time probability calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from ecommerce_intelligence.models.temporal import MODEL_FEATURES, TARGET, TemporalSplit

RANDOM_STATE = 42
XGB_PARAMETERS = {
    "n_estimators": 300,
    "max_depth": 3,
    "learning_rate": 0.04,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "min_child_weight": 10,
    "reg_lambda": 5.0,
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "n_jobs": 4,
    "random_state": RANDOM_STATE,
}


def logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def calibrated_probabilities(
    model: XGBClassifier, calibrator: LogisticRegression, features: pd.DataFrame
) -> np.ndarray:
    raw = model.predict_proba(features.loc[:, list(MODEL_FEATURES)])[:, 1]
    return calibrator.predict_proba(logit(raw))[:, 1]


def fit_churn(split: TemporalSplit) -> tuple[XGBClassifier, LogisticRegression]:
    model = XGBClassifier(**XGB_PARAMETERS)
    train_x = split.train.loc[:, list(MODEL_FEATURES)]
    train_y = split.train[TARGET].to_numpy(dtype=int)
    calibration_x = split.calibration.loc[:, list(MODEL_FEATURES)]
    calibration_y = split.calibration[TARGET].to_numpy(dtype=int)
    if np.unique(train_y).size != 2 or np.unique(calibration_y).size != 2:
        raise ValueError("Training and calibration each require both target classes")

    model.fit(train_x, train_y)
    raw_calibration = model.predict_proba(calibration_x)[:, 1]
    calibrator = LogisticRegression(solver="lbfgs", random_state=RANDOM_STATE)
    calibrator.fit(logit(raw_calibration), calibration_y)
    return model, calibrator
