"""A calibrated 60-day inactivity model trained on a forward-in-time split.

The target is an inactivity proxy, not a confirmed account closure. Customer
IDs, snapshot dates, and the target are never predictors.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from xgboost import DMatrix, XGBClassifier
from xgboost.core import XGBoostError

from ecommerce_intelligence.customers.features import FEATURES, TARGET

TRAIN_SNAPSHOTS = ("2019-10-01", "2019-11-01", "2019-12-01", "2020-01-01", "2020-02-01")
CALIBRATION_SNAPSHOTS = ("2020-03-01", "2020-04-01")
TEST_SNAPSHOTS = ("2020-05-01", "2020-06-01", "2020-07-01")
# Training sample: a fixed 40% of customers keeps fitting fast; every
# evaluation uses the full population of each snapshot.
TRAIN_SAMPLE_MODULO = (5, 2)
TUNING_TRIALS = 25
SEED = 42
BASE_PARAMS = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "tree_method": "hist",
    "n_jobs": 0,
    "random_state": SEED,
}


def sampled(frame: pd.DataFrame) -> pd.DataFrame:
    modulo, keep = TRAIN_SAMPLE_MODULO
    return frame.loc[frame["customer_idx"] % modulo < keep]


def logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


class CalibratedChurnModel:
    """XGBoost scores mapped to probabilities by a Platt (sigmoid) calibrator."""

    def __init__(self, model: XGBClassifier, calibrator: LogisticRegression):
        self.model = model
        self.calibrator = calibrator

    def raw(self, frame: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(frame[FEATURES])[:, 1]

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        return self.calibrator.predict_proba(logit(self.raw(frame)))[:, 1]

    def contributions(self, frame: pd.DataFrame) -> np.ndarray:
        """Per-feature contributions to the raw log-odds (last column: bias).

        Exact TreeSHAP runs on the GPU when one is available (about 3x faster
        here, identical to 1e-5) and falls back to the CPU otherwise.
        """
        matrix = DMatrix(frame[FEATURES], feature_names=FEATURES)
        booster = self.model.get_booster()
        try:
            booster.set_param({"device": "cuda"})
            return booster.predict(matrix, pred_contribs=True)
        except XGBoostError:
            booster.set_param({"device": "cpu"})
            return booster.predict(matrix, pred_contribs=True)
        finally:
            booster.set_param({"device": "cpu"})


def fit_calibrated(
    params: dict, train: pd.DataFrame, calibration: pd.DataFrame
) -> CalibratedChurnModel:
    model = XGBClassifier(**BASE_PARAMS, **params)
    model.fit(train[FEATURES], train[TARGET])
    calibrator = LogisticRegression()
    calibrator.fit(
        logit(model.predict_proba(calibration[FEATURES])[:, 1]), calibration[TARGET]
    )
    return CalibratedChurnModel(model, calibrator)


def tune(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    *,
    trials: int = TUNING_TRIALS,
    log: Callable[[str], None] = print,
) -> dict:
    """Search tree hyperparameters by validation log loss."""
    from sklearn.metrics import log_loss

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 150, 600, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "min_child_weight": trial.suggest_float(
                "min_child_weight", 1, 50, log=True
            ),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 20, log=True),
        }
        model = XGBClassifier(**BASE_PARAMS, **params)
        model.fit(train[FEATURES], train[TARGET])
        return log_loss(valid[TARGET], model.predict_proba(valid[FEATURES])[:, 1])

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=SEED)
    )
    study.optimize(objective, n_trials=trials)
    log(f"  best tuning log loss {study.best_value:.4f} after {trials} trials")
    return study.best_params


def _log_features(frame: pd.DataFrame) -> np.ndarray:
    values = frame[FEATURES].to_numpy(dtype=np.float64)
    values = np.where(np.isnan(values), 0.0, values)
    return np.sign(values) * np.log1p(np.abs(values))


def logistic_baseline(features: list[str] | None = None):
    """Standardized logistic regression on log-scaled features."""
    if features is None:
        transform = FunctionTransformer(_log_features)
    else:

        def transform_subset(frame: pd.DataFrame) -> np.ndarray:
            values = frame[features].to_numpy(dtype=np.float64)
            return np.log1p(np.nan_to_num(values))

        transform = FunctionTransformer(transform_subset)
    return make_pipeline(transform, StandardScaler(), LogisticRegression(max_iter=500))
