"""Fixed, forward-only splits for the 2011 customer snapshots."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ecommerce_intelligence.features.snapshots import FEATURE_COLUMNS

TRAIN_MONTHS = ("2011-06-01", "2011-07-01", "2011-08-01")
CALIBRATION_MONTHS = ("2011-09-01",)
TEST_MONTHS = ("2011-10-01", "2011-11-01")
MODEL_FEATURES = tuple(
    column for column in FEATURE_COLUMNS if column != "observed_history_days"
)
TARGET = "inactive_next_30d"


@dataclass(frozen=True)
class TemporalSplit:
    train: pd.DataFrame
    calibration: pd.DataFrame
    test: pd.DataFrame


def split_snapshots(snapshots: pd.DataFrame) -> TemporalSplit:
    required = {"customer_id", "snapshot_date", TARGET, "observed_history_days"}
    required.update(MODEL_FEATURES)
    missing = required.difference(snapshots.columns)
    if missing:
        raise ValueError(f"Snapshot table is missing columns: {sorted(missing)}")
    if snapshots.duplicated(["customer_id", "snapshot_date"]).any():
        raise ValueError("Snapshot table contains duplicate customer-month rows")

    dates = pd.to_datetime(snapshots["snapshot_date"])

    def select(months: tuple[str, ...]) -> pd.DataFrame:
        selected = snapshots.loc[dates.isin(pd.to_datetime(months))].copy()
        if set(selected["snapshot_date"].dt.strftime("%Y-%m-%d")) != set(months):
            raise ValueError(f"A requested month is missing: {months}")
        if not selected["observed_history_days"].eq(180).all():
            raise ValueError("Training requires a full 180-day source history")
        if selected[list(MODEL_FEATURES) + [TARGET]].isna().any().any():
            raise ValueError("Model rows contain null features or labels")
        if not selected[TARGET].isin([0, 1]).all():
            raise ValueError("The target must be binary")
        return selected.sort_values(["snapshot_date", "customer_id"]).reset_index(
            drop=True
        )

    train = select(TRAIN_MONTHS)
    calibration = select(CALIBRATION_MONTHS)
    test = select(TEST_MONTHS)
    if not (
        train["snapshot_date"].max()
        < calibration["snapshot_date"].min()
        < test["snapshot_date"].min()
    ):
        raise ValueError("Training, calibration, and test dates are out of order")
    return TemporalSplit(train=train, calibration=calibration, test=test)
