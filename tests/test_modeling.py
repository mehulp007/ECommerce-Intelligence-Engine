from __future__ import annotations

import pandas as pd
import pytest

from ecommerce_intelligence.models.temporal import (
    CALIBRATION_MONTHS,
    MODEL_FEATURES,
    TARGET,
    TEST_MONTHS,
    TRAIN_MONTHS,
    split_snapshots,
)


def sample_snapshots() -> pd.DataFrame:
    months = pd.date_range("2011-04-01", "2011-11-01", freq="MS")
    rows = []
    for index, month in enumerate(months):
        row = {
            "customer_id": str(index),
            "snapshot_date": month,
            "observed_history_days": min(
                180, (month - pd.Timestamp("2010-12-01")).days
            ),
            TARGET: index % 2,
        }
        row.update({feature: index + 1 for feature in MODEL_FEATURES})
        rows.append(row)
    return pd.DataFrame(rows)


def test_temporal_split_excludes_partial_history_and_future_rows() -> None:
    snapshots = sample_snapshots()
    split = split_snapshots(snapshots)

    assert tuple(split.train["snapshot_date"].dt.strftime("%Y-%m-%d")) == TRAIN_MONTHS
    assert (
        tuple(split.calibration["snapshot_date"].dt.strftime("%Y-%m-%d"))
        == CALIBRATION_MONTHS
    )
    assert tuple(split.test["snapshot_date"].dt.strftime("%Y-%m-%d")) == TEST_MONTHS
    assert TARGET not in MODEL_FEATURES
    assert "customer_id" not in MODEL_FEATURES
    assert "snapshot_date" not in MODEL_FEATURES

    snapshots.loc[snapshots["snapshot_date"].eq("2011-10-01"), TARGET] = 0
    snapshots.loc[snapshots["snapshot_date"].eq("2011-10-01"), "monetary_180d"] = 9999
    unchanged = split_snapshots(snapshots)
    pd.testing.assert_frame_equal(split.train, unchanged.train)
    pd.testing.assert_frame_equal(split.calibration, unchanged.calibration)


def test_temporal_split_rejects_incomplete_history_and_duplicate_rows() -> None:
    snapshots = sample_snapshots()
    snapshots.loc[
        snapshots["snapshot_date"].eq("2011-06-01"), "observed_history_days"
    ] = 179
    with pytest.raises(ValueError, match="full 180-day"):
        split_snapshots(snapshots)

    snapshots = sample_snapshots()
    snapshots = pd.concat([snapshots, snapshots.iloc[[2]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        split_snapshots(snapshots)
