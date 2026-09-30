"""Build customer features from past purchases and labels from future purchases."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

HISTORY_DAYS = 180
RECENT_DAYS = 30
LABEL_DAYS = 30
DEFAULT_SNAPSHOTS = tuple(pd.date_range("2011-04-01", "2011-11-01", freq="MS"))
FEATURE_COLUMNS = (
    "recency_days",
    "frequency_180d",
    "monetary_180d",
    "orders_30d",
    "spend_30d",
    "distinct_products_180d",
    "return_invoices_180d",
    "observed_history_days",
)
SNAPSHOT_COLUMNS = (
    "customer_id",
    "snapshot_date",
    "last_purchase_date",
    *FEATURE_COLUMNS,
    "inactive_next_30d",
)


def build_snapshot(
    purchases: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    as_of: pd.Timestamp | str,
    observation_start: pd.Timestamp | str,
    observation_end: pd.Timestamp | str,
) -> pd.DataFrame:
    """Build one as-of table; the end timestamp is the last observed source event."""
    as_of = pd.Timestamp(as_of).normalize()
    observation_start = pd.Timestamp(observation_start).normalize()
    observation_end = pd.Timestamp(observation_end)
    label_end = as_of + pd.Timedelta(days=LABEL_DAYS)
    if observation_end < label_end:
        raise ValueError(f"The {as_of.date()} label lacks 30 days of observation")
    if observation_start >= as_of:
        raise ValueError("The snapshot has no observed purchase history")

    history_start = as_of - pd.Timedelta(days=HISTORY_DAYS)
    recent_start = as_of - pd.Timedelta(days=RECENT_DAYS)
    history = purchases.loc[
        purchases["invoice_date"].ge(history_start)
        & purchases["invoice_date"].lt(as_of)
    ]
    if history.empty:
        return pd.DataFrame(columns=SNAPSHOT_COLUMNS)

    grouped = history.groupby("customer_id", sort=True).agg(
        last_purchase_date=("invoice_date", "max"),
        frequency_180d=("invoice_no", "nunique"),
        monetary_180d=("line_total", "sum"),
        distinct_products_180d=("stock_code", "nunique"),
    )
    recent = (
        history.loc[history["invoice_date"].ge(recent_start)]
        .groupby("customer_id")
        .agg(orders_30d=("invoice_no", "nunique"), spend_30d=("line_total", "sum"))
    )
    return_counts = (
        returns.loc[
            returns["invoice_date"].ge(history_start)
            & returns["invoice_date"].lt(as_of)
        ]
        .groupby("customer_id")["invoice_no"]
        .nunique()
    )

    grouped["orders_30d"] = recent["orders_30d"].reindex(grouped.index, fill_value=0)
    grouped["spend_30d"] = recent["spend_30d"].reindex(grouped.index, fill_value=0.0)
    grouped["return_invoices_180d"] = return_counts.reindex(grouped.index, fill_value=0)
    grouped["recency_days"] = (
        as_of - grouped["last_purchase_date"].dt.normalize()
    ).dt.days
    grouped["observed_history_days"] = min(
        HISTORY_DAYS, (as_of - observation_start).days
    )
    grouped["monetary_180d"] = grouped["monetary_180d"].round(2)
    grouped["spend_30d"] = grouped["spend_30d"].round(2)
    for column in ("monetary_180d", "spend_30d"):
        grouped[column] = grouped[column].astype("float64")
    for column in (
        "frequency_180d",
        "orders_30d",
        "distinct_products_180d",
        "return_invoices_180d",
        "recency_days",
        "observed_history_days",
    ):
        grouped[column] = grouped[column].astype("int64")

    future_customers = purchases.loc[
        purchases["invoice_date"].ge(as_of) & purchases["invoice_date"].lt(label_end),
        "customer_id",
    ].unique()
    grouped["inactive_next_30d"] = (~grouped.index.isin(future_customers)).astype(
        "int8"
    )
    grouped.insert(0, "snapshot_date", as_of)
    snapshot = grouped.reset_index().loc[:, list(SNAPSHOT_COLUMNS)]
    if snapshot[list(FEATURE_COLUMNS)].isna().any().any():
        raise ValueError(f"Snapshot {as_of.date()} contains null model features")
    if not snapshot["recency_days"].between(1, HISTORY_DAYS).all():
        raise ValueError(f"Snapshot {as_of.date()} has invalid recency values")
    return snapshot


def build_monthly_snapshots(
    purchases: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    observation_start: pd.Timestamp | str,
    observation_end: pd.Timestamp | str,
    snapshot_dates: Iterable[pd.Timestamp | str] = DEFAULT_SNAPSHOTS,
) -> pd.DataFrame:
    dates = [pd.Timestamp(date).normalize() for date in snapshot_dates]
    if not dates or len(dates) != len(set(dates)):
        raise ValueError("Snapshot dates must be nonempty and unique")
    frames = [
        build_snapshot(
            purchases,
            returns,
            as_of=date,
            observation_start=observation_start,
            observation_end=observation_end,
        )
        for date in dates
    ]
    result = pd.concat(frames, ignore_index=True)
    result = result.sort_values(["snapshot_date", "customer_id"]).reset_index(drop=True)
    if result.duplicated(["snapshot_date", "customer_id"]).any():
        raise ValueError("A customer occurs more than once in a snapshot")
    if not result["inactive_next_30d"].isin([0, 1]).all():
        raise ValueError("Labels must be binary")
    return result
