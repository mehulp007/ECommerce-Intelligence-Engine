from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ecommerce_intelligence.data.cleaning import (
    SOURCE_COLUMNS,
    clean_transactions,
    validate_uci_source,
)
from ecommerce_intelligence.features.snapshots import (
    FEATURE_COLUMNS,
    build_snapshot,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
GENERATED = PROJECT_ROOT / "data" / "processed" / "customer_snapshots.parquet"
PURCHASES = PROJECT_ROOT / "data" / "interim" / "purchases.parquet"


def transaction(
    invoice: str,
    customer: str | None,
    when: str,
    *,
    stock: str = "10001",
    quantity: int = 1,
    price: float = 10.0,
) -> dict:
    return {
        "InvoiceNo": invoice,
        "StockCode": stock,
        "Description": "Test product",
        "Quantity": quantity,
        "InvoiceDate": pd.Timestamp(when),
        "UnitPrice": price,
        "CustomerID": customer,
        "Country": "United Kingdom",
    }


def test_cleaning_excludes_invalid_sales_without_erasing_valid_line_items() -> None:
    first = transaction("100", "12345", "2021-01-10", quantity=2, price=3)
    rows = pd.DataFrame(
        [
            first,
            first.copy(),  # Duplicate-looking lines can be separate purchased units.
            transaction("C101", "12345", "2021-01-11", quantity=-1),
            transaction("102", None, "2021-01-12"),
            transaction("103", "12345", "2021-01-13", price=0),
        ]
    )

    cleaned = clean_transactions(rows)

    assert len(cleaned.purchases) == 2
    assert cleaned.purchases["line_total"].sum() == 12
    assert len(cleaned.returns) == 1
    assert cleaned.report["duplicate_source_rows"] == 1
    assert cleaned.report["missing_customer_id_rows"] == 1
    assert cleaned.report["cancellation_invoice_rows"] == 1
    assert cleaned.report["non_positive_unit_price_rows"] == 1


def test_future_purchases_change_labels_but_never_features() -> None:
    before = pd.DataFrame(
        [
            transaction("100", "1", "2021-01-01", price=10),
            transaction("101", "1", "2021-01-15", stock="10002", price=8),
            transaction("102", "2", "2021-01-20", price=5),
        ]
    )
    baseline = clean_transactions(before)
    later = pd.concat(
        [before, pd.DataFrame([transaction("103", "2", "2021-02-01")])],
        ignore_index=True,
    )
    with_future_sale = clean_transactions(later)
    kwargs = {
        "as_of": "2021-02-01",
        "observation_start": "2020-01-01",
        "observation_end": "2021-04-01",
    }

    first = build_snapshot(baseline.purchases, baseline.returns, **kwargs)
    second = build_snapshot(
        with_future_sale.purchases, with_future_sale.returns, **kwargs
    )

    pd.testing.assert_frame_equal(
        first.loc[:, ["customer_id", *FEATURE_COLUMNS]],
        second.loc[:, ["customer_id", *FEATURE_COLUMNS]],
    )
    assert first.set_index("customer_id").loc["2", "inactive_next_30d"] == 1
    assert second.set_index("customer_id").loc["2", "inactive_next_30d"] == 0
    assert first.set_index("customer_id").loc["1", "frequency_180d"] == 2
    assert first.set_index("customer_id").loc["1", "monetary_180d"] == 18


def test_label_window_excludes_a_purchase_exactly_30_days_later() -> None:
    rows = pd.DataFrame(
        [
            transaction("100", "1", "2021-01-01"),
            transaction("101", "1", "2021-03-03"),
        ]
    )
    cleaned = clean_transactions(rows)
    snapshot = build_snapshot(
        cleaned.purchases,
        cleaned.returns,
        as_of="2021-02-01",
        observation_start="2020-01-01",
        observation_end="2021-04-01",
    )
    assert snapshot.loc[0, "inactive_next_30d"] == 1


def test_partial_source_history_is_explicit_and_unobserved_labels_fail() -> None:
    cleaned = clean_transactions(pd.DataFrame([transaction("100", "1", "2010-12-01")]))
    snapshot = build_snapshot(
        cleaned.purchases,
        cleaned.returns,
        as_of="2011-04-01",
        observation_start="2010-12-01",
        observation_end="2011-05-01",
    )
    assert snapshot.loc[0, "observed_history_days"] == 121
    with pytest.raises(ValueError, match="lacks 30 days"):
        build_snapshot(
            cleaned.purchases,
            cleaned.returns,
            as_of="2011-04-01",
            observation_start="2010-12-01",
            observation_end="2011-04-20",
        )


def test_truncated_or_misshapen_uci_workbook_is_rejected() -> None:
    row = transaction("100", "1", "2021-01-01")
    with pytest.raises(ValueError, match="Expected"):
        validate_uci_source(pd.DataFrame([row]))
    with pytest.raises(ValueError, match="missing columns"):
        validate_uci_source(
            pd.DataFrame([{key: row[key] for key in SOURCE_COLUMNS[:-1]}])
        )


@pytest.mark.skipif(not GENERATED.exists(), reason="Run the data pipeline first")
def test_generated_uci_snapshots_pass_integrity_checks() -> None:
    snapshots = pd.read_parquet(GENERATED)
    report = json.loads((PROJECT_ROOT / "reports" / "data_quality.json").read_text())

    assert report["source_rows"] == 541_909
    assert len(snapshots) == report["snapshot_rows"]
    assert snapshots["snapshot_date"].nunique() == 8
    assert not snapshots.duplicated(["snapshot_date", "customer_id"]).any()
    assert not snapshots[list(FEATURE_COLUMNS)].isna().any().any()
    assert snapshots["recency_days"].between(1, 180).all()
    assert snapshots["inactive_next_30d"].isin([0, 1]).all()


@pytest.mark.skipif(not GENERATED.exists(), reason="Run the data pipeline first")
def test_saved_features_and_labels_match_transaction_windows() -> None:
    snapshots = pd.read_parquet(GENERATED)
    purchases = pd.read_parquet(PURCHASES)

    for row in snapshots.groupby("snapshot_date", sort=True).head(1).itertuples():
        customer = purchases.loc[purchases["customer_id"].eq(row.customer_id)]
        history = customer.loc[
            customer["invoice_date"].ge(row.snapshot_date - pd.Timedelta(days=180))
            & customer["invoice_date"].lt(row.snapshot_date)
        ]
        future = customer.loc[
            customer["invoice_date"].ge(row.snapshot_date)
            & customer["invoice_date"].lt(row.snapshot_date + pd.Timedelta(days=30))
        ]

        assert row.last_purchase_date == history["invoice_date"].max()
        assert row.frequency_180d == history["invoice_no"].nunique()
        assert row.monetary_180d == pytest.approx(history["line_total"].sum())
        assert row.inactive_next_30d == int(future.empty)
