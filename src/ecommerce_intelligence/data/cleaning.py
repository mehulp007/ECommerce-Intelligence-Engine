"""Validate the source and separate purchase lines from cancellations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ecommerce_intelligence.data.source import EXPECTED_SOURCE_ROWS

SOURCE_COLUMNS = (
    "InvoiceNo",
    "StockCode",
    "Description",
    "Quantity",
    "InvoiceDate",
    "UnitPrice",
    "CustomerID",
    "Country",
)
TEXT_DTYPES = {
    column: "string"
    for column in ("InvoiceNo", "StockCode", "Description", "CustomerID", "Country")
}


@dataclass(frozen=True)
class CleanedTransactions:
    purchases: pd.DataFrame
    returns: pd.DataFrame
    report: dict
    source_first_date: pd.Timestamp
    source_last_date: pd.Timestamp


def read_workbook(path: Path) -> pd.DataFrame:
    return pd.read_excel(path, engine="openpyxl", dtype=TEXT_DTYPES)


def validate_uci_source(frame: pd.DataFrame) -> None:
    missing = sorted(set(SOURCE_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"UCI workbook is missing columns: {missing}")
    if len(frame) != EXPECTED_SOURCE_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_SOURCE_ROWS:,} UCI rows, found {len(frame):,}"
        )


def clean_transactions(frame: pd.DataFrame) -> CleanedTransactions:
    missing = sorted(set(SOURCE_COLUMNS).difference(frame.columns))
    if missing:
        raise ValueError(f"Transaction data is missing columns: {missing}")
    if frame.empty:
        raise ValueError("Transaction data is empty")

    rows = frame.loc[:, list(SOURCE_COLUMNS)].copy()
    for column in TEXT_DTYPES:
        rows[column] = rows[column].astype("string").str.strip()

    rows["InvoiceNo"] = rows["InvoiceNo"].fillna("")
    rows["StockCode"] = rows["StockCode"].fillna("")
    customer_id = rows["CustomerID"].str.replace(r"\.0$", "", regex=True)
    invalid_customer_id = customer_id.notna() & ~customer_id.str.fullmatch(
        r"\d+", na=False
    )
    rows["CustomerID"] = customer_id.where(~invalid_customer_id)
    rows["Quantity"] = pd.to_numeric(rows["Quantity"], errors="coerce")
    rows["UnitPrice"] = pd.to_numeric(rows["UnitPrice"], errors="coerce")
    rows["InvoiceDate"] = pd.to_datetime(rows["InvoiceDate"], errors="coerce")

    dates = rows["InvoiceDate"].dropna()
    if dates.empty:
        raise ValueError("Transaction data has no parseable invoice dates")
    source_first_date = dates.min()
    source_last_date = dates.max()

    cancellation = rows["InvoiceNo"].str.upper().str.startswith("C")
    valid_identity = (
        rows["CustomerID"].notna()
        & rows["InvoiceNo"].ne("")
        & rows["StockCode"].ne("")
        & rows["InvoiceDate"].notna()
    )
    valid_quantity = rows["Quantity"].gt(0) & np.isfinite(rows["Quantity"])
    valid_price = rows["UnitPrice"].gt(0) & np.isfinite(rows["UnitPrice"])
    purchase_mask = valid_identity & ~cancellation & valid_quantity & valid_price
    return_mask = valid_identity & (cancellation | rows["Quantity"].lt(0))

    renamed = {
        "InvoiceNo": "invoice_no",
        "StockCode": "stock_code",
        "Description": "description",
        "Quantity": "quantity",
        "InvoiceDate": "invoice_date",
        "UnitPrice": "unit_price",
        "CustomerID": "customer_id",
        "Country": "country",
    }
    purchases = rows.loc[purchase_mask].rename(columns=renamed).copy()
    purchases["line_total"] = purchases["quantity"] * purchases["unit_price"]
    purchases = purchases.sort_values(
        ["invoice_date", "invoice_no", "stock_code"], kind="stable"
    ).reset_index(drop=True)
    returns = rows.loc[return_mask].rename(columns=renamed).copy()
    returns = returns.sort_values(
        ["invoice_date", "invoice_no", "stock_code"], kind="stable"
    ).reset_index(drop=True)

    if purchases.empty:
        raise ValueError("No qualifying purchases remain after cleaning")
    if purchases["customer_id"].isna().any() or not purchases["line_total"].gt(0).all():
        raise ValueError("Cleaned purchases failed their integrity checks")

    report = {
        "source_rows": len(rows),
        "source_first_date": source_first_date.isoformat(),
        "source_last_date": source_last_date.isoformat(),
        "missing_customer_id_rows": int(rows["CustomerID"].isna().sum()),
        "invalid_customer_id_rows": int(invalid_customer_id.sum()),
        "missing_description_rows": int(rows["Description"].isna().sum()),
        "missing_invoice_date_rows": int(rows["InvoiceDate"].isna().sum()),
        "missing_invoice_no_rows": int(rows["InvoiceNo"].eq("").sum()),
        "missing_stock_code_rows": int(rows["StockCode"].eq("").sum()),
        "non_positive_quantity_rows": int(rows["Quantity"].le(0).sum()),
        "non_positive_unit_price_rows": int(rows["UnitPrice"].le(0).sum()),
        "cancellation_invoice_rows": int(cancellation.sum()),
        "duplicate_source_rows": int(rows.duplicated().sum()),
        "valid_purchase_rows": len(purchases),
        "excluded_purchase_rows": len(rows) - len(purchases),
        "return_rows_with_customer_id": len(returns),
        "purchase_customers": int(purchases["customer_id"].nunique()),
        "purchase_invoices": int(purchases["invoice_no"].nunique()),
        "purchase_products": int(purchases["stock_code"].nunique()),
    }
    return CleanedTransactions(
        purchases=purchases,
        returns=returns,
        report=report,
        source_first_date=source_first_date,
        source_last_date=source_last_date,
    )
