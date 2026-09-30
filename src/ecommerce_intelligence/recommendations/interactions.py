"""Merchandise filtering, temporal split, and sparse customer-product counts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

CUTOFF = pd.Timestamp("2011-11-01")
HOLDOUT_END = pd.Timestamp("2011-12-01")
MERCHANDISE_CODE_PATTERN = r"[0-9]{5}[A-Z]{0,2}"
MAX_INVOICE_WEIGHT = 10
REQUIRED_COLUMNS = {
    "customer_id",
    "stock_code",
    "description",
    "invoice_no",
    "invoice_date",
    "quantity",
    "unit_price",
    "line_total",
}


@dataclass(frozen=True)
class InteractionData:
    matrix: csr_matrix
    customer_ids: list[str]
    stock_codes: list[str]
    catalog: pd.DataFrame
    pair_count: int


def merchandise_purchases(purchases: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED_COLUMNS.difference(purchases.columns)
    if missing:
        raise ValueError(f"Purchase data is missing columns: {sorted(missing)}")
    if purchases.empty:
        raise ValueError("No valid purchases are available")

    rows = purchases.loc[:, sorted(REQUIRED_COLUMNS)].copy()
    rows["customer_id"] = rows["customer_id"].astype("string").str.strip()
    rows["stock_code"] = rows["stock_code"].astype("string").str.strip()
    rows["description"] = rows["description"].astype("string").str.strip()
    rows["invoice_no"] = rows["invoice_no"].astype("string").str.strip()
    rows["invoice_date"] = pd.to_datetime(rows["invoice_date"], errors="coerce")
    valid = (
        rows["customer_id"].notna()
        & rows["customer_id"].ne("")
        & rows["stock_code"].str.fullmatch(MERCHANDISE_CODE_PATTERN, na=False)
        & rows["description"].notna()
        & rows["description"].ne("")
        & rows["invoice_no"].notna()
        & ~rows["invoice_no"].str.upper().str.startswith("C", na=False)
        & rows["invoice_date"].notna()
        & rows["quantity"].gt(0)
        & rows["unit_price"].gt(0)
        & rows["line_total"].gt(0)
    )
    result = rows.loc[valid].reset_index(drop=True)
    if result.empty:
        raise ValueError("No merchandise purchases remain after filtering")
    return result


def split_purchases(
    purchases: pd.DataFrame,
    *,
    cutoff: pd.Timestamp = CUTOFF,
    holdout_end: pd.Timestamp = HOLDOUT_END,
    observation_end: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = pd.Timestamp(cutoff)
    holdout_end = pd.Timestamp(holdout_end)
    if cutoff >= holdout_end:
        raise ValueError("The holdout must follow the training cutoff")
    if observation_end is not None and pd.Timestamp(observation_end) < holdout_end:
        raise ValueError("The complete holdout window was not observed")
    train = purchases.loc[purchases["invoice_date"].lt(cutoff)].copy()
    holdout = purchases.loc[
        purchases["invoice_date"].ge(cutoff) & purchases["invoice_date"].lt(holdout_end)
    ].copy()
    if train.empty or holdout.empty:
        raise ValueError("Training and holdout both need merchandise purchases")
    return train, holdout


def build_interactions(train: pd.DataFrame) -> InteractionData:
    if train.empty:
        raise ValueError("Cannot build an interaction matrix from no purchases")
    customer_ids = sorted(train["customer_id"].unique().tolist())
    stock_codes = sorted(train["stock_code"].unique().tolist())
    customer_index = {value: index for index, value in enumerate(customer_ids)}
    stock_index = {value: index for index, value in enumerate(stock_codes)}

    pairs = train.groupby(["customer_id", "stock_code"], sort=True).agg(
        invoice_count=("invoice_no", "nunique")
    )
    pairs = pairs.reset_index()
    row = pairs["customer_id"].map(customer_index).to_numpy(dtype=np.int32)
    column = pairs["stock_code"].map(stock_index).to_numpy(dtype=np.int32)
    weights = np.minimum(
        pairs["invoice_count"].to_numpy(dtype=np.float32), MAX_INVOICE_WEIGHT
    )
    matrix = csr_matrix(
        (weights, (row, column)),
        shape=(len(customer_ids), len(stock_codes)),
        dtype=np.float32,
    )
    matrix.sort_indices()
    if matrix.nnz != len(pairs) or np.any(matrix.data <= 0):
        raise ValueError("The interaction matrix lost or invalidated purchase pairs")

    by_product = train.groupby("stock_code", sort=True).agg(
        unique_buyers=("customer_id", "nunique"),
        invoice_count=("invoice_no", "nunique"),
        purchase_lines=("invoice_no", "size"),
        median_unit_price=("unit_price", "median"),
        last_sold=("invoice_date", "max"),
    )
    latest_description = (
        train.sort_values("invoice_date", kind="stable")
        .groupby("stock_code")["description"]
        .last()
    )
    catalog = by_product.join(latest_description).reset_index()
    catalog["item_id"] = catalog["stock_code"].map(stock_index).astype("int32")
    catalog["median_unit_price"] = catalog["median_unit_price"].round(2)
    ranked = catalog.sort_values(
        ["unique_buyers", "invoice_count", "stock_code"],
        ascending=[False, False, True],
        kind="stable",
    )
    catalog["popularity_rank"] = catalog["stock_code"].map(
        {code: rank for rank, code in enumerate(ranked["stock_code"].tolist(), 1)}
    )
    catalog = catalog.sort_values("item_id").reset_index(drop=True)
    if not catalog["item_id"].eq(np.arange(len(stock_codes))).all():
        raise ValueError("Product catalog and matrix columns are misaligned")
    return InteractionData(
        matrix=matrix,
        customer_ids=customer_ids,
        stock_codes=stock_codes,
        catalog=catalog,
        pair_count=len(pairs),
    )
