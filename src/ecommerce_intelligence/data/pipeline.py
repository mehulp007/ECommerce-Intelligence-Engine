"""Ingest UCI transactions and build historical customer features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ecommerce_intelligence.data.cleaning import (
    clean_transactions,
    read_workbook,
    validate_uci_source,
)
from ecommerce_intelligence.data.source import ensure_workbook
from ecommerce_intelligence.features.snapshots import (
    DEFAULT_SNAPSHOTS,
    FEATURE_COLUMNS,
    build_monthly_snapshots,
)


def write_parquet(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.part")
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run(*, project_root: Path, force_download: bool = False) -> dict:
    workbook, source_report = ensure_workbook(
        project_root / "data" / "raw", force_download=force_download
    )
    source = read_workbook(workbook)
    validate_uci_source(source)
    cleaned = clean_transactions(source)
    snapshots = build_monthly_snapshots(
        cleaned.purchases,
        cleaned.returns,
        observation_start=cleaned.source_first_date,
        observation_end=cleaned.source_last_date,
    )
    if snapshots.empty or snapshots["snapshot_date"].nunique() != len(
        DEFAULT_SNAPSHOTS
    ):
        raise ValueError("Not all expected monthly snapshots were generated")

    interim = project_root / "data" / "interim"
    processed = project_root / "data" / "processed"
    reports = project_root / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    write_parquet(cleaned.purchases, interim / "purchases.parquet")
    write_parquet(cleaned.returns, interim / "returns.parquet")
    write_parquet(snapshots, processed / "customer_snapshots.parquet")

    summary = snapshots.groupby("snapshot_date", sort=True).agg(
        customers=("customer_id", "size"),
        inactive_rate_30d=("inactive_next_30d", "mean"),
        observed_history_days=("observed_history_days", "first"),
    )
    summary["inactive_rate_30d"] = summary["inactive_rate_30d"].round(4)
    summary.to_csv(reports / "snapshot_summary.csv")

    report = {
        **source_report,
        **cleaned.report,
        "snapshot_dates": [date.date().isoformat() for date in DEFAULT_SNAPSHOTS],
        "snapshot_rows": len(snapshots),
        "feature_columns": list(FEATURE_COLUMNS),
        "history_days": 180,
        "label_days": 30,
        "label_definition": "1 = no qualifying purchase in [snapshot, snapshot + 30 days)",
    }
    (reports / "data_quality.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Replace the cached UCI archive before rebuilding the data",
    )
    args = parser.parse_args()
    project_root = Path(__file__).resolve().parents[3]
    report = run(project_root=project_root, force_download=args.force_download)
    print(
        f"Validated {report['source_rows']:,} source rows; "
        f"kept {report['valid_purchase_rows']:,} purchase lines; "
        f"built {report['snapshot_rows']:,} customer snapshots."
    )


if __name__ == "__main__":
    main()
