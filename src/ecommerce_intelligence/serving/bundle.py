"""Export and validate an immutable local inference bundle."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd

from ecommerce_intelligence.models.temporal import MODEL_FEATURES

AS_OF = "2011-11-01"
ARTIFACT_FILES = (
    "churn_xgboost.json",
    "churn_calibrator.joblib",
    "rfm_personas.joblib",
    "feature_schema.json",
    "persona_names.json",
    "bm25_recommender.npz",
    "recommender_user_items.npz",
    "recommender_mappings.json",
    "product_catalog.parquet",
    "recommender_schema.json",
)
FEATURE_FILE = "customer_features.parquet"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def validate_bundle(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    if manifest["schema_version"] != 1 or manifest["as_of"] != AS_OF:
        raise ValueError("Unsupported serving bundle schema or as-of date")
    if directory.name != manifest["bundle_version"]:
        raise ValueError("Serving bundle directory and version disagree")
    if set(manifest["files"]) != {*ARTIFACT_FILES, FEATURE_FILE}:
        raise ValueError("Serving bundle file list is incomplete")
    for name, expected in manifest["files"].items():
        if file_hash(directory / name) != expected:
            raise ValueError(f"Serving bundle checksum failed: {name}")
    return manifest


def export_bundle(project_root: Path, model_versions: dict[str, int]) -> Path:
    source = project_root / "artifacts" / "serving"
    snapshots = pd.read_parquet(
        project_root / "data" / "processed" / "customer_snapshots.parquet"
    )
    selected = snapshots.loc[
        snapshots["snapshot_date"].eq(pd.Timestamp(AS_OF)),
        ["customer_id", "snapshot_date", *MODEL_FEATURES],
    ].sort_values("customer_id")
    if selected.empty or selected["customer_id"].duplicated().any():
        raise ValueError("November feature rows are missing or duplicated")
    if selected.isna().any().any():
        raise ValueError("November feature rows contain nulls")
    if not all((source / name).is_file() for name in ARTIFACT_FILES):
        raise FileNotFoundError(
            "Train the personas, inactivity model, and recommender before exporting"
        )
    feature_bytes = selected.to_parquet(index=False)
    hashes = {name: file_hash(source / name) for name in ARTIFACT_FILES}
    hashes[FEATURE_FILE] = hashlib.sha256(feature_bytes).hexdigest()
    identity = hashlib.sha256(
        json.dumps(hashes, sort_keys=True).encode("utf-8")
    ).hexdigest()[:12]
    version = f"uci-2011-11-01-{identity}"
    destination = project_root / "artifacts" / "bundles" / version
    if destination.exists():
        validate_bundle(destination)
        return destination
    destination.mkdir(parents=True)
    for name in ARTIFACT_FILES:
        shutil.copy2(source / name, destination / name)
    (destination / FEATURE_FILE).write_bytes(feature_bytes)
    _write_json(
        destination / "manifest.json",
        {
            "schema_version": 1,
            "bundle_version": version,
            "as_of": AS_OF,
            "target": "30-day inactivity proxy",
            "source_window": "features strictly before 2011-11-01",
            "customer_count": len(selected),
            "model_versions": model_versions,
            "files": hashes,
        },
    )
    validate_bundle(destination)
    return destination
