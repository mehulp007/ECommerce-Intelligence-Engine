"""Register selected local models and export the frozen serving bundle."""

from __future__ import annotations

import json
import os
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient

from ecommerce_intelligence.serving.bundle import export_bundle, file_hash

MODELS = {
    "eie-churn": ("churn_mlflow_run_id", "churn_xgboost.json"),
    "eie-personas": ("personas", "rfm_personas.joblib"),
    "eie-recommender": (None, "bm25_recommender.npz"),
}


def promote(project_root: Path) -> Path:
    uri = f"sqlite:///{(project_root / 'mlflow.db').as_posix()}"
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    mlflow.set_tracking_uri(uri)
    mlflow.set_registry_uri(uri)
    client = MlflowClient()
    report = json.loads(
        (project_root / "reports" / "milestone2_metrics.json").read_text()
    )
    sources = project_root / "artifacts" / "serving"
    versions = {}
    for name, (run_key, filename) in MODELS.items():
        checksum = file_hash(sources / filename)
        if run_key == "personas":
            run_id = report["personas"]["mlflow_run_id"]
        elif run_key:
            run_id = report[run_key]
        else:
            existing = client.search_runs(
                experiment_ids=["0"],
                filter_string=f"tags.model_sha256 = '{checksum}'",
            )
            if existing:
                run_id = existing[0].info.run_id
            else:
                with mlflow.start_run(run_name="bm25-recommender-serving") as run:
                    mlflow.set_tag("model_sha256", checksum)
                    for artifact in (
                        "bm25_recommender.npz",
                        "recommender_schema.json",
                        "product_catalog.parquet",
                        "recommender_mappings.json",
                        "recommender_user_items.npz",
                    ):
                        mlflow.log_artifact(str(sources / artifact))
                    run_id = run.info.run_id
        try:
            client.get_registered_model(name)
        except mlflow.exceptions.MlflowException:
            client.create_registered_model(name)
        existing_versions = client.search_model_versions(f"name = '{name}'")
        matching = next(
            (v for v in existing_versions if v.tags.get("source_sha256") == checksum),
            None,
        )
        if matching is None:
            artifact_uri = client.get_run(run_id).info.artifact_uri
            matching = client.create_model_version(
                name=name,
                source=f"{artifact_uri}/{filename}",
                run_id=run_id,
                tags={"source_sha256": checksum},
            )
        client.set_registered_model_alias(name, "champion", matching.version)
        versions[name] = int(matching.version)
    return export_bundle(project_root, versions)


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    destination = promote(root)
    print(f"Promoted serving bundle: {destination.name}")


if __name__ == "__main__":
    main()
