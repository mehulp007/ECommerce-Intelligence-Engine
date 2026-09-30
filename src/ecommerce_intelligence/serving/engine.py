"""Load one historical bundle and answer customer-ID inference requests."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from xgboost import DMatrix, XGBClassifier

from ecommerce_intelligence.models.churn import calibrated_probabilities
from ecommerce_intelligence.recommendations.recommender import ProductRecommender
from ecommerce_intelligence.serving.bundle import validate_bundle


class CustomerNotFound(LookupError):
    pass


class InferenceEngine:
    def __init__(self, bundle_dir: Path) -> None:
        self.manifest = validate_bundle(bundle_dir)
        schema = json.loads((bundle_dir / "feature_schema.json").read_text())
        self.model_features = schema["model_features"]
        self.rfm_features = schema["rfm_features"]
        self.model = XGBClassifier()
        self.model.load_model(bundle_dir / "churn_xgboost.json")
        self.calibrator = joblib.load(bundle_dir / "churn_calibrator.joblib")
        self.personas = joblib.load(bundle_dir / "rfm_personas.joblib")
        self.persona_names = json.loads((bundle_dir / "persona_names.json").read_text())
        self.recommender = ProductRecommender.load(bundle_dir)
        features = pd.read_parquet(bundle_dir / "customer_features.parquet")
        if (
            features["customer_id"].duplicated().any()
            or len(features) != self.manifest["customer_count"]
            or not features["snapshot_date"].eq(self.manifest["as_of"]).all()
        ):
            raise ValueError("Serving customer features failed validation")
        self.features = features.set_index("customer_id")

    def _customer(self, customer_id: str) -> pd.DataFrame:
        if customer_id not in self.features.index:
            raise CustomerNotFound(customer_id)
        return self.features.loc[[customer_id]]

    def demo_customers(self, limit: int = 12) -> list[str]:
        known = self.features.index.intersection(self.recommender.customer_ids)
        return sorted(known.tolist())[:limit]

    def predict(self, customer_id: str) -> dict:
        row = self._customer(customer_id)
        x = row.loc[:, self.model_features]
        probability = float(calibrated_probabilities(self.model, self.calibrator, x)[0])
        persona_id = int(self.personas.predict(row.loc[:, self.rfm_features])[0])
        contributions = self.model.get_booster().predict(
            DMatrix(x, feature_names=self.model_features), pred_contribs=True
        )[0]
        if len(contributions) != len(self.model_features) + 1:
            raise ValueError("Unexpected XGBoost contribution shape")
        drivers = [
            {
                "feature": feature,
                "value": float(row.iloc[0][feature]),
                "log_odds_contribution": float(value),
            }
            for feature, value in zip(
                self.model_features, contributions[:-1], strict=True
            )
        ]
        drivers.sort(key=lambda item: abs(item["log_odds_contribution"]), reverse=True)
        return {
            "customer_id": customer_id,
            "as_of": self.manifest["as_of"],
            "bundle_version": self.manifest["bundle_version"],
            "target": self.manifest["target"],
            "inactivity_probability_30d": probability,
            "risk_band": "higher" if probability >= 0.5 else "lower",
            "persona": {"id": persona_id, "name": self.persona_names[str(persona_id)]},
            "base_log_odds": float(contributions[-1]),
            "drivers": drivers,
        }

    def recommend(self, customer_id: str, limit: int) -> dict:
        self._customer(customer_id)
        return {
            "customer_id": customer_id,
            "as_of": self.manifest["as_of"],
            "bundle_version": self.manifest["bundle_version"],
            "products": self.recommender.recommend(customer_id, k=limit),
        }
