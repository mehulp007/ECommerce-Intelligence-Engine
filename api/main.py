"""FastAPI routes for a frozen historical UCI serving bundle."""

from __future__ import annotations

import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from ecommerce_intelligence.serving.engine import CustomerNotFound, InferenceEngine


class CustomerRequest(BaseModel):
    customer_id: str = Field(pattern=r"^[0-9]+$", min_length=1, max_length=20)


class RecommendationRequest(CustomerRequest):
    limit: int = Field(default=5, ge=1, le=5)


class Persona(BaseModel):
    id: int
    name: str


class Driver(BaseModel):
    feature: str
    value: float
    log_odds_contribution: float


class PredictionResponse(BaseModel):
    customer_id: str
    as_of: str
    bundle_version: str
    target: str
    inactivity_probability_30d: float
    risk_band: Literal["lower", "higher"]
    persona: Persona
    base_log_odds: float
    drivers: list[Driver]


class Product(BaseModel):
    stock_code: str
    description: str
    median_unit_price: float
    score: float | None
    source: Literal["bm25", "popular_fallback"]


class RecommendationResponse(BaseModel):
    customer_id: str
    as_of: str
    bundle_version: str
    products: list[Product]


bearer = HTTPBearer(auto_error=False)


def create_app(
    *, bundle_dir: Path | None = None, api_key: str | None = None
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        path = bundle_dir or Path(os.environ["EIE_BUNDLE_DIR"])
        key = api_key or os.environ["EIE_API_KEY"]
        if not key:
            raise ValueError("EIE_API_KEY must be nonempty")
        app.state.api_key = key
        app.state.engine = InferenceEngine(path)
        yield

    app = FastAPI(
        title="E-Commerce Intelligence Engine", version="1.0", lifespan=lifespan
    )

    def require_key(
        request: Request,
        credentials: HTTPAuthorizationCredentials | None = Depends(bearer),
    ) -> None:
        if credentials is None or not secrets.compare_digest(
            credentials.credentials, request.app.state.api_key
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API key"
            )

    def engine(request: Request) -> InferenceEngine:
        return request.app.state.engine

    @app.get("/health")
    def health(service: InferenceEngine = Depends(engine)) -> dict:
        return {"status": "ready", "bundle_version": service.manifest["bundle_version"]}

    @app.get("/demo/customers")
    def demo_customers(
        _: None = Depends(require_key),
        service: InferenceEngine = Depends(engine),
    ) -> dict:
        return {
            "customer_ids": service.demo_customers(),
            "as_of": service.manifest["as_of"],
        }

    @app.post("/predict/churn", response_model=PredictionResponse)
    def predict(
        payload: CustomerRequest,
        _: None = Depends(require_key),
        service: InferenceEngine = Depends(engine),
    ) -> dict:
        try:
            return service.predict(payload.customer_id)
        except CustomerNotFound as error:
            raise HTTPException(
                status_code=404, detail="Customer not in historical cohort"
            ) from error

    @app.post("/recommend", response_model=RecommendationResponse)
    def recommend(
        payload: RecommendationRequest,
        _: None = Depends(require_key),
        service: InferenceEngine = Depends(engine),
    ) -> dict:
        try:
            return service.recommend(payload.customer_id, payload.limit)
        except CustomerNotFound as error:
            raise HTTPException(
                status_code=404, detail="Customer not in historical cohort"
            ) from error

    return app


app = create_app()
