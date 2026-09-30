from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from api.main import create_app
from ecommerce_intelligence.serving.bundle import validate_bundle

ROOT = Path(__file__).resolve().parents[1]
BUNDLES = ROOT / "artifacts" / "bundles"
BUNDLE = next(iter(sorted(BUNDLES.glob("uci-2011-11-01-*"))), None)


async def request(
    app, method: str, path: str, payload: dict | None = None, *, key: str | None = None
):
    body = json.dumps(payload).encode() if payload is not None else b""
    headers = [(b"content-type", b"application/json")]
    if key is not None:
        headers.append((b"authorization", f"Bearer {key}".encode()))
    scope = {
        "type": "http",
        "asgi": {"version": "3.0"},
        "http_version": "1.1",
        "method": method,
        "scheme": "http",
        "path": path,
        "raw_path": path.encode(),
        "query_string": b"",
        "headers": headers,
        "client": ("127.0.0.1", 12345),
        "server": ("test", 80),
    }
    sent = []
    received = False

    async def receive():
        nonlocal received
        if not received:
            received = True
            return {"type": "http.request", "body": body, "more_body": False}
        return {"type": "http.disconnect"}

    async def send(message):
        sent.append(message)

    await app(scope, receive, send)
    status = next(
        message["status"]
        for message in sent
        if message["type"] == "http.response.start"
    )
    response = b"".join(
        message.get("body", b"")
        for message in sent
        if message["type"] == "http.response.body"
    )
    return status, json.loads(response)


@pytest.mark.skipif(BUNDLE is None, reason="Run serving promotion first")
def test_customer_inference_and_api_contract() -> None:
    async def exercise():
        app = create_app(bundle_dir=BUNDLE, api_key="test-secret")
        async with app.router.lifespan_context(app):
            health_status, health = await request(app, "GET", "/health")
            assert health_status == 200, health
            assert health["bundle_version"] == BUNDLE.name
            assert (await request(app, "GET", "/demo/customers"))[0] == 401
            _, demo = await request(app, "GET", "/demo/customers", key="test-secret")
            customer_id = demo["customer_ids"][0]
            assert (
                await request(
                    app, "POST", "/predict/churn", {"customer_id": customer_id}
                )
            )[0] == 401
            status, result = await request(
                app,
                "POST",
                "/predict/churn",
                {"customer_id": customer_id},
                key="test-secret",
            )
            assert status == 200
            assert result["as_of"] == "2011-11-01"
            assert result["target"] == "30-day inactivity proxy"
            assert 0 <= result["inactivity_probability_30d"] <= 1
            assert len(result["drivers"]) == 7
            assert result["persona"]["name"]
            assert np.isfinite(result["base_log_odds"])
            raw_margin = app.state.engine.model.predict(
                app.state.engine.features.loc[
                    [customer_id], app.state.engine.model_features
                ],
                output_margin=True,
            )[0]
            assert result["base_log_odds"] + sum(
                item["log_odds_contribution"] for item in result["drivers"]
            ) == pytest.approx(float(raw_margin), abs=1e-4)

            status, recommendation = await request(
                app,
                "POST",
                "/recommend",
                {"customer_id": customer_id, "limit": 5},
                key="test-secret",
            )
            assert status == 200
            products = recommendation["products"]
            assert len(products) == 5
            assert len({item["stock_code"] for item in products}) == 5
            service = app.state.engine
            seen = service.recommender.seen_item_ids(customer_id)
            indices = {
                service.recommender.stock_codes.index(item["stock_code"])
                for item in products
            }
            assert not indices & seen

            cold_ids = sorted(
                set(service.features.index) - set(service.recommender.customer_ids)
            )
            assert cold_ids
            cold_status, cold_result = await request(
                app,
                "POST",
                "/recommend",
                {"customer_id": cold_ids[0]},
                key="test-secret",
            )
            assert cold_status == 200
            assert {item["source"] for item in cold_result["products"]} == {
                "popular_fallback"
            }

            for route in ("/predict/churn", "/recommend"):
                assert (
                    await request(
                        app,
                        "POST",
                        route,
                        {"customer_id": "999999999"},
                        key="test-secret",
                    )
                )[0] == 404
                assert (
                    await request(
                        app, "POST", route, {"customer_id": "bad-id"}, key="test-secret"
                    )
                )[0] == 422
            assert (
                await request(
                    app,
                    "POST",
                    "/recommend",
                    {"customer_id": customer_id, "limit": 6},
                    key="test-secret",
                )
            )[0] == 422

    asyncio.run(exercise())


@pytest.mark.skipif(BUNDLE is None, reason="Run serving promotion first")
def test_bundle_rejects_changed_files(tmp_path: Path) -> None:
    copy = tmp_path / BUNDLE.name
    shutil.copytree(BUNDLE, copy)
    assert json.loads((copy / "manifest.json").read_text())["customer_count"] > 0
    validate_bundle(copy)
    with (copy / "feature_schema.json").open("a", encoding="utf-8") as stream:
        stream.write(" ")
    with pytest.raises(ValueError, match="checksum"):
        validate_bundle(copy)
