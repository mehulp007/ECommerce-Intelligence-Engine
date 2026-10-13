from __future__ import annotations

import pandas as pd
import pytest
from conftest import Encoder16
from fastapi.testclient import TestClient
from pydantic import SecretStr

from ecommerce_intelligence.api.app import create_app
from ecommerce_intelligence.api.settings import Settings
from ecommerce_intelligence.customers import campaign

KEY = "test-key"
AUTH = {"Authorization": f"Bearer {KEY}"}


@pytest.fixture(scope="module")
def client(world):
    settings = Settings(eie_api_key=SecretStr(KEY), _env_file=None)
    with TestClient(create_app(settings, world, encoder_factory=Encoder16)) as c:
        yield c


def _scores(world) -> pd.DataFrame:
    return pd.read_parquet(world.root / "data" / "serving" / "customer_scores.parquet")


def test_health_is_public_and_reports_counts(client) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ready" and body["text_search"] is True
    assert body["counts"]["customers"] == 400
    assert body["counts"]["articles"] == 60


def test_data_endpoints_require_the_bearer_key(client, world) -> None:
    assert client.get("/api/segments").status_code == 401
    assert (
        client.get("/api/segments", headers={"Authorization": "Bearer x"}).status_code
        == 401
    )
    assert client.get("/api/segments", headers=AUTH).status_code == 200
    unconfigured = Settings(eie_api_key=None, _env_file=None)
    with TestClient(create_app(unconfigured, world, encoder_factory=Encoder16)) as c:
        assert c.get("/api/segments", headers=AUTH).status_code == 503


def test_customer_360_matches_the_scores(client, world) -> None:
    scores = _scores(world)
    active = scores.dropna(subset=["churn_probability"]).iloc[0]
    body = client.get(f"/api/customers/{active.customer_id}", headers=AUTH).json()
    assert body["churn_probability"] == pytest.approx(
        active.churn_probability, rel=1e-5
    )
    assert body["segment_name"] == active.segment_name
    assert [d["feature"] for d in body["drivers"]][0] == active.driver1_feature
    assert body["summary"]["purchases"] > 0

    history = client.get(
        f"/api/customers/{active.customer_id}/history?limit=5", headers=AUTH
    ).json()
    dates = [row["purchase_date"] for row in history]
    assert 0 < len(history) <= 5 and dates == sorted(dates, reverse=True)

    recs = client.get(
        f"/api/customers/{active.customer_id}/recommendations", headers=AUTH
    ).json()
    assert [r["rank"] for r in recs] == list(range(1, 13))
    assert all(r["article"]["article_id"] == r["article_id"] for r in recs)


def test_lapsed_customers_are_served_without_a_churn_score(client, world) -> None:
    lapsed = _scores(world).query("risk_band == 'lapsed'")
    if lapsed.empty:
        pytest.skip("synthetic data produced no lapsed customers")
    body = client.get(
        f"/api/customers/{lapsed.customer_id.iloc[0]}", headers=AUTH
    ).json()
    assert body["churn_probability"] is None and body["segment_id"] == -1


def test_unknown_and_malformed_ids(client) -> None:
    assert client.get("/api/customers/" + "f" * 64, headers=AUTH).status_code == 404
    assert client.get("/api/customers/not-a-hash", headers=AUTH).status_code == 422
    assert client.get("/api/products/0000000001", headers=AUTH).status_code == 404
    assert client.get("/api/products/12", headers=AUTH).status_code == 422
    missing = client.get(
        "/api/customers/" + "f" * 64 + "/recommendations", headers=AUTH
    )
    assert missing.status_code == 404


def test_customer_sample_respects_filters(client) -> None:
    rows = client.get(
        "/api/customers/sample?risk_band=high&limit=5", headers=AUTH
    ).json()
    assert rows and all(r["risk_band"] == "high" for r in rows)
    again = client.get("/api/customers/sample?risk_band=high&limit=5", headers=AUTH)
    assert again.json() == rows  # Same seed, same sample.
    prefix = rows[0]["customer_id"][:6]
    found = client.get(f"/api/customers/lookup?prefix={prefix}", headers=AUTH).json()
    assert rows[0]["customer_id"] in found


def test_catalog_browse_detail_and_similar(client) -> None:
    page = client.get("/api/products?limit=5&offset=0", headers=AUTH).json()
    assert page["total"] == 60 and len(page["items"]) == 5
    filtered = client.get(
        "/api/products?index_group=Menswear&limit=100", headers=AUTH
    ).json()
    assert filtered["total"] == 12
    assert all(i["index_group_name"] == "Menswear" for i in filtered["items"])
    first = page["items"][0]["article_id"]
    detail = client.get(f"/api/products/{first}", headers=AUTH).json()
    assert len(detail["weekly_sales_12w"]) == 12
    similar = client.get(f"/api/products/{first}/similar?k=4", headers=AUTH).json()
    assert len(similar) == 4 and first not in {
        h["article"]["article_id"] for h in similar
    }
    facets = client.get("/api/products/facets", headers=AUTH).json()
    assert len(facets["index_groups"]) == 5


def test_text_search_and_images(client, world) -> None:
    hits = client.get("/api/search?q=black%20dress&k=5", headers=AUTH).json()
    assert len(hits) == 5 and all("article" in h for h in hits)
    assert client.get("/api/search?q=a", headers=AUTH).status_code == 422
    article = pd.read_parquet(world.table("articles"))["article_id"].iloc[0]
    image = client.get(f"/images/{article}.webp")
    assert image.status_code == 200 and image.headers["content-type"] == "image/webp"
    assert client.get("/images/0000000001.webp").status_code == 404


def test_campaign_plan_matches_the_simulator(client, world) -> None:
    request = {"budget": 0.5, "cost_per_contact": 0.005, "uplift": 0.2, "top": 3}
    body = client.post("/api/campaigns/plan", json=request, headers=AUTH).json()
    expected = campaign.plan(
        _scores(world).dropna(subset=["churn_probability"]),
        budget=0.5,
        cost_per_contact=0.005,
        uplift=0.2,
    )
    assert body["contacted"] == expected.contacted
    assert body["expected_net"] == pytest.approx(expected.expected_net)
    assert len(body["top_customers"]) == min(3, expected.contacted)
    bad = client.post("/api/campaigns/plan", json={"uplift": 1.5}, headers=AUTH)
    assert bad.status_code == 422


def test_overview_models_and_openapi(client) -> None:
    overview = client.get("/api/overview", headers=AUTH).json()
    assert overview["kpis"]["purchases"] > 0 and overview["weekly"]
    models = client.get("/api/models", headers=AUTH).json()
    assert {"recommender_metrics", "customer_metrics"} <= set(models)
    schema = client.get("/openapi.json").json()
    assert "/api/campaigns/plan" in schema["paths"]
    assert "Customer" in schema["components"]["schemas"]
