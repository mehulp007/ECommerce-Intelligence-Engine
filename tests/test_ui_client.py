from __future__ import annotations

import httpx
import pytest

from ui.client import ApiClient, ApiError


def test_ui_client_sends_key_only_to_api_and_reads_customer_flow() -> None:
    calls = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ready", "bundle_version": "v1"})
        if request.url.path == "/demo/customers":
            return httpx.Response(200, json={"customer_ids": ["12345"]})
        if request.url.path == "/predict/churn":
            return httpx.Response(
                200, json={"customer_id": "12345", "risk_band": "higher"}
            )
        return httpx.Response(200, json={"products": [{"stock_code": "10001"}]})

    client = ApiClient(
        "https://api.example.test/",
        "server-secret",
        transport=httpx.MockTransport(respond),
    )
    assert client.health()["status"] == "ready"
    assert client.demo_customers()["customer_ids"] == ["12345"]
    assert client.prediction("12345")["risk_band"] == "higher"
    assert len(client.recommendations("12345")["products"]) == 1
    assert [request.url.path for request in calls] == [
        "/health",
        "/demo/customers",
        "/predict/churn",
        "/recommend",
    ]
    assert all(
        request.headers["authorization"] == "Bearer server-secret" for request in calls
    )
    assert all("server-secret" not in str(request.url) for request in calls)


@pytest.mark.parametrize(
    ("status", "expected"),
    [(401, "key"), (404, "outside"), (422, "numeric"), (503, "503")],
)
def test_ui_client_turns_api_failures_into_actionable_messages(
    status: int, expected: str
) -> None:
    client = ApiClient(
        "https://api.example.test",
        "server-secret",
        transport=httpx.MockTransport(lambda _: httpx.Response(status)),
    )
    with pytest.raises(ApiError, match=expected):
        client.prediction("12345")


def test_ui_client_handles_connection_failure() -> None:
    def unavailable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    client = ApiClient(
        "https://api.example.test",
        "server-secret",
        transport=httpx.MockTransport(unavailable),
    )
    with pytest.raises(ApiError, match="unavailable"):
        client.demo_customers()
