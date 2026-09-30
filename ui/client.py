"""Server-side HTTP client for the historical inference API."""

from __future__ import annotations

from dataclasses import dataclass

import httpx


class ApiError(Exception):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass(frozen=True)
class ApiClient:
    base_url: str
    api_key: str
    timeout_seconds: float = 12.0
    transport: httpx.BaseTransport | None = None

    def _request(self, method: str, path: str, payload: dict | None = None) -> dict:
        url = f"{self.base_url.rstrip('/')}{path}"
        headers = {"Authorization": f"Bearer {self.api_key}"}
        try:
            with httpx.Client(
                timeout=self.timeout_seconds, transport=self.transport
            ) as client:
                response = client.request(method, url, headers=headers, json=payload)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as error:
            status = error.response.status_code
            if status == 401:
                message = "The API key was rejected. Check the dashboard's server-side secret."
            elif status == 404:
                message = "This customer is outside the November 2011 demo cohort. Choose a sample ID."
            elif status == 422:
                message = (
                    "Enter a numeric customer ID and request at most five products."
                )
            else:
                message = f"The API returned an error ({status}). Please try again."
            raise ApiError(message, status_code=status) from error
        except (httpx.TimeoutException, httpx.NetworkError) as error:
            raise ApiError(
                "The API is unavailable. Check that the local API is running, then retry."
            ) from error
        except (ValueError, TypeError) as error:
            raise ApiError("The API returned an unreadable response.") from error

    def health(self) -> dict:
        return self._request("GET", "/health")

    def demo_customers(self) -> dict:
        return self._request("GET", "/demo/customers")

    def prediction(self, customer_id: str) -> dict:
        return self._request("POST", "/predict/churn", {"customer_id": customer_id})

    def recommendations(self, customer_id: str, *, limit: int = 5) -> dict:
        return self._request(
            "POST", "/recommend", {"customer_id": customer_id, "limit": limit}
        )
