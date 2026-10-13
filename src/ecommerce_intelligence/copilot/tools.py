"""Functions the copilot model may call, and the UI events each one produces.

Every tool returns two things: a compact payload for the model (kept small to
save tokens) and events for the interface (full tables, charts, cards).
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from ecommerce_intelligence.copilot.sandbox import QueryResult, Sandbox, UnsafeQuery
from ecommerce_intelligence.search.encoder import Encoder
from ecommerce_intelligence.search.index import ProductIndex
from ecommerce_intelligence.serving.store import NotFound, ServingStore

MODEL_ROWS = 40
UI_ROWS = 200
CUSTOMER_ID = re.compile(r"^[0-9a-f]{64}$")
ARTICLE_ID = re.compile(r"^\d{10}$")

DECLARATIONS = [
    {
        "name": "run_sql",
        "description": (
            "Run one read-only DuckDB SELECT over the allowed tables and return the "
            "rows. Aggregate in SQL; results are capped at 500 rows."
        ),
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "sql": {"type": "string", "description": "A single SELECT statement."},
                "purpose": {
                    "type": "string",
                    "description": "One short sentence: what this query answers.",
                },
            },
            "required": ["sql", "purpose"],
        },
    },
    {
        "name": "create_chart",
        "description": (
            "Show a chart of an earlier run_sql result in the interface. Use for "
            "trends (line) and comparisons (bar)."
        ),
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "result_id": {"type": "string"},
                "kind": {"type": "string", "enum": ["bar", "line"]},
                "x": {"type": "string", "description": "Column for the x axis."},
                "y": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "One or two numeric columns.",
                },
                "title": {"type": "string"},
            },
            "required": ["result_id", "kind", "x", "y", "title"],
        },
    },
    {
        "name": "get_customer",
        "description": (
            "Customer 360 for one 64-character customer ID: inactivity risk and its "
            "top drivers, 180-day value, segment, purchase summary, and the top "
            "recommendations with reasons."
        ),
        "parameters_json_schema": {
            "type": "object",
            "properties": {"customer_id": {"type": "string"}},
            "required": ["customer_id"],
        },
    },
    {
        "name": "search_products",
        "description": (
            "Find products matching a natural-language description using image and "
            "text embeddings."
        ),
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "k": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["query"],
        },
    },
    {
        "name": "similar_products",
        "description": "Products that look most like a given 10-digit article ID.",
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "article_id": {"type": "string"},
                "k": {"type": "integer", "minimum": 1, "maximum": 20},
            },
            "required": ["article_id"],
        },
    },
    {
        "name": "plan_campaign",
        "description": (
            "Plan a retention campaign: rank active customers by expected value "
            "saved (inactivity probability x 180-day value x uplift - cost) and "
            "report contacts, spend, expected value saved, net value, and ROI."
        ),
        "parameters_json_schema": {
            "type": "object",
            "properties": {
                "budget": {"type": "number", "minimum": 0},
                "cost_per_contact": {"type": "number", "exclusiveMinimum": 0},
                "uplift": {"type": "number", "exclusiveMinimum": 0, "maximum": 1},
                "segment_names": {"type": "array", "items": {"type": "string"}},
                "risk_bands": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["low", "medium", "high"]},
                },
            },
            "required": ["budget", "cost_per_contact", "uplift"],
        },
    },
]


class ToolError(ValueError):
    """A recoverable problem the model should see and correct."""


@dataclass
class Toolbox:
    sandbox: Sandbox
    store: ServingStore
    index: ProductIndex
    encoder: Callable[[], Encoder]
    results: dict[str, QueryResult] = field(default_factory=dict)

    def call(self, name: str, args: dict[str, Any]) -> tuple[dict, list[dict]]:
        handler = getattr(self, f"_{name}", None)
        if handler is None or name not in {d["name"] for d in DECLARATIONS}:
            raise ToolError(f"Unknown tool {name}")
        try:
            return handler(**args)
        except TypeError as error:
            raise ToolError(f"Bad arguments for {name}: {error}") from None

    def _run_sql(self, sql: str, purpose: str = "") -> tuple[dict, list[dict]]:
        try:
            result = self.sandbox.run(sql)
        except UnsafeQuery as error:
            raise ToolError(str(error)) from None
        except Exception as error:  # SQL errors are for the model to fix.
            raise ToolError(f"{type(error).__name__}: {error}") from None
        result_id = f"r{len(self.results) + 1}"
        self.results[result_id] = result
        event = {
            "type": "table",
            "result_id": result_id,
            "purpose": purpose,
            "sql": result.sql,
            "columns": result.columns,
            "rows": result.rows[:UI_ROWS],
            "row_count": len(result.rows),
            "truncated": result.truncated or len(result.rows) > UI_ROWS,
            "seconds": result.seconds,
        }
        payload = {
            "result_id": result_id,
            "columns": result.columns,
            "types": result.types,
            "row_count": len(result.rows),
            "truncated": result.truncated,
            "rows": result.rows[:MODEL_ROWS],
        }
        if len(result.rows) > MODEL_ROWS:
            payload["note"] = f"Only the first {MODEL_ROWS} rows are shown to you."
        return payload, [event]

    def _create_chart(
        self, result_id: str, kind: str, x: str, y: list[str], title: str
    ) -> tuple[dict, list[dict]]:
        result = self.results.get(result_id)
        if result is None:
            raise ToolError(f"No result {result_id}; run_sql first in this answer")
        if kind not in ("bar", "line"):
            raise ToolError("kind must be bar or line")
        y = [y] if isinstance(y, str) else list(y)
        missing = [c for c in [x, *y] if c not in result.columns]
        if missing or not 1 <= len(y) <= 2:
            raise ToolError(f"Columns {missing or y} invalid; have {result.columns}")
        positions = [result.columns.index(c) for c in [x, *y]]
        rows = [[row[i] for i in positions] for row in result.rows[:UI_ROWS]]
        event = {
            "type": "chart",
            "kind": kind,
            "title": title,
            "x": x,
            "y": y,
            "rows": rows,
        }
        return {"ok": True, "points": len(rows)}, [event]

    def _get_customer(self, customer_id: str) -> tuple[dict, list[dict]]:
        customer_id = customer_id.strip().lower()
        if not CUSTOMER_ID.match(customer_id):
            raise ToolError("Customer IDs are 64 lowercase hex characters")
        try:
            customer = self.store.customer(customer_id)
            recommendations = self.store.recommendations(customer_id)
        except NotFound:
            raise ToolError(f"Unknown customer {customer_id}") from None
        top = [
            {
                "rank": r["rank"],
                "article_id": r["article_id"],
                "name": (r["article"] or {}).get("prod_name"),
                "type": (r["article"] or {}).get("product_type_name"),
                "colour": (r["article"] or {}).get("colour_group_name"),
                "reason": r["reason"],
            }
            for r in recommendations[:6]
        ]
        payload = {
            key: customer.get(key)
            for key in (
                "customer_id",
                "churn_probability",
                "risk_band",
                "drivers",
                "segment_name",
                "p_alive",
                "expected_visits_180d",
                "clv_180d",
                "profile",
            )
        }
        # Label every activity figure with its period so it cannot be mixed up.
        payload["last_12_months"] = {
            "visits": customer.get("visits_365d"),
            "spend_price_units": customer.get("spend_365d"),
            "days_since_last_purchase": customer.get("recency_days"),
        }
        payload["all_time_2018_09_20_to_2020_09_22"] = customer.get("summary")
        payload["top_recommendations"] = top
        event = {
            "type": "customer",
            "customer": customer,
            "recommendations": recommendations,
        }
        return payload, [event]

    def _search_products(self, query: str, k: int = 8) -> tuple[dict, list[dict]]:
        if not query.strip():
            raise ToolError("Search text is empty")
        hits = self.index.search_text(self.encoder(), query, max(1, min(int(k), 20)))
        return self._products(hits, f"Products matching “{query}”")

    def _similar_products(self, article_id: str, k: int = 8) -> tuple[dict, list[dict]]:
        article_id = article_id.strip().zfill(10)
        if not ARTICLE_ID.match(article_id):
            raise ToolError("Article IDs are 10 digits")
        try:
            hits = self.index.similar(article_id, max(1, min(int(k), 20)))
        except KeyError:
            raise ToolError(f"Unknown article {article_id}") from None
        return self._products(hits, f"Similar to article {article_id}")

    def _products(self, hits, title: str) -> tuple[dict, list[dict]]:
        details = self.store.articles([h.article_id for h in hits])
        items = [{"score": h.score, "article": details[h.article_id]} for h in hits]
        payload = {
            "products": [
                {
                    "article_id": h.article_id,
                    "name": details[h.article_id]["prod_name"],
                    "type": details[h.article_id]["product_type_name"],
                    "colour": details[h.article_id]["colour_group_name"],
                    "sales_last_4_weeks": details[h.article_id]["sales_4w"],
                    "score": round(h.score, 3),
                }
                for h in hits
            ]
        }
        return payload, [{"type": "products", "title": title, "items": items}]

    def _plan_campaign(
        self,
        budget: float,
        cost_per_contact: float,
        uplift: float,
        segment_names: list[str] | None = None,
        risk_bands: list[str] | None = None,
    ) -> tuple[dict, list[dict]]:
        try:
            plan = self.store.plan_campaign(
                budget=float(budget),
                cost_per_contact=float(cost_per_contact),
                uplift=float(uplift),
                segment_names=segment_names,
                risk_bands=risk_bands,
                top=10,
            )
        except ValueError as error:
            raise ToolError(str(error)) from None
        payload = {k: v for k, v in plan.items() if k not in ("curve", "top_customers")}
        payload["inputs"] = {
            "budget": budget,
            "cost_per_contact": cost_per_contact,
            "uplift": uplift,
            "segment_names": segment_names,
            "risk_bands": risk_bands,
        }
        return payload, [
            {"type": "campaign", "plan": plan, "inputs": payload["inputs"]}
        ]
