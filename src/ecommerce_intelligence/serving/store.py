"""Read-only access to the serving tables for the API.

One in-memory DuckDB database holds views over the Parquet outputs; each call
uses its own cursor, so concurrent requests do not share connection state.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import duckdb
import pandas as pd

from ecommerce_intelligence.customers import campaign
from ecommerce_intelligence.data.duck import sql_path
from ecommerce_intelligence.paths import ProjectPaths
from ecommerce_intelligence.serving.prepare import serving_dir

ARTICLE_COLUMNS = (
    "article_id, prod_name, product_type_name, product_group_name, "
    "colour_group_name, department_name, index_group_name, section_name, "
    "garment_group_name, detail_desc"
)
REPORTS = (
    "data_quality",
    "recommender_metrics",
    "search_quality",
    "customer_metrics",
    "thumbnails",
)


class NotFound(LookupError):
    pass


def _records(frame: pd.DataFrame) -> list[dict]:
    """JSON-safe records: NaN becomes None and timestamps become ISO dates."""
    frame = frame.astype(object).where(frame.notna(), None)
    return [
        {k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in row.items()}
        for row in frame.to_dict(orient="records")
    ]


class ServingStore:
    def __init__(self, paths: ProjectPaths):
        self.paths = paths
        self.dir = serving_dir(paths)
        sources = {
            "articles": paths.table("articles"),
            "customers": paths.table("customers"),
            "history": self.dir / "customer_history.parquet",
            "product_stats": self.dir / "product_stats.parquet",
            "recommendations": self.dir / "recommendations.parquet",
            "scores": self.dir / "customer_scores.parquet",
        }
        missing = [name for name, path in sources.items() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                f"Serving inputs missing: {missing}; run `eie serve prepare`"
            )
        self._con = duckdb.connect()
        self._lock = threading.Lock()
        for name, path in sources.items():
            self._con.execute(
                f"CREATE VIEW {name} AS SELECT * FROM read_parquet({sql_path(path)})"
            )
        self.overview = json.loads(
            (self.dir / "overview.json").read_text(encoding="utf-8")
        )
        self.manifest = json.loads(
            (self.dir / "manifest.json").read_text(encoding="utf-8")
        )
        self.reports = {
            name: json.loads(path.read_text(encoding="utf-8"))
            for name in REPORTS
            if (path := paths.reports / f"{name}.json").is_file()
        }
        # Campaign planning needs every active customer's probability and value.
        self.campaign_scores = self.query("""
            SELECT customer_id, churn_probability, clv_180d, segment_name, risk_band
            FROM scores WHERE churn_probability IS NOT NULL
            """)
        self.has_thumbnail = self._thumbnail_ids()

    def _thumbnail_ids(self) -> set[str]:
        root = self.paths.thumbnails
        return {p.stem for p in root.rglob("*.webp")} if root.is_dir() else set()

    def query(self, sql: str, params: list | None = None) -> pd.DataFrame:
        with self._lock:
            cursor = self._con.cursor()
        try:
            return cursor.execute(sql, params or []).df()
        finally:
            cursor.close()

    def thumbnail(self, article_id: str) -> Path | None:
        if article_id not in self.has_thumbnail:
            return None
        return self.paths.thumbnails / article_id[:3] / f"{article_id}.webp"

    def counts(self) -> dict:
        return self.query("""
            SELECT (SELECT count(*) FROM scores) AS customers,
                   (SELECT count(*) FROM scores WHERE churn_probability IS NOT NULL)
                       AS active_customers,
                   (SELECT count(*) FROM articles) AS articles,
                   (SELECT count(*) FROM recommendations) AS recommendations
            """).iloc[0].to_dict()

    # Articles -----------------------------------------------------------------

    def articles(self, article_ids: list[str]) -> dict[str, dict]:
        if not article_ids:
            return {}
        frame = self.query(
            f"""
            SELECT {ARTICLE_COLUMNS}, s.median_price, s.sales_4w, s.popularity_rank_4w
            FROM articles AS a LEFT JOIN product_stats AS s USING (article_id)
            WHERE list_contains(?, article_id)
            """,
            [article_ids],
        )
        found = {row["article_id"]: row for row in _records(frame)}
        for row in found.values():
            row["has_image"] = row["article_id"] in self.has_thumbnail
        return found

    def article(self, article_id: str) -> dict:
        frame = self.query(
            f"""
            SELECT {ARTICLE_COLUMNS}, s.* EXCLUDE (article_id)
            FROM articles AS a LEFT JOIN product_stats AS s USING (article_id)
            WHERE article_id = ?
            """,
            [article_id],
        )
        if frame.empty:
            raise NotFound(article_id)
        record = _records(frame)[0]
        weekly = record.get("weekly_sales_12w")
        record["weekly_sales_12w"] = (
            [int(v) for v in weekly] if weekly is not None else []
        )
        record["has_image"] = article_id in self.has_thumbnail
        return record

    def browse(
        self,
        *,
        index_group: str | None,
        product_group: str | None,
        colour: str | None,
        text: str | None,
        sort: str,
        limit: int,
        offset: int,
    ) -> tuple[int, list[dict]]:
        clauses, params = [], []
        for column, value in (
            ("index_group_name", index_group),
            ("product_group_name", product_group),
            ("colour_group_name", colour),
        ):
            if value:
                clauses.append(f"{column} = ?")
                params.append(value)
        if text:
            clauses.append("(prod_name ILIKE ? OR product_type_name ILIKE ?)")
            params += [f"%{text}%", f"%{text}%"]
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        order = {
            "popular": "s.sales_4w DESC NULLS LAST, article_id",
            "newest": "s.first_sale DESC NULLS LAST, article_id",
            "price_low": "s.median_price ASC NULLS LAST, article_id",
            "price_high": "s.median_price DESC NULLS LAST, article_id",
        }[sort]
        base = (
            "FROM articles AS a LEFT JOIN product_stats AS s USING (article_id) "
            + where
        )
        total = int(self.query(f"SELECT count(*) AS n {base}", params)["n"].iloc[0])
        frame = self.query(
            f"""
            SELECT {ARTICLE_COLUMNS}, s.median_price, s.sales_4w, s.popularity_rank_4w
            {base} ORDER BY {order} LIMIT ? OFFSET ?
            """,
            params + [limit, offset],
        )
        rows = _records(frame)
        for row in rows:
            row["has_image"] = row["article_id"] in self.has_thumbnail
        return total, rows

    def facets(self) -> dict:
        def values(column: str) -> list[str]:
            return self.query(
                f"SELECT {column} AS v, count(*) AS n FROM articles "
                "GROUP BY ALL ORDER BY n DESC"
            )["v"].tolist()

        return {
            "index_groups": values("index_group_name"),
            "product_groups": values("product_group_name"),
            "colours": values("colour_group_name"),
        }

    # Customers ----------------------------------------------------------------

    def customer(self, customer_id: str) -> dict:
        scores = self.query("SELECT * FROM scores WHERE customer_id = ?", [customer_id])
        if scores.empty:
            raise NotFound(customer_id)
        profile = self.query(
            """
            SELECT age, club_member_status, fashion_news_frequency, active,
                   fashion_news_subscriber
            FROM customers WHERE customer_id = ?
            """,
            [customer_id],
        )
        summary = self.query(
            """
            SELECT count(*) AS purchases, sum(price) AS spend_units,
                   count(DISTINCT purchase_date) AS visits,
                   min(purchase_date) AS first_purchase,
                   max(purchase_date) AS last_purchase,
                   avg((sales_channel_id = 2)::INTEGER) AS online_share
            FROM history WHERE customer_id = ?
            """,
            [customer_id],
        )
        record = _records(scores)[0]
        drivers = [
            {
                "feature": record.pop(f"driver{i}_feature"),
                "value": record.pop(f"driver{i}_value"),
                "log_odds": record.pop(f"driver{i}_log_odds"),
            }
            for i in (1, 2, 3)
        ]
        record["drivers"] = [d for d in drivers if d["feature"] is not None]
        record["profile"] = _records(profile)[0] if not profile.empty else {}
        record["summary"] = _records(summary)[0]
        return record

    def history(self, customer_id: str, limit: int) -> list[dict]:
        frame = self.query(
            """
            SELECT purchase_date, article_id, price,
                   CASE sales_channel_id WHEN 2 THEN 'online' ELSE 'store' END
                       AS channel
            FROM history WHERE customer_id = ?
            ORDER BY purchase_date DESC, article_id LIMIT ?
            """,
            [customer_id, limit],
        )
        rows = _records(frame)
        details = self.articles(sorted({r["article_id"] for r in rows}))
        for row in rows:
            row["article"] = details.get(row["article_id"])
        return rows

    def recommendations(self, customer_id: str) -> list[dict]:
        frame = self.query(
            """
            SELECT rank, article_id, score, reason FROM recommendations
            WHERE customer_id = ? ORDER BY rank
            """,
            [customer_id],
        )
        if (
            frame.empty
            and self.query(
                "SELECT 1 FROM scores WHERE customer_id = ?", [customer_id]
            ).empty
        ):
            raise NotFound(customer_id)
        rows = _records(frame)
        details = self.articles([r["article_id"] for r in rows])
        for row in rows:
            row["article"] = details.get(row["article_id"])
        return rows

    def sample_customers(
        self, *, segment_id: int | None, risk_band: str | None, limit: int, seed: int
    ) -> list[dict]:
        clauses, params = ["churn_probability IS NOT NULL"], []
        if segment_id is not None:
            clauses.append("segment_id = ?")
            params.append(segment_id)
        if risk_band:
            clauses.append("risk_band = ?")
            params.append(risk_band)
        # A deterministic hash order gives a varied but repeatable sample.
        frame = self.query(
            f"""
            SELECT customer_id, churn_probability, risk_band, segment_id, segment_name,
                   clv_180d, visits_365d, spend_365d, recency_days
            FROM scores WHERE {' AND '.join(clauses)}
            ORDER BY hash(customer_id || ?::VARCHAR) LIMIT ?
            """,
            params + [seed, limit],
        )
        return _records(frame)

    def customer_exists_prefix(self, prefix: str, limit: int) -> list[str]:
        frame = self.query(
            "SELECT customer_id FROM scores WHERE starts_with(customer_id, ?) "
            "ORDER BY customer_id LIMIT ?",
            [prefix, limit],
        )
        return frame["customer_id"].tolist()

    # Segments and campaigns ---------------------------------------------------

    def segments(self) -> list[dict]:
        frame = self.query("""
            SELECT segment_id, segment_name, count(*) AS customers,
                   avg(churn_probability) AS mean_churn_probability,
                   median(clv_180d) AS median_clv_180d,
                   sum(clv_180d) AS total_clv_180d,
                   median(visits_365d) AS median_visits_365d,
                   median(spend_365d) AS median_spend_365d
            FROM scores GROUP BY ALL ORDER BY segment_id
            """)
        return _records(frame)

    def plan_campaign(
        self,
        *,
        budget: float,
        cost_per_contact: float,
        uplift: float,
        segment_names: list[str] | None,
        risk_bands: list[str] | None,
        top: int,
    ) -> dict:
        scores = self.campaign_scores
        if segment_names:
            scores = scores.loc[scores["segment_name"].isin(segment_names)]
        if risk_bands:
            scores = scores.loc[scores["risk_band"].isin(risk_bands)]
        plan = campaign.plan(
            scores, budget=budget, cost_per_contact=cost_per_contact, uplift=uplift
        )
        curve = campaign.roi_curve(
            scores, cost_per_contact=cost_per_contact, uplift=uplift, points=40
        )
        chosen = plan.customers
        by_segment = (
            chosen.groupby("segment_name").size().sort_values(ascending=False).to_dict()
            if len(chosen)
            else {}
        )
        return {
            "eligible_customers": len(scores),
            "contacted": plan.contacted,
            "spend": plan.spend,
            "expected_value_saved": plan.expected_value_saved,
            "expected_net": plan.expected_net,
            "roi": plan.roi,
            "contacted_by_segment": {k: int(v) for k, v in by_segment.items()},
            "top_customers": _records(chosen.head(top)),
            "curve": _records(curve),
            # Customers whose expected saved value exceeds the cost of contact.
            "profitable_customers": int(
                (campaign.expected_value(scores, uplift) > cost_per_contact).sum()
            ),
        }
