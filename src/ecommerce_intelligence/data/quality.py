"""Profile and validate the ingested H&M tables."""

from __future__ import annotations

import json
from pathlib import Path

import duckdb

from ecommerce_intelligence.data.duck import connect, sql_path
from ecommerce_intelligence.paths import ProjectPaths

FIRST_DATE = "2018-09-20"
LAST_DATE = "2020-09-22"


def _one(connection: duckdb.DuckDBPyConnection, query: str) -> dict:
    cursor = connection.execute(query)
    names = [column[0] for column in cursor.description]
    return dict(zip(names, cursor.fetchone(), strict=True))


def _many(connection: duckdb.DuckDBPyConnection, query: str) -> list[dict]:
    cursor = connection.execute(query)
    names = [column[0] for column in cursor.description]
    return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]


def _jsonable(value):
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, float):
        return round(value, 6)
    return value


def profile(paths: ProjectPaths) -> dict:
    connection = connect(paths)
    try:
        for name in ("transactions", "customers", "articles"):
            connection.execute(
                f"CREATE VIEW {name} AS "
                f"SELECT * FROM read_parquet({sql_path(paths.table(name))})"
            )
        transactions = _one(
            connection,
            """
            SELECT
                count(*) AS rows,
                min(purchase_date) AS first_date,
                max(purchase_date) AS last_date,
                count(DISTINCT customer_id) AS customers,
                count(DISTINCT article_id) AS articles,
                sum(price) AS revenue_units,
                min(price) AS price_min,
                quantile_cont(price, 0.5) AS price_median,
                quantile_cont(price, 0.99) AS price_p99,
                max(price) AS price_max,
                count(*) FILTER (WHERE price <= 0) AS nonpositive_price_rows,
                count(*) FILTER (
                    WHERE purchase_date IS NULL OR customer_id IS NULL
                        OR article_id IS NULL OR price IS NULL
                        OR sales_channel_id IS NULL
                ) AS rows_with_nulls,
                count(*) FILTER (WHERE sales_channel_id = 2) / count(*)
                    AS online_share
            FROM transactions
            """,
        )
        transactions["repeated_line_rows"] = connection.execute("""
            SELECT count(*) - (
                SELECT count(*) FROM (SELECT DISTINCT * FROM transactions)
            ) FROM transactions
            """).fetchone()[0]
        transactions["orphan_customer_rows"] = connection.execute("""
            SELECT count(*) FROM transactions AS t
            ANTI JOIN customers AS c USING (customer_id)
            """).fetchone()[0]
        transactions["orphan_article_rows"] = connection.execute("""
            SELECT count(*) FROM transactions AS t
            ANTI JOIN articles AS a USING (article_id)
            """).fetchone()[0]
        transactions["weekly"] = _many(
            connection,
            """
            SELECT
                CAST(date_trunc('week', purchase_date) AS DATE) AS week_start,
                count(*) AS rows,
                count(DISTINCT customer_id) AS customers,
                sum(price) AS revenue_units
            FROM transactions
            GROUP BY ALL
            ORDER BY week_start
            """,
        )

        customers = _one(
            connection,
            """
            SELECT
                count(*) AS rows,
                count(DISTINCT customer_id) AS distinct_ids,
                count(*) FILTER (WHERE age IS NULL) AS missing_age,
                min(age) AS age_min,
                quantile_cont(age, 0.5) AS age_median,
                max(age) AS age_max,
                avg(fashion_news_subscriber::INTEGER) AS fashion_news_share,
                avg(active::INTEGER) AS active_share
            FROM customers
            """,
        )
        customers["with_transactions"] = connection.execute("""
            SELECT count(*) FROM customers
            SEMI JOIN (SELECT DISTINCT customer_id FROM transactions)
                USING (customer_id)
            """).fetchone()[0]
        customers["club_member_status"] = dict(connection.execute("""
                SELECT coalesce(club_member_status, 'missing'), count(*)
                FROM customers GROUP BY ALL ORDER BY 2 DESC
                """).fetchall())
        customers["fashion_news_frequency"] = dict(connection.execute("""
                SELECT coalesce(fashion_news_frequency, 'missing'), count(*)
                FROM customers GROUP BY ALL ORDER BY 2 DESC
                """).fetchall())

        articles = _one(
            connection,
            """
            SELECT
                count(*) AS rows,
                count(DISTINCT article_id) AS distinct_ids,
                count(DISTINCT product_code) AS products,
                count(*) FILTER (WHERE detail_desc IS NULL) AS missing_description
            FROM articles
            """,
        )
        articles["never_sold"] = connection.execute("""
            SELECT count(*) FROM articles
            ANTI JOIN (SELECT DISTINCT article_id FROM transactions) USING (article_id)
            """).fetchone()[0]
        articles["product_groups"] = dict(connection.execute("""
                SELECT product_group_name, count(*) FROM articles
                GROUP BY ALL ORDER BY 2 DESC
                """).fetchall())
    finally:
        connection.close()

    if paths.thumbnails.is_dir():
        articles["with_thumbnail"] = sum(1 for _ in paths.thumbnails.rglob("*.webp"))

    return _jsonable(
        {
            "tables": {
                name: {"parquet_bytes": paths.table(name).stat().st_size}
                for name in ("transactions", "customers", "articles")
            },
            "transactions": transactions,
            "customers": customers,
            "articles": articles,
        }
    )


def validate(report: dict) -> list[str]:
    """Return failed integrity checks; an empty list means the data is usable."""
    failures = []
    t, c, a = report["transactions"], report["customers"], report["articles"]
    if t["rows"] == 0:
        failures.append("transactions table is empty")
    if t["rows_with_nulls"]:
        failures.append(f"{t['rows_with_nulls']} transaction rows contain nulls")
    if t["orphan_customer_rows"]:
        failures.append(f"{t['orphan_customer_rows']} transactions lack a customer")
    if t["orphan_article_rows"]:
        failures.append(f"{t['orphan_article_rows']} transactions lack an article")
    if t["nonpositive_price_rows"]:
        failures.append(f"{t['nonpositive_price_rows']} transactions have price <= 0")
    if c["rows"] != c["distinct_ids"]:
        failures.append("customer IDs are not unique")
    if a["rows"] != a["distinct_ids"]:
        failures.append("article IDs are not unique")
    if t["rows"] and not (FIRST_DATE <= t["first_date"] <= t["last_date"] <= LAST_DATE):
        failures.append(
            f"transaction dates {t['first_date']}..{t['last_date']} fall outside "
            f"{FIRST_DATE}..{LAST_DATE}"
        )
    return failures


def write_report(paths: ProjectPaths) -> tuple[Path, dict, list[str]]:
    report = profile(paths)
    failures = validate(report)
    report["checks"] = {"passed": not failures, "failures": failures}
    paths.reports.mkdir(parents=True, exist_ok=True)
    target = paths.reports / "data_quality.json"
    target.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return target, report, failures
