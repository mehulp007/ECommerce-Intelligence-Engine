"""Load transactions into DuckDB with compact integer customer and article keys."""

from __future__ import annotations

import duckdb

from ecommerce_intelligence.data.duck import connect, sql_path
from ecommerce_intelligence.paths import ProjectPaths

# Category codes are re-numbered 0..n-1 so tree models see compact categories.
CATEGORY_CODES = ", ".join(
    f"(dense_rank() OVER (ORDER BY a.{column}) - 1)::INTEGER AS {column}"
    for column in (
        "product_type_no",
        "graphical_appearance_no",
        "colour_group_code",
        "perceived_colour_master_id",
        "department_no",
        "index_group_no",
        "section_no",
        "garment_group_no",
    )
)
AGE_BUCKET_SQL = """
    CASE
        WHEN age IS NULL THEN 0
        WHEN age < 25 THEN 1
        WHEN age < 35 THEN 2
        WHEN age < 45 THEN 3
        WHEN age < 55 THEN 4
        ELSE 5
    END
"""


def open_context(paths: ProjectPaths) -> duckdb.DuckDBPyConnection:
    """Create an in-memory database holding the tables every stage needs.

    Tables: customer_map, article_map, tx, article_attrs, customer_attrs.
    """
    for name in ("transactions", "customers", "articles"):
        if not paths.table(name).is_file():
            raise FileNotFoundError("Run `eie data ingest` before the recommender")
    con = connect(paths)
    con.execute(f"""
        CREATE TABLE customer_map AS
        SELECT (row_number() OVER (ORDER BY customer_id) - 1)::INTEGER AS customer_idx,
               customer_id, age
        FROM read_parquet({sql_path(paths.table("customers"))})
        """)
    con.execute(f"""
        CREATE TABLE article_map AS
        SELECT (row_number() OVER (ORDER BY article_id) - 1)::INTEGER AS article_idx,
               article_id
        FROM read_parquet({sql_path(paths.table("articles"))})
        """)
    con.execute(f"""
        CREATE TABLE tx AS
        SELECT c.customer_idx, a.article_idx, t.purchase_date,
               t.price::FLOAT AS price, (t.sales_channel_id = 2) AS online
        FROM read_parquet({sql_path(paths.table("transactions"))}) AS t
        JOIN customer_map AS c USING (customer_id)
        JOIN article_map AS a USING (article_id)
        """)
    con.execute(f"""
        CREATE TABLE article_attrs AS
        SELECT m.article_idx, a.product_code::INTEGER AS product_code,
               {CATEGORY_CODES}
        FROM read_parquet({sql_path(paths.table("articles"))}) AS a
        JOIN article_map AS m USING (article_id)
        """)
    con.execute(f"""
        CREATE TABLE customer_attrs AS
        SELECT customer_idx, age, ({AGE_BUCKET_SQL})::TINYINT AS age_bucket
        FROM customer_map
        """)
    return con


def week_labels(
    con: duckdb.DuckDBPyConnection, as_of: str, days: int = 7
) -> dict[int, set[int]]:
    """Articles each customer bought in [as_of, as_of + days)."""
    rows = con.execute(f"""
        SELECT customer_idx, list(DISTINCT article_idx)
        FROM tx
        WHERE purchase_date >= DATE '{as_of}'
          AND purchase_date < DATE '{as_of}' + INTERVAL {days} DAY
        GROUP BY customer_idx
        """).fetchall()
    return {customer: set(items) for customer, items in rows}
