"""Materialize the Parquet tables into one local DuckDB analytics warehouse."""

from __future__ import annotations

from ecommerce_intelligence.data.duck import connect, sql_path
from ecommerce_intelligence.paths import ProjectPaths

BASE_TABLES = ("transactions", "customers", "articles")

# A denormalized view for analytics questions about what sold, where, and to whom.
SALES_VIEW = """
CREATE OR REPLACE VIEW sales AS
SELECT
    t.purchase_date,
    t.customer_id,
    t.article_id,
    t.price,
    CASE t.sales_channel_id WHEN 2 THEN 'online' ELSE 'store' END AS sales_channel,
    a.prod_name,
    a.product_type_name,
    a.product_group_name,
    a.colour_group_name,
    a.department_name,
    a.index_group_name,
    a.section_name,
    a.garment_group_name,
    c.age AS customer_age,
    c.club_member_status
FROM transactions AS t
JOIN articles AS a USING (article_id)
LEFT JOIN customers AS c USING (customer_id)
"""


def build_warehouse(paths: ProjectPaths) -> dict[str, int]:
    missing = [name for name in BASE_TABLES if not paths.table(name).is_file()]
    if missing:
        raise FileNotFoundError(f"Run `eie data ingest` first; missing {missing}")
    partial = paths.warehouse.with_suffix(".duckdb.part")
    partial.unlink(missing_ok=True)
    connection = connect(paths, partial)
    try:
        counts = {}
        for name in BASE_TABLES:
            connection.execute(
                f"CREATE TABLE {name} AS "
                f"SELECT * FROM read_parquet({sql_path(paths.table(name))})"
            )
            counts[name] = connection.execute(
                f"SELECT count(*) FROM {name}"
            ).fetchone()[0]
        connection.execute("ALTER TABLE customers ADD PRIMARY KEY (customer_id)")
        connection.execute("ALTER TABLE articles ADD PRIMARY KEY (article_id)")
        connection.execute(SALES_VIEW)
        connection.execute("CHECKPOINT")
    finally:
        connection.close()
    paths.warehouse.unlink(missing_ok=True)
    partial.replace(paths.warehouse)
    return counts
