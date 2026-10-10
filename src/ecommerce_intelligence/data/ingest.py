"""Convert the H&M CSV tables into typed, compressed Parquet files."""

from __future__ import annotations

import duckdb

from ecommerce_intelligence.data.duck import connect, sql_path
from ecommerce_intelligence.paths import ProjectPaths

TRANSACTION_CSV_TYPES = {
    "t_dat": "DATE",
    "customer_id": "VARCHAR",
    "article_id": "VARCHAR",
    "price": "DOUBLE",
    "sales_channel_id": "TINYINT",
}
CUSTOMER_CSV_TYPES = {
    "customer_id": "VARCHAR",
    "FN": "DOUBLE",
    "Active": "DOUBLE",
    "club_member_status": "VARCHAR",
    "fashion_news_frequency": "VARCHAR",
    "age": "DOUBLE",
    "postal_code": "VARCHAR",
}
ARTICLE_CSV_TYPES = {
    "article_id": "VARCHAR",
    "product_code": "VARCHAR",
    "prod_name": "VARCHAR",
    "product_type_no": "INTEGER",
    "product_type_name": "VARCHAR",
    "product_group_name": "VARCHAR",
    "graphical_appearance_no": "INTEGER",
    "graphical_appearance_name": "VARCHAR",
    "colour_group_code": "INTEGER",
    "colour_group_name": "VARCHAR",
    "perceived_colour_value_id": "INTEGER",
    "perceived_colour_value_name": "VARCHAR",
    "perceived_colour_master_id": "INTEGER",
    "perceived_colour_master_name": "VARCHAR",
    "department_no": "INTEGER",
    "department_name": "VARCHAR",
    "index_code": "VARCHAR",
    "index_name": "VARCHAR",
    "index_group_no": "INTEGER",
    "index_group_name": "VARCHAR",
    "section_no": "INTEGER",
    "section_name": "VARCHAR",
    "garment_group_no": "INTEGER",
    "garment_group_name": "VARCHAR",
    "detail_desc": "VARCHAR",
}

# Article IDs are ten digits with a leading zero, so they stay text throughout.
TRANSACTIONS_SELECT = """
    SELECT
        t_dat AS purchase_date,
        customer_id,
        lpad(article_id, 10, '0') AS article_id,
        price,
        sales_channel_id
    FROM {source}
    ORDER BY purchase_date, customer_id, article_id
"""
CUSTOMERS_SELECT = """
    SELECT
        customer_id,
        coalesce(FN = 1, false) AS fashion_news_subscriber,
        coalesce(Active = 1, false) AS active,
        lower(nullif(trim(club_member_status), '')) AS club_member_status,
        CASE
            WHEN lower(trim(fashion_news_frequency)) IN ('none', '') THEN 'none'
            ELSE lower(trim(fashion_news_frequency))
        END AS fashion_news_frequency,
        CAST(age AS SMALLINT) AS age,
        postal_code
    FROM {source}
    ORDER BY customer_id
"""
ARTICLES_SELECT = """
    SELECT
        lpad(article_id, 10, '0') AS article_id,
        lpad(product_code, 7, '0') AS product_code,
        * EXCLUDE (article_id, product_code)
    FROM {source}
    ORDER BY article_id
"""
TABLES = {
    "transactions": (
        "transactions_train.csv",
        TRANSACTION_CSV_TYPES,
        TRANSACTIONS_SELECT,
    ),
    "customers": ("customers.csv", CUSTOMER_CSV_TYPES, CUSTOMERS_SELECT),
    "articles": ("articles.csv", ARTICLE_CSV_TYPES, ARTICLES_SELECT),
}


def _csv_source(path, types: dict[str, str]) -> str:
    columns = ", ".join(f"'{name}': '{kind}'" for name, kind in types.items())
    return (
        f"read_csv({sql_path(path)}, header = true, delim = ',', quote = '\"', "
        f"columns = {{{columns}}})"
    )


def convert_table(
    connection: duckdb.DuckDBPyConnection, paths: ProjectPaths, name: str
) -> int:
    csv_name, types, select = TABLES[name]
    source = _csv_source(paths.csv / csv_name, types)
    target = paths.table(name)
    partial = target.with_suffix(".parquet.part")
    partial.unlink(missing_ok=True)
    connection.execute(
        f"COPY ({select.format(source=source)}) TO {sql_path(partial)} "
        "(FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 1000000)"
    )
    partial.replace(target)
    return connection.execute(
        f"SELECT count(*) FROM read_parquet({sql_path(target)})"
    ).fetchone()[0]


def ingest(paths: ProjectPaths) -> dict[str, int]:
    paths.parquet.mkdir(parents=True, exist_ok=True)
    connection = connect(paths)
    try:
        return {name: convert_table(connection, paths, name) for name in TABLES}
    finally:
        connection.close()
