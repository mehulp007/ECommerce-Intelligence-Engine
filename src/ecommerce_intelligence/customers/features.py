"""Point-in-time customer snapshots for churn and segmentation.

A snapshot at date T describes every customer with at least one purchase in
the 365 days before T, using purchases strictly before T only. The label, when
requested, looks forward: 1 if the customer makes no purchase in [T, T + 60).
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from ecommerce_intelligence.data.duck import sql_path
from ecommerce_intelligence.paths import ProjectPaths

ACTIVE_DAYS = 365
LABEL_DAYS = 60
TARGET = "inactive_next_60d"
INDEX_GROUPS = {
    "Ladieswear": "share_ladieswear",
    "Divided": "share_divided",
    "Menswear": "share_menswear",
    "Baby/Children": "share_kids",
    "Sport": "share_sport",
}
FEATURES = [
    "recency_days",
    "tenure_days",
    "visits_30d",
    "visits_90d",
    "visits_180d",
    "visits_365d",
    "spend_30d",
    "spend_90d",
    "spend_365d",
    "items_365d",
    "distinct_articles_365d",
    "avg_visit_spend",
    "items_per_visit",
    "mean_gap_days",
    "last_gap_days",
    "online_share",
    *INDEX_GROUPS.values(),
    "mean_price",
    "spend_trend",
    "visit_trend",
    "age",
]


def add_index_groups(con: duckdb.DuckDBPyConnection, paths: ProjectPaths) -> None:
    """Attach each article's index group name (Ladieswear, Divided, ...)."""
    con.execute(f"""
        CREATE OR REPLACE TABLE article_group AS
        SELECT m.article_idx, a.index_group_name
        FROM read_parquet({sql_path(paths.table("articles"))}) AS a
        JOIN article_map AS m USING (article_id)
        """)


def build_snapshot(
    con: duckdb.DuckDBPyConnection, as_of: str, *, label: bool = True
) -> pd.DataFrame:
    """Features for customers active in the year before `as_of`."""
    shares = ",\n".join(
        f"avg((g.index_group_name = '{name}')::INTEGER) AS {column}"
        for name, column in INDEX_GROUPS.items()
    )
    label_sql = (
        f"""
        , (NOT EXISTS (
            SELECT 1 FROM tx AS f
            WHERE f.customer_idx = s.customer_idx
              AND f.purchase_date >= DATE '{as_of}'
              AND f.purchase_date < DATE '{as_of}' + INTERVAL {LABEL_DAYS} DAY
        ))::TINYINT AS {TARGET}
        """
        if label
        else ""
    )
    frame = con.execute(f"""
        WITH history AS (
            SELECT t.*, date_diff('day', t.purchase_date, DATE '{as_of}') AS age_days
            FROM tx AS t
            WHERE t.purchase_date < DATE '{as_of}'
        ),
        active AS (
            SELECT DISTINCT customer_idx FROM history
            WHERE age_days <= {ACTIVE_DAYS}
        ),
        lines AS (
            SELECT h.* FROM history AS h SEMI JOIN active USING (customer_idx)
        ),
        visits AS (
            SELECT customer_idx, purchase_date,
                   date_diff('day', lag(purchase_date) OVER (
                       PARTITION BY customer_idx ORDER BY purchase_date
                   ), purchase_date) AS gap
            FROM (SELECT DISTINCT customer_idx, purchase_date FROM lines)
        ),
        gaps AS (
            SELECT customer_idx, avg(gap) AS mean_gap_days,
                   arg_max(gap, purchase_date) AS last_gap_days
            FROM visits GROUP BY customer_idx
        ),
        year AS (
            SELECT l.customer_idx,
                   count(*) AS items_365d,
                   count(DISTINCT l.article_idx) AS distinct_articles_365d,
                   avg(l.online::INTEGER) AS online_share,
                   avg(l.price) AS mean_price,
                   {shares}
            FROM lines AS l JOIN article_group AS g USING (article_idx)
            WHERE l.age_days <= {ACTIVE_DAYS}
            GROUP BY l.customer_idx
        ),
        summary AS (
            SELECT customer_idx,
                   min(age_days) AS recency_days,
                   max(age_days) AS tenure_days,
                   count(DISTINCT purchase_date) FILTER (WHERE age_days <= 30)
                       AS visits_30d,
                   count(DISTINCT purchase_date) FILTER (WHERE age_days <= 90)
                       AS visits_90d,
                   count(DISTINCT purchase_date) FILTER (WHERE age_days <= 180)
                       AS visits_180d,
                   count(DISTINCT purchase_date) FILTER (WHERE age_days <= 365)
                       AS visits_365d,
                   coalesce(sum(price) FILTER (WHERE age_days <= 30), 0) AS spend_30d,
                   coalesce(sum(price) FILTER (WHERE age_days <= 90), 0) AS spend_90d,
                   coalesce(sum(price) FILTER (WHERE age_days <= 365), 0)
                       AS spend_365d
            FROM lines GROUP BY customer_idx
        )
        SELECT s.*, y.items_365d, y.distinct_articles_365d,
               s.spend_365d / s.visits_365d AS avg_visit_spend,
               y.items_365d / s.visits_365d AS items_per_visit,
               g.mean_gap_days, g.last_gap_days, y.online_share,
               y.share_ladieswear, y.share_divided, y.share_menswear,
               y.share_kids, y.share_sport, y.mean_price,
               s.spend_90d / (s.spend_365d / 4 + 1e-6) AS spend_trend,
               s.visits_90d / (s.visits_365d / 4.0) AS visit_trend,
               c.age
               {label_sql}
        FROM summary AS s
        JOIN year AS y USING (customer_idx)
        JOIN gaps AS g USING (customer_idx)
        JOIN customer_attrs AS c USING (customer_idx)
        ORDER BY s.customer_idx
        """).df()
    for column in FEATURES:
        frame[column] = frame[column].astype(np.float32)
    frame["customer_idx"] = frame["customer_idx"].astype(np.int32)
    frame.insert(1, "snapshot_date", pd.Timestamp(as_of))
    return frame
