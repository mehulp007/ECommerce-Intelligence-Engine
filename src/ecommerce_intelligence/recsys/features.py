"""Stage two inputs: describe each (customer, candidate article) pair.

All statistics use purchases strictly before `as_of`. Customer and article
identifiers are carried for joining but are never model features.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

SOURCE_FEATURES = [
    "repurchase_rank",
    "popular_rank",
    "age_popular_rank",
    "cooc_score",
    "cooc_rank",
    "als_score",
    "als_rank",
    "style_score",
    "style_rank",
    "n_sources",
]
ARTICLE_FEATURES = [
    "sales_1w",
    "sales_2w",
    "sales_4w",
    "sales_12w",
    "sales_total",
    "buyers_4w",
    "price_4w",
    "online_share_4w",
    "days_since_first_sale",
    "days_since_last_sale",
    "trend_1w_4w",
    "age_bucket_sales_1w",
]
CATEGORICAL_FEATURES = [
    "product_type_no",
    "graphical_appearance_no",
    "colour_group_code",
    "perceived_colour_master_id",
    "department_no",
    "index_group_no",
    "section_no",
    "garment_group_no",
]
CUSTOMER_FEATURES = [
    "age",
    "purchases_total",
    "purchases_4w",
    "purchases_12w",
    "visits_total",
    "days_since_last_purchase",
    "customer_avg_price",
    "customer_online_share",
    "distinct_articles",
]
PAIR_FEATURES = [
    "pair_purchases",
    "pair_days_since",
    "product_purchases",
    "type_share",
    "index_group_share",
    "garment_share",
    "colour_share",
    "price_ratio",
    "style_similarity",
]
FEATURES = (
    SOURCE_FEATURES
    + ARTICLE_FEATURES
    + CATEGORICAL_FEATURES
    + CUSTOMER_FEATURES
    + PAIR_FEATURES
)
FEATURE_GROUPS = {
    "source": SOURCE_FEATURES,
    "article": ARTICLE_FEATURES + CATEGORICAL_FEATURES,
    "customer": CUSTOMER_FEATURES,
    "pair": PAIR_FEATURES,
}


def _since(as_of: str, days: int) -> str:
    return f"purchase_date >= DATE '{as_of}' - INTERVAL {days} DAY"


def build_features(
    con: duckdb.DuckDBPyConnection, as_of: str, style=None
) -> pd.DataFrame:
    """Join statistics onto the `candidates` table and return a feature frame."""
    before = f"purchase_date < DATE '{as_of}'"
    con.execute(f"""
        CREATE OR REPLACE TABLE article_stats AS
        SELECT article_idx,
               count(*) FILTER (WHERE {_since(as_of, 7)}) AS sales_1w,
               count(*) FILTER (WHERE {_since(as_of, 14)}) AS sales_2w,
               count(*) FILTER (WHERE {_since(as_of, 28)}) AS sales_4w,
               count(*) FILTER (WHERE {_since(as_of, 84)}) AS sales_12w,
               count(*) AS sales_total,
               count(DISTINCT customer_idx) FILTER (WHERE {_since(as_of, 28)})
                   AS buyers_4w,
               avg(price) FILTER (WHERE {_since(as_of, 28)}) AS price_4w,
               avg(online::INTEGER) FILTER (WHERE {_since(as_of, 28)})
                   AS online_share_4w,
               date_diff('day', min(purchase_date), DATE '{as_of}')
                   AS days_since_first_sale,
               date_diff('day', max(purchase_date), DATE '{as_of}')
                   AS days_since_last_sale
        FROM tx SEMI JOIN (SELECT DISTINCT article_idx FROM candidates)
            USING (article_idx)
        WHERE {before}
        GROUP BY article_idx
        """)
    con.execute(f"""
        CREATE OR REPLACE TABLE age_article_1w AS
        SELECT c.age_bucket, t.article_idx, count(*) AS age_bucket_sales_1w
        FROM tx AS t JOIN customer_attrs AS c USING (customer_idx)
        WHERE {before} AND {_since(as_of, 7)}
        GROUP BY c.age_bucket, t.article_idx
        """)
    con.execute(f"""
        CREATE OR REPLACE TABLE target_tx AS
        SELECT t.*, a.product_code, a.product_type_no, a.index_group_no,
               a.garment_group_no, a.colour_group_code
        FROM tx AS t
        SEMI JOIN targets USING (customer_idx)
        JOIN article_attrs AS a USING (article_idx)
        WHERE {before}
        """)
    con.execute(f"""
        CREATE OR REPLACE TABLE customer_stats AS
        SELECT customer_idx,
               count(*) AS purchases_total,
               count(*) FILTER (WHERE {_since(as_of, 28)}) AS purchases_4w,
               count(*) FILTER (WHERE {_since(as_of, 84)}) AS purchases_12w,
               count(DISTINCT purchase_date) AS visits_total,
               date_diff('day', max(purchase_date), DATE '{as_of}')
                   AS days_since_last_purchase,
               avg(price) AS customer_avg_price,
               avg(online::INTEGER) AS customer_online_share,
               count(DISTINCT article_idx) AS distinct_articles
        FROM target_tx
        GROUP BY customer_idx
        """)
    shares = []
    for column, name in (
        ("product_type_no", "type_share"),
        ("index_group_no", "index_group_share"),
        ("garment_group_no", "garment_share"),
        ("colour_group_code", "colour_share"),
    ):
        con.execute(f"""
            CREATE OR REPLACE TABLE share_{name} AS
            SELECT customer_idx, {column},
                   count(*) / sum(count(*)) OVER (PARTITION BY customer_idx)
                       AS {name}
            FROM target_tx GROUP BY customer_idx, {column}
            """)
        shares.append((column, name))
    share_select = ", ".join(f"coalesce(s_{n}.{n}, 0) AS {n}" for _, n in shares)
    share_joins = " ".join(
        f"LEFT JOIN share_{n} AS s_{n} "
        f"ON s_{n}.customer_idx = c.customer_idx AND s_{n}.{col} = a.{col}"
        for col, n in shares
    )
    frame = con.execute(f"""
        WITH pair_history AS (
            SELECT customer_idx, article_idx, count(*) AS pair_purchases,
                   date_diff('day', max(purchase_date), DATE '{as_of}')
                       AS pair_days_since
            FROM target_tx
            SEMI JOIN candidates USING (customer_idx, article_idx)
            GROUP BY customer_idx, article_idx
        ),
        product_history AS (
            SELECT customer_idx, product_code, count(*) AS product_purchases
            FROM target_tx GROUP BY customer_idx, product_code
        )
        SELECT c.customer_idx, c.article_idx,
               c.repurchase_rank, c.popular_rank, c.age_popular_rank,
               c.cooc_score, c.cooc_rank, c.als_score, c.als_rank,
               c.style_score, c.style_rank,
               (c.repurchase_rank IS NOT NULL)::INTEGER
                 + (c.popular_rank IS NOT NULL)::INTEGER
                 + (c.age_popular_rank IS NOT NULL)::INTEGER
                 + (c.cooc_rank IS NOT NULL)::INTEGER
                 + (c.als_rank IS NOT NULL)::INTEGER
                 + (c.style_rank IS NOT NULL)::INTEGER AS n_sources,
               coalesce(s.sales_1w, 0) AS sales_1w,
               coalesce(s.sales_2w, 0) AS sales_2w,
               coalesce(s.sales_4w, 0) AS sales_4w,
               coalesce(s.sales_12w, 0) AS sales_12w,
               coalesce(s.sales_total, 0) AS sales_total,
               coalesce(s.buyers_4w, 0) AS buyers_4w,
               s.price_4w, s.online_share_4w,
               s.days_since_first_sale, s.days_since_last_sale,
               coalesce(s.sales_1w, 0) / (coalesce(s.sales_4w, 0) / 4.0 + 1)
                   AS trend_1w_4w,
               coalesce(aa.age_bucket_sales_1w, 0) AS age_bucket_sales_1w,
               a.product_type_no, a.graphical_appearance_no, a.colour_group_code,
               a.perceived_colour_master_id, a.department_no, a.index_group_no,
               a.section_no, a.garment_group_no,
               ca.age,
               coalesce(cs.purchases_total, 0) AS purchases_total,
               coalesce(cs.purchases_4w, 0) AS purchases_4w,
               coalesce(cs.purchases_12w, 0) AS purchases_12w,
               coalesce(cs.visits_total, 0) AS visits_total,
               cs.days_since_last_purchase, cs.customer_avg_price,
               cs.customer_online_share,
               coalesce(cs.distinct_articles, 0) AS distinct_articles,
               coalesce(ph.pair_purchases, 0) AS pair_purchases,
               ph.pair_days_since,
               coalesce(pr.product_purchases, 0) AS product_purchases,
               {share_select},
               s.price_4w / cs.customer_avg_price AS price_ratio
        FROM candidates AS c
        JOIN article_attrs AS a USING (article_idx)
        JOIN customer_attrs AS ca USING (customer_idx)
        LEFT JOIN article_stats AS s USING (article_idx)
        LEFT JOIN age_article_1w AS aa
          ON aa.age_bucket = ca.age_bucket AND aa.article_idx = c.article_idx
        LEFT JOIN customer_stats AS cs ON cs.customer_idx = c.customer_idx
        LEFT JOIN pair_history AS ph
          ON ph.customer_idx = c.customer_idx AND ph.article_idx = c.article_idx
        LEFT JOIN product_history AS pr
          ON pr.customer_idx = c.customer_idx AND pr.product_code = a.product_code
        {share_joins}
        ORDER BY c.customer_idx, c.article_idx
        """).df()
    frame["style_similarity"] = (
        style.similarity(
            frame["customer_idx"].to_numpy(), frame["article_idx"].to_numpy()
        )
        if style is not None
        else np.nan
    )
    for column in FEATURES:
        if column not in CATEGORICAL_FEATURES:
            frame[column] = frame[column].astype(np.float32)
    for column in CATEGORICAL_FEATURES:
        frame[column] = frame[column].astype(np.int32)
    frame["customer_idx"] = frame["customer_idx"].astype(np.int32)
    frame["article_idx"] = frame["article_idx"].astype(np.int32)
    return frame
