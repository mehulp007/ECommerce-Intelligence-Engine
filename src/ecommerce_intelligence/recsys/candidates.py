"""Stage one: propose a few dozen plausible articles per customer.

Every function reads the `targets(customer_idx)` table and writes one
`cand_<source>` table keyed by (customer_idx, article_idx) with a rank column.
Only purchases strictly before `as_of` are used.
"""

from __future__ import annotations

import duckdb
import numpy as np
import pandas as pd

from ecommerce_intelligence.recsys import config
from ecommerce_intelligence.recsys.als import AlsModel
from ecommerce_intelligence.recsys.style import StyleModel


def _window(as_of: str, days: int) -> str:
    return (
        f"purchase_date >= DATE '{as_of}' - INTERVAL {days} DAY "
        f"AND purchase_date < DATE '{as_of}'"
    )


def repurchase(con: duckdb.DuckDBPyConnection, as_of: str) -> None:
    con.execute(f"""
        CREATE OR REPLACE TABLE cand_repurchase AS
        SELECT customer_idx, article_idx,
               row_number() OVER (
                   PARTITION BY customer_idx
                   ORDER BY last_date DESC, purchases DESC, article_idx
               )::SMALLINT AS repurchase_rank
        FROM (
            SELECT customer_idx, article_idx, max(purchase_date) AS last_date,
                   count(*) AS purchases
            FROM tx SEMI JOIN targets USING (customer_idx)
            WHERE {_window(as_of, config.REPURCHASE_DAYS)}
            GROUP BY ALL
        )
        QUALIFY repurchase_rank <= {config.REPURCHASE_CANDIDATES}
        """)


def popular(con: duckdb.DuckDBPyConnection, as_of: str) -> None:
    con.execute(f"""
        CREATE OR REPLACE TABLE popular_items AS
        SELECT article_idx,
               row_number() OVER (ORDER BY count(*) DESC, article_idx)::SMALLINT
                   AS popular_rank
        FROM tx WHERE {_window(as_of, 7)}
        GROUP BY article_idx
        QUALIFY popular_rank <= {config.POPULAR_CANDIDATES}
        """)
    con.execute("""
        CREATE OR REPLACE TABLE cand_popular AS
        SELECT t.customer_idx, p.article_idx, p.popular_rank
        FROM targets AS t CROSS JOIN popular_items AS p
        """)


def age_popular(con: duckdb.DuckDBPyConnection, as_of: str) -> None:
    con.execute(f"""
        CREATE OR REPLACE TABLE age_popular_items AS
        SELECT c.age_bucket, t.article_idx,
               row_number() OVER (
                   PARTITION BY c.age_bucket ORDER BY count(*) DESC, t.article_idx
               )::SMALLINT AS age_popular_rank
        FROM tx AS t JOIN customer_attrs AS c USING (customer_idx)
        WHERE {_window(as_of, 7)}
        GROUP BY c.age_bucket, t.article_idx
        QUALIFY age_popular_rank <= {config.AGE_POPULAR_CANDIDATES}
        """)
    con.execute("""
        CREATE OR REPLACE TABLE cand_age_popular AS
        SELECT t.customer_idx, p.article_idx, p.age_popular_rank
        FROM targets AS t
        JOIN customer_attrs AS c USING (customer_idx)
        JOIN age_popular_items AS p USING (age_bucket)
        """)


def cooccurrence(con: duckdb.DuckDBPyConnection, as_of: str) -> None:
    """Articles bought in the same basket as the customer's recent purchases."""
    con.execute(f"""
        CREATE OR REPLACE TABLE baskets AS
        WITH lines AS (
            SELECT DISTINCT customer_idx, purchase_date, article_idx
            FROM tx WHERE {_window(as_of, config.COOC_DAYS)}
        ),
        sized AS (
            SELECT customer_idx, purchase_date FROM lines
            GROUP BY ALL
            HAVING count(*) BETWEEN 2 AND {config.COOC_BASKET_LIMIT}
        )
        SELECT lines.* FROM lines SEMI JOIN sized USING (customer_idx, purchase_date)
        """)
    con.execute(f"""
        CREATE OR REPLACE TABLE item_pairs AS
        SELECT a.article_idx AS seed_idx, b.article_idx,
               count(*)::INTEGER AS pair_count
        FROM baskets AS a JOIN baskets AS b
          ON a.customer_idx = b.customer_idx
         AND a.purchase_date = b.purchase_date
         AND a.article_idx <> b.article_idx
        GROUP BY a.article_idx, b.article_idx
        QUALIFY row_number() OVER (
            PARTITION BY seed_idx ORDER BY pair_count DESC, b.article_idx
        ) <= {config.COOC_PARTNERS}
        """)
    con.execute(f"""
        CREATE OR REPLACE TABLE cand_cooc AS
        WITH seeds AS (
            SELECT customer_idx, article_idx AS seed_idx
            FROM (
                SELECT customer_idx, article_idx, max(purchase_date) AS last_date
                FROM tx SEMI JOIN targets USING (customer_idx)
                WHERE {_window(as_of, config.REPURCHASE_DAYS)}
                GROUP BY ALL
            )
            QUALIFY row_number() OVER (
                PARTITION BY customer_idx ORDER BY last_date DESC, article_idx
            ) <= {config.COOC_SEEDS}
        )
        SELECT customer_idx, article_idx, sum(pair_count)::FLOAT AS cooc_score,
               row_number() OVER (
                   PARTITION BY customer_idx ORDER BY sum(pair_count) DESC, article_idx
               )::SMALLINT AS cooc_rank
        FROM seeds JOIN item_pairs USING (seed_idx)
        GROUP BY customer_idx, article_idx
        QUALIFY cooc_rank <= {config.COOC_CANDIDATES}
        """)


def collaborative(con: duckdb.DuckDBPyConnection, as_of: str, model: AlsModel) -> None:
    """Top ALS articles for targets that have interactions in the ALS window."""
    users = con.execute("SELECT customer_idx FROM targets").fetchnumpy()["customer_idx"]
    users = users[model.has_history(users)]
    frames = []
    for start in range(0, len(users), 50_000):
        chunk = users[start : start + 50_000]
        items, scores = model.recommend(chunk, config.ALS_CANDIDATES)
        frames.append(
            pd.DataFrame(
                {
                    "customer_idx": np.repeat(chunk, items.shape[1]).astype("int32"),
                    "article_idx": items.ravel().astype("int32"),
                    "als_score": scores.ravel().astype("float32"),
                    "als_rank": np.tile(
                        np.arange(1, items.shape[1] + 1, dtype="int16"), len(chunk)
                    ),
                }
            )
        )
    result = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(
            {
                "customer_idx": pd.Series(dtype="int32"),
                "article_idx": pd.Series(dtype="int32"),
                "als_score": pd.Series(dtype="float32"),
                "als_rank": pd.Series(dtype="int16"),
            }
        )
    )
    result = result[result["article_idx"] >= 0]
    con.register("als_frame", result)
    con.execute("CREATE OR REPLACE TABLE cand_als AS SELECT * FROM als_frame")
    con.unregister("als_frame")


def build_candidates(
    con: duckdb.DuckDBPyConnection,
    as_of: str,
    als_model: AlsModel,
    style_model: StyleModel,
    sources: tuple[str, ...] = config.SOURCES,
) -> None:
    """Union every source into `candidates` with one row per pair."""
    builders = {
        "repurchase": lambda: repurchase(con, as_of),
        "popular": lambda: popular(con, as_of),
        "age_popular": lambda: age_popular(con, as_of),
        "cooc": lambda: cooccurrence(con, as_of),
        "als": lambda: collaborative(con, as_of, als_model),
        "style": lambda: style_model.candidates(con, as_of),
    }
    for source in sources:
        builders[source]()
    pairs = " UNION ".join(
        f"SELECT customer_idx, article_idx FROM cand_{source}" for source in sources
    )
    # A source left out of an ablation contributes neither rows nor columns.
    columns = {
        "repurchase": ["repurchase_rank"],
        "popular": ["popular_rank"],
        "age_popular": ["age_popular_rank"],
        "cooc": ["cooc_score", "cooc_rank"],
        "als": ["als_score", "als_rank"],
        "style": ["style_score", "style_rank"],
    }
    select = []
    joins = []
    for source in config.SOURCES:
        for column in columns[source]:
            select.append(
                f"{source}_t.{column}" if source in sources else f"NULL AS {column}"
            )
        if source in sources:
            joins.append(
                f"LEFT JOIN cand_{source} AS {source}_t "
                "USING (customer_idx, article_idx)"
            )
    con.execute(f"""
        CREATE OR REPLACE TABLE candidates AS
        SELECT p.customer_idx, p.article_idx, {", ".join(select)}
        FROM ({pairs}) AS p
        {" ".join(joins)}
        """)
