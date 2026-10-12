"""Build the tables the API reads, so every request is a cheap lookup.

Outputs in data/serving/:
- customer_history.parquet  purchases sorted by customer for point lookups
- product_stats.parquet     per-article sales statistics and a 12-week trend
- overview.json             dashboard aggregates
- manifest.json             what was built, from which inputs, and when
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime

from ecommerce_intelligence.data.duck import connect, sql_path
from ecommerce_intelligence.paths import ProjectPaths

DATA_END = "2020-09-22"
TREND_WEEKS = 12
REQUIRED_INPUTS = ("recommendations.parquet", "customer_scores.parquet")


def serving_dir(paths: ProjectPaths):
    return paths.root / "data" / "serving"


def _view(con, name: str, path) -> None:
    con.execute(
        f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet({sql_path(path)})"
    )


def build(paths: ProjectPaths, *, log=print) -> dict:
    out = serving_dir(paths)
    missing = [n for n in REQUIRED_INPUTS if not (out / n).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing {missing}; run `eie recsys train` and `eie customers train` first"
        )
    con = connect(paths)
    for name in ("transactions", "articles", "customers"):
        _view(con, name, paths.table(name))
    timings = {}

    started = time.perf_counter()
    con.execute(f"""
        COPY (
            SELECT customer_id, purchase_date, article_id, price, sales_channel_id
            FROM transactions
            ORDER BY customer_id, purchase_date DESC, article_id
        ) TO {sql_path(out / "customer_history.parquet")}
        (FORMAT parquet, COMPRESSION zstd, ROW_GROUP_SIZE 122880)
        """)
    timings["customer_history"] = round(time.perf_counter() - started, 1)
    log(f"customer history: {timings['customer_history']}s")

    started = time.perf_counter()
    weeks = ", ".join(
        f"count(*) FILTER (WHERE purchase_date >= DATE '{DATA_END}' - INTERVAL "
        f"{7 * (i + 1) - 1} DAY AND purchase_date <= DATE '{DATA_END}' - INTERVAL "
        f"{7 * i} DAY)"
        for i in reversed(range(TREND_WEEKS))
    )
    con.execute(f"""
        COPY (
            SELECT a.article_id,
                   count(t.article_id) AS sales_total,
                   count(t.article_id) FILTER (
                       WHERE t.purchase_date > DATE '{DATA_END}' - INTERVAL 7 DAY
                   ) AS sales_1w,
                   count(t.article_id) FILTER (
                       WHERE t.purchase_date > DATE '{DATA_END}' - INTERVAL 28 DAY
                   ) AS sales_4w,
                   -- 64-bit hashes keep the distinct count within memory;
                   -- collisions are negligible for 1.4M customers.
                   count(DISTINCT hash(t.customer_id)) AS buyers_total,
                   median(t.price) AS median_price,
                   min(t.purchase_date) AS first_sale,
                   max(t.purchase_date) AS last_sale,
                   avg((t.sales_channel_id = 2)::INTEGER) AS online_share,
                   [{weeks}] AS weekly_sales_12w,
                   dense_rank() OVER (
                       ORDER BY count(t.article_id) FILTER (
                           WHERE t.purchase_date > DATE '{DATA_END}' - INTERVAL 28 DAY
                       ) DESC
                   ) AS popularity_rank_4w
            FROM articles AS a
            LEFT JOIN transactions AS t USING (article_id)
            GROUP BY a.article_id
            ORDER BY a.article_id
        ) TO {sql_path(out / "product_stats.parquet")}
        (FORMAT parquet, COMPRESSION zstd)
        """)
    timings["product_stats"] = round(time.perf_counter() - started, 1)
    log(f"product stats: {timings['product_stats']}s")

    started = time.perf_counter()
    overview = build_overview(con)
    (out / "overview.json").write_text(
        json.dumps(overview, indent=2, default=str) + "\n", encoding="utf-8"
    )
    timings["overview"] = round(time.perf_counter() - started, 1)
    log(f"overview: {timings['overview']}s")
    con.close()

    files = {
        path.name: {
            "bytes": path.stat().st_size,
            "modified": datetime.fromtimestamp(path.stat().st_mtime, UTC).isoformat(),
        }
        for path in sorted(out.glob("*.parquet")) + [out / "overview.json"]
    }
    manifest = {
        "built_at": datetime.now(UTC).isoformat(),
        "data_end": DATA_END,
        "files": files,
        "timings_seconds": timings,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def build_overview(con) -> dict:
    def rows(query: str) -> list[dict]:
        cursor = con.execute(query)
        names = [d[0] for d in cursor.description]
        return [dict(zip(names, row, strict=True)) for row in cursor.fetchall()]

    kpis = rows("""
        SELECT count(*) AS purchases, sum(price) AS revenue_units,
               count(DISTINCT customer_id) AS buyers,
               count(DISTINCT article_id) AS articles_sold,
               count(DISTINCT (customer_id, purchase_date)) AS orders,
               avg((sales_channel_id = 2)::INTEGER) AS online_share,
               min(purchase_date) AS first_date, max(purchase_date) AS last_date
        FROM transactions
        """)[0]
    weekly = rows("""
        SELECT CAST(date_trunc('week', purchase_date) AS DATE) AS week,
               count(*) AS purchases, sum(price) AS revenue_units,
               count(DISTINCT customer_id) AS buyers,
               avg((sales_channel_id = 2)::INTEGER) AS online_share
        FROM transactions GROUP BY ALL ORDER BY week
        """)
    groups = rows("""
        SELECT a.index_group_name AS index_group,
               count(*) AS purchases, sum(t.price) AS revenue_units
        FROM transactions AS t JOIN articles AS a USING (article_id)
        GROUP BY ALL ORDER BY revenue_units DESC
        """)
    product_groups = rows(f"""
        SELECT a.product_group_name AS product_group,
               count(*) FILTER (
                   WHERE t.purchase_date > DATE '{DATA_END}' - INTERVAL 28 DAY
               ) AS purchases_4w,
               count(*) FILTER (
                   WHERE t.purchase_date > DATE '{DATA_END}' - INTERVAL 393 DAY
                     AND t.purchase_date <= DATE '{DATA_END}' - INTERVAL 365 DAY
               ) AS purchases_same_4w_last_year
        FROM transactions AS t JOIN articles AS a USING (article_id)
        GROUP BY ALL ORDER BY purchases_4w DESC LIMIT 12
        """)
    ages = rows("""
        SELECT (floor(c.age / 10) * 10)::INTEGER AS age_band,
               count(DISTINCT t.customer_id) AS buyers, sum(t.price) AS revenue_units
        FROM transactions AS t JOIN customers AS c USING (customer_id)
        WHERE c.age IS NOT NULL GROUP BY ALL ORDER BY age_band
        """)
    # Monthly cohorts: share of each first-purchase month still buying later.
    cohorts = rows("""
        WITH firsts AS (
            SELECT customer_id, date_trunc('month', min(purchase_date)) AS cohort
            FROM transactions GROUP BY customer_id
        ),
        activity AS (
            SELECT DISTINCT customer_id, date_trunc('month', purchase_date) AS month
            FROM transactions
        )
        SELECT CAST(f.cohort AS DATE) AS cohort,
               datediff('month', f.cohort, a.month) AS months_since,
               count(*) AS customers
        FROM firsts AS f JOIN activity AS a USING (customer_id)
        WHERE f.cohort >= DATE '2018-10-01'
        GROUP BY ALL ORDER BY cohort, months_since
        """)
    sizes = {r["cohort"]: r["customers"] for r in cohorts if r["months_since"] == 0}
    retention = [
        {**r, "retention": r["customers"] / sizes[r["cohort"]]}
        for r in cohorts
        if r["months_since"] <= 12
    ]
    return {
        "kpis": kpis,
        "weekly": weekly,
        "index_groups": groups,
        "product_groups": product_groups,
        "age_bands": ages,
        "cohort_retention": retention,
    }
