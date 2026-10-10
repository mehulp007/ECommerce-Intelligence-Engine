from __future__ import annotations

import duckdb
import pandas as pd
import pyarrow.parquet as pq
from conftest import write_archive
from PIL import Image

from ecommerce_intelligence.data.images import build_thumbnails, thumbnail_path
from ecommerce_intelligence.data.ingest import ingest
from ecommerce_intelligence.data.quality import write_report
from ecommerce_intelligence.data.source import extract_csvs
from ecommerce_intelligence.data.warehouse import build_warehouse


def _prepare(project) -> None:
    extract_csvs(project)
    ingest(project)


def test_ingest_preserves_ids_and_types(project) -> None:
    sizes = extract_csvs(project)
    assert set(sizes) == {"articles.csv", "customers.csv", "transactions_train.csv"}
    assert ingest(project) == {"transactions": 4, "customers": 4, "articles": 3}

    transactions = pd.read_parquet(project.table("transactions"))
    assert transactions["article_id"].str.len().eq(10).all()
    assert transactions["article_id"].iloc[0] == "0108775015"
    assert transactions["purchase_date"].is_monotonic_increasing
    schema = pq.read_schema(project.table("transactions"))
    assert str(schema.field("purchase_date").type) == "date32[day]"
    assert str(schema.field("sales_channel_id").type) == "int8"

    articles = pd.read_parquet(project.table("articles"))
    assert articles.loc[0, "product_code"] == "0108775"
    assert articles.loc[0, "detail_desc"] == "Jersey top with narrow, shoulder straps."
    assert articles.loc[1, "garment_group_name"] == "Under-, Nightwear"

    customers = pd.read_parquet(project.table("customers")).set_index("customer_id")
    assert customers.loc["c1", "fashion_news_subscriber"]
    assert not customers.loc["c2", "active"]
    assert customers.loc["c2", "fashion_news_frequency"] == "none"
    assert customers.loc["c3", "fashion_news_frequency"] == "none"
    assert pd.isna(customers.loc["c3", "age"])
    assert pd.isna(customers.loc["c4", "club_member_status"])


def test_extract_is_idempotent(project) -> None:
    first = extract_csvs(project)
    stamp = (project.csv / "customers.csv").stat().st_mtime_ns
    assert extract_csvs(project) == first
    assert (project.csv / "customers.csv").stat().st_mtime_ns == stamp


def test_quality_report_passes_on_clean_data(project) -> None:
    _prepare(project)
    target, report, failures = write_report(project)

    assert target.is_file()
    assert failures == []
    t = report["transactions"]
    assert t["rows"] == 4
    assert t["repeated_line_rows"] == 1  # Repeated lines represent extra units.
    assert t["first_date"] == "2018-09-20" and t["last_date"] == "2020-09-22"
    assert t["online_share"] == 0.75
    assert report["customers"]["with_transactions"] == 2
    assert report["articles"]["never_sold"] == 0


def test_quality_flags_orphans_and_bad_prices(tmp_path) -> None:
    from ecommerce_intelligence.paths import ProjectPaths

    paths = ProjectPaths(tmp_path)
    write_archive(
        paths,
        transactions=(
            "t_dat,customer_id,article_id,price,sales_channel_id\n"
            "2019-01-01,ghost,0108775015,0.05,2\n"
            "2019-01-02,c1,0999999999,0.0,1\n"
        ),
    )
    _prepare(paths)
    _, _, failures = write_report(paths)
    assert any("lack a customer" in failure for failure in failures)
    assert any("lack an article" in failure for failure in failures)
    assert any("price <= 0" in failure for failure in failures)


def test_warehouse_tables_and_sales_view(project) -> None:
    _prepare(project)
    assert build_warehouse(project) == {
        "transactions": 4,
        "customers": 4,
        "articles": 3,
    }
    with duckdb.connect(str(project.warehouse), read_only=True) as connection:
        rows = connection.execute("""
            SELECT sales_channel, count(*), any_value(customer_age)
            FROM sales WHERE customer_id = 'c1' GROUP BY ALL
            """).fetchall()
    assert rows == [("online", 3, 49)]


def test_thumbnails_resize_skip_and_report_failures(project) -> None:
    summary = build_thumbnails(project, workers=1)

    assert summary["archive_images"] == 3  # Malformed member names are ignored.
    assert summary["created"] == 2
    assert summary["failed_article_ids"] == ["0111565001"]
    with Image.open(thumbnail_path(project.thumbnails, "0108775015")) as image:
        assert image.format == "WEBP"
        assert image.width <= 256 and image.height <= 384
        assert image.size == (256, 384)

    again = build_thumbnails(project, workers=1)
    assert again["created"] == 0 and again["skipped_existing"] == 2
