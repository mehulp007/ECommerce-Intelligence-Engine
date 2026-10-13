"""A miniature H&M archive with the real file layout and column names."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pandas as pd
import pytest
from PIL import Image
from test_customers import _write
from test_search import FakeEncoder

from ecommerce_intelligence.customers.pipeline import run as run_customers
from ecommerce_intelligence.data.images import thumbnail_path
from ecommerce_intelligence.data.warehouse import build_warehouse
from ecommerce_intelligence.paths import ProjectPaths
from ecommerce_intelligence.recsys.pipeline import run as run_recsys
from ecommerce_intelligence.serving.prepare import build

ARTICLES_CSV = (
    "article_id,product_code,prod_name,product_type_no,product_type_name,"
    "product_group_name,graphical_appearance_no,graphical_appearance_name,"
    "colour_group_code,colour_group_name,perceived_colour_value_id,"
    "perceived_colour_value_name,perceived_colour_master_id,"
    "perceived_colour_master_name,department_no,department_name,index_code,"
    "index_name,index_group_no,index_group_name,section_no,section_name,"
    "garment_group_no,garment_group_name,detail_desc\n"
    "0108775015,0108775,Strap top,253,Vest top,Garment Upper body,1010016,Solid,"
    "9,Black,4,Dark,5,Black,1676,Jersey Basic,A,Ladieswear,1,Ladieswear,16,"
    'Womens Everyday Basics,1002,Jersey Basic,"Jersey top with narrow, '
    'shoulder straps."\n'
    "0110065001,0110065,OP T-shirt (Idro),306,Bra,Underwear,1010016,Solid,9,"
    "Black,4,Dark,5,Black,1339,Clean Lingerie,B,Lingeries/Tights,1,Ladieswear,"
    '61,Womens Lingerie,1017,"Under-, Nightwear",Microfibre T-shirt bra.\n'
    "0111565001,0111565,20 den 1p Stockings,304,Underwear Tights,Socks & Tights,"
    "1010016,Solid,9,Black,4,Dark,5,Black,3608,Tights basic,B,Lingeries/Tights,"
    "1,Ladieswear,62,Womens Nightwear Socks & Tigh,1021,Socks and Tights,\n"
)
CUSTOMERS_CSV = (
    "customer_id,FN,Active,club_member_status,fashion_news_frequency,age,postal_code\n"
    "c1,1.0,1.0,ACTIVE,Regularly,49.0,p1\n"
    "c2,,,ACTIVE,NONE,25.0,p2\n"
    "c3,,,PRE-CREATE,None,,p3\n"
    "c4,1.0,1.0,,Monthly,61.0,p1\n"
)
TRANSACTIONS_CSV = (
    "t_dat,customer_id,article_id,price,sales_channel_id\n"
    "2018-09-20,c1,0108775015,0.050830508474576264,2\n"
    "2018-09-20,c1,0108775015,0.050830508474576264,2\n"
    "2019-05-01,c2,0110065001,0.030491525423728815,1\n"
    "2020-09-22,c1,0111565001,0.016932203389830508,2\n"
)


def _jpeg(size: tuple[int, int]) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (200, 30, 60)).save(buffer, format="JPEG")
    return buffer.getvalue()


def write_archive(paths: ProjectPaths, *, transactions: str = TRANSACTIONS_CSV) -> Path:
    paths.raw.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(paths.archive, "w") as archive:
        archive.writestr("articles.csv", ARTICLES_CSV)
        archive.writestr("customers.csv", CUSTOMERS_CSV)
        archive.writestr("transactions_train.csv", transactions)
        archive.writestr("sample_submission.csv", "customer_id,prediction\n")
        archive.writestr("images/010/0108775015.jpg", _jpeg((1166, 1750)))
        archive.writestr("images/011/0110065001.jpg", _jpeg((600, 400)))
        archive.writestr("images/011/not-an-article.jpg", _jpeg((10, 10)))
        archive.writestr("images/099/0110065002.jpg", _jpeg((10, 10)))
        archive.writestr("images/011/0111565001.jpg", b"not a jpeg")
    return paths.archive


@pytest.fixture
def project(tmp_path: Path) -> ProjectPaths:
    paths = ProjectPaths(tmp_path)
    write_archive(paths)
    return paths


class Encoder16(FakeEncoder):
    dim = 16  # Matches the synthetic embeddings written by _write.


@pytest.fixture(scope="session")
def world(tmp_path_factory) -> ProjectPaths:
    """Synthetic shoppers run through the real pipelines and serving build."""
    paths = _write(tmp_path_factory.mktemp("world"))
    build_warehouse(paths)
    run_recsys(paths, ablations=False, log=lambda _: None)
    run_customers(paths, trials=1, log=lambda _: None)
    for article_id in pd.read_parquet(paths.table("articles"))["article_id"][:10]:
        target = thumbnail_path(paths.thumbnails, article_id)
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (8, 12), "white").save(target, format="WEBP")
    build(paths, log=lambda _: None)
    return paths
