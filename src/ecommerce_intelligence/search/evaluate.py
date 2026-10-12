"""A small labelled benchmark for natural-language product search."""

from __future__ import annotations

import json

import pandas as pd

from ecommerce_intelligence.paths import ProjectPaths
from ecommerce_intelligence.search.encoder import Encoder
from ecommerce_intelligence.search.index import ProductIndex

K = 10
# Each query names a product kind; a hit is a result of that product type.
QUERIES = {
    "red floral summer dress": {"Dress"},
    "black leather ankle boots": {"Boots"},
    "kids dinosaur t-shirt": {"T-shirt"},
    "blue denim jeans": {"Trousers"},
    "striped swimsuit": {"Swimsuit"},
    "wool scarf": {"Scarf"},
    "baby bodysuit": {"Bodysuit"},
    "white sneakers": {"Sneakers"},
    "leather handbag": {"Bag"},
    "knitted cardigan": {"Cardigan"},
    "lace bra": {"Bra"},
    "winter puffer jacket": {"Jacket"},
}


def evaluate(paths: ProjectPaths, index: ProductIndex, encoder: Encoder) -> dict:
    types = pd.read_parquet(
        paths.table("articles"), columns=["article_id", "product_type_name"]
    ).set_index("article_id")["product_type_name"]
    per_query = {}
    for query, wanted in QUERIES.items():
        hits = index.search_text(encoder, query, K)
        per_query[query] = sum(types.get(h.article_id) in wanted for h in hits) / K
    report = {
        "precision_at_10": round(sum(per_query.values()) / len(per_query), 4),
        "per_query": per_query,
        "queries": len(QUERIES),
    }
    paths.reports.mkdir(parents=True, exist_ok=True)
    (paths.reports / "search_quality.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report
