from __future__ import annotations

import hashlib

import numpy as np
import pytest
from PIL import Image

from ecommerce_intelligence.data.images import build_thumbnails
from ecommerce_intelligence.data.ingest import ingest
from ecommerce_intelligence.data.source import extract_csvs
from ecommerce_intelligence.search.embeddings import (
    ArticleEmbeddings,
    article_text,
    build_embeddings,
)
from ecommerce_intelligence.search.encoder import normalize
from ecommerce_intelligence.search.index import ProductIndex


class FakeEncoder:
    """Deterministic stand-in for FashionCLIP: colours and words become vectors."""

    name = "fake"
    dim = 8

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        means = [
            np.asarray(image, dtype=np.float32).mean(axis=(0, 1)) for image in images
        ]
        return normalize(np.hstack([np.array(means), np.ones((len(images), 5))]))

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        rows = []
        for text in texts:
            seed = int(hashlib.sha256(text.encode()).hexdigest()[:8], 16)
            rows.append(np.random.default_rng(seed).normal(size=self.dim))
        return normalize(np.array(rows))


def test_build_embeddings_aligns_rows_and_marks_missing_photos(project) -> None:
    extract_csvs(project)
    ingest(project)
    build_thumbnails(project, workers=1)

    manifest = build_embeddings(project, FakeEncoder(), log=lambda _: None)
    embeddings = ArticleEmbeddings.load(project.embeddings)

    assert manifest["articles"] == 3 and manifest["with_image"] == 2
    assert list(embeddings.article_ids) == ["0108775015", "0110065001", "0111565001"]
    assert embeddings.has_image.tolist() == [True, True, False]
    assert np.allclose(np.linalg.norm(embeddings.text, axis=1), 1, atol=1e-3)
    # The article without a photo falls back to its text embedding.
    assert np.allclose(embeddings.item[2], embeddings.text[2])


def test_article_text_skips_missing_fields() -> None:
    class Row:
        prod_name = "Strap top"
        colour_group_name = "Black"
        product_type_name = "Vest top"
        index_group_name = "Ladieswear"
        detail_desc = None

    assert article_text(Row) == "Strap top. Black. Vest top. Ladieswear"


def _index() -> ProductIndex:
    rng = np.random.default_rng(0)
    base = normalize(rng.normal(size=(6, 8)))
    image = base.copy()
    image[1] = normalize(base[0] + 0.05 * rng.normal(size=8))  # Near-duplicate of 0.
    return ProductIndex(
        ArticleEmbeddings(
            article_ids=np.array([f"{i:010d}" for i in range(6)]),
            image=image,
            text=normalize(rng.normal(size=(6, 8))),
            has_image=np.array([True] * 6),
        )
    )


def test_similar_returns_nearest_and_never_itself() -> None:
    hits = _index().similar("0000000000", k=3)
    assert hits[0].article_id == "0000000001"
    assert "0000000000" not in {hit.article_id for hit in hits}
    assert len(hits) == 3
    assert hits[0].score >= hits[1].score >= hits[2].score
    with pytest.raises(KeyError):
        _index().similar("missing")


def test_text_search_blends_photo_and_description() -> None:
    index = _index()
    target = index.embeddings
    query = normalize(0.6 * target.image[3] + 0.4 * target.text[3])
    assert index.search_vector(query, k=1)[0].article_id == "0000000003"
    with pytest.raises(ValueError):
        index.search_text(FakeEncoder(), "   ")
