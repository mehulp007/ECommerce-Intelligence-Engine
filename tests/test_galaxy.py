from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ecommerce_intelligence.search.embeddings import ArticleEmbeddings
from ecommerce_intelligence.search.galaxy import (
    GalaxyLayout,
    build_layout,
    pca_reducer,
    to_unit_sphere,
)


def test_layout_covers_every_article_in_embedding_order(world) -> None:
    layout = GalaxyLayout.load(world.embeddings)
    embeddings = ArticleEmbeddings.load(world.embeddings)
    assert layout.meta["count"] == len(embeddings.article_ids) == len(layout.positions)
    assert [f"{i:010d}" for i in layout.article_ids] == list(embeddings.article_ids)
    assert np.isfinite(layout.positions).all()
    assert layout.positions.shape == (layout.meta["count"], 3)


def test_codes_name_each_article_group_and_colour(world) -> None:
    layout = GalaxyLayout.load(world.embeddings)
    articles = pd.read_parquet(world.table("articles")).set_index("article_id")
    for i, article_id in enumerate(f"{a:010d}" for a in layout.article_ids):
        row = articles.loc[article_id]
        assert layout.meta["product_groups"][layout.group[i]] == row.product_group_name
        assert layout.meta["colours"][layout.colour[i]] == row.colour_group_name


def test_rebuilding_gives_the_same_layout(world) -> None:
    first = GalaxyLayout.load(world.embeddings).positions.copy()
    build_layout(world, reducer=pca_reducer, log=lambda _: None)
    assert np.allclose(GalaxyLayout.load(world.embeddings).positions, first)


def test_unit_sphere_centres_and_scales_to_the_99th_percentile() -> None:
    rng = np.random.default_rng(0)
    points = rng.normal(loc=50.0, scale=7.0, size=(5000, 3))
    scaled = to_unit_sphere(points)
    radius = np.linalg.norm(scaled, axis=1)
    assert np.quantile(radius, 0.99) == pytest.approx(1.0, abs=1e-6)
    assert np.allclose(np.median(scaled, axis=0), 0.0, atol=1e-6)
    assert scaled.dtype == np.float32


def test_load_rejects_a_truncated_buffer(tmp_path, world) -> None:
    layout = GalaxyLayout.load(world.embeddings)
    layout.save(tmp_path)
    data = (tmp_path / "galaxy.bin").read_bytes()
    (tmp_path / "galaxy.bin").write_bytes(data[:-3])
    with pytest.raises(ValueError, match="expected"):
        GalaxyLayout.load(tmp_path)
    with pytest.raises(FileNotFoundError):
        GalaxyLayout.load(tmp_path / "missing")
