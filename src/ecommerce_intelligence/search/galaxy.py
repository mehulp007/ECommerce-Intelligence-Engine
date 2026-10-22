"""A 3D map of the catalogue: products that look alike sit close together.

Each article's joint photo-and-description vector (the one text search uses)
is reduced to three dimensions with PCA and then UMAP, so neighbourhoods in
the map match neighbourhoods in search.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ecommerce_intelligence.paths import ProjectPaths
from ecommerce_intelligence.search.embeddings import ArticleEmbeddings
from ecommerce_intelligence.search.index import joint_vectors

POINTS_FILE = "galaxy.bin"
META_FILE = "galaxy.json"
PCA_DIMS = 50
N_NEIGHBORS = 30
MIN_DIST = 0.15
SEED = 42

Reducer = Callable[[np.ndarray], np.ndarray]


def umap_reducer(vectors: np.ndarray) -> np.ndarray:
    """PCA to 50 dimensions, then UMAP to 3 with cosine distance."""
    import umap
    from sklearn.decomposition import PCA

    reduced = PCA(n_components=PCA_DIMS, random_state=SEED).fit_transform(vectors)
    return umap.UMAP(
        n_components=3,
        n_neighbors=N_NEIGHBORS,
        min_dist=MIN_DIST,
        metric="cosine",
        random_state=SEED,
        n_jobs=1,  # A fixed seed makes UMAP single-threaded anyway.
    ).fit_transform(reduced)


def pca_reducer(vectors: np.ndarray) -> np.ndarray:
    """A fast linear layout, used in tests."""
    from sklearn.decomposition import PCA

    return PCA(n_components=3, random_state=SEED).fit_transform(vectors)


def to_unit_sphere(points: np.ndarray) -> np.ndarray:
    """Centre the cloud and scale it so 99% of points lie within radius 1."""
    centred = points - np.median(points, axis=0)
    radius = np.quantile(np.linalg.norm(centred, axis=1), 0.99)
    return (centred / max(float(radius), 1e-9)).astype(np.float32)


def _codes(values: pd.Series) -> tuple[np.ndarray, list[str]]:
    """Small integer codes for a text column; names are sorted by frequency."""
    names = values.fillna("Unknown").value_counts().index.tolist()
    lookup = {name: i for i, name in enumerate(names)}
    return values.fillna("Unknown").map(lookup).to_numpy(np.uint8), names


@dataclass
class GalaxyLayout:
    """Points in the same order as the article embeddings."""

    positions: np.ndarray
    article_ids: np.ndarray
    group: np.ndarray
    colour: np.ndarray
    meta: dict

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        with open(directory / POINTS_FILE, "wb") as out:
            out.write(self.positions.astype("<f4").tobytes())
            out.write(self.article_ids.astype("<u4").tobytes())
            out.write(self.group.astype(np.uint8).tobytes())
            out.write(self.colour.astype(np.uint8).tobytes())
        (directory / META_FILE).write_text(
            json.dumps(self.meta, indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> GalaxyLayout:
        meta_path = directory / META_FILE
        if not meta_path.is_file():
            raise FileNotFoundError("Run `eie search galaxy` to build the map")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        n = meta["count"]
        raw = (directory / POINTS_FILE).read_bytes()
        if len(raw) != n * 18:
            raise ValueError(f"{POINTS_FILE} has {len(raw)} bytes, expected {n * 18}")
        positions = np.frombuffer(raw, "<f4", n * 3, 0).reshape(n, 3)
        ids = np.frombuffer(raw, "<u4", n, n * 12)
        group = np.frombuffer(raw, np.uint8, n, n * 16)
        colour = np.frombuffer(raw, np.uint8, n, n * 17)
        return cls(positions, ids, group, colour, meta)


def build_layout(
    paths: ProjectPaths, reducer: Reducer | None = None, *, log=print
) -> GalaxyLayout:
    """Compute and save the 3D map; `reducer` defaults to PCA then UMAP."""
    started = time.perf_counter()
    embeddings = ArticleEmbeddings.load(paths.embeddings)
    joint, _, _ = joint_vectors(embeddings)
    log(f"Laying out {len(joint):,} articles in 3D")
    positions = to_unit_sphere((reducer or umap_reducer)(joint))

    articles = pd.read_parquet(
        paths.table("articles"),
        columns=["article_id", "product_group_name", "colour_group_name"],
    ).set_index("article_id")
    articles = articles.reindex(embeddings.article_ids)
    group, groups = _codes(articles["product_group_name"])
    colour, colours = _codes(articles["colour_group_name"])
    method = "pca+umap" if reducer is None else getattr(reducer, "__name__", "custom")
    meta = {
        "count": len(positions),
        "product_groups": groups,
        "colours": colours,
        "method": method,
        "pca_dims": PCA_DIMS,
        "n_neighbors": N_NEIGHBORS,
        "min_dist": MIN_DIST,
        "seed": SEED,
        "seconds": round(time.perf_counter() - started, 1),
        "built_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    layout = GalaxyLayout(
        positions=positions,
        article_ids=embeddings.article_ids.astype(np.int64).astype(np.uint32),
        group=group,
        colour=colour,
        meta=meta,
    )
    layout.save(paths.embeddings)
    return layout
