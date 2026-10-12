"""Compute, save, and load per-article image and text embeddings."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from ecommerce_intelligence.data.images import thumbnail_path
from ecommerce_intelligence.paths import ProjectPaths
from ecommerce_intelligence.search.encoder import Encoder, normalize

IMAGE_BATCH = 256
TEXT_BATCH = 512


def article_text(row) -> str:
    """A short product description; CLIP reads at most 77 tokens."""
    parts = [
        row.prod_name,
        row.colour_group_name,
        row.product_type_name,
        row.index_group_name,
        row.detail_desc,
    ]
    return ". ".join(str(part) for part in parts if isinstance(part, str) and part)


@dataclass
class ArticleEmbeddings:
    """Rows follow `article_ids`, which is sorted like the articles table."""

    article_ids: np.ndarray
    image: np.ndarray
    text: np.ndarray
    has_image: np.ndarray

    @property
    def item(self) -> np.ndarray:
        """Image embedding when a photo exists, otherwise the text embedding."""
        return np.where(self.has_image[:, None], self.image, self.text)

    def save(self, directory: Path, manifest: dict) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.save(directory / "image.npy", self.image.astype(np.float16))
        np.save(directory / "text.npy", self.text.astype(np.float16))
        pd.DataFrame(
            {"article_id": self.article_ids, "has_image": self.has_image}
        ).to_parquet(directory / "articles.parquet", index=False)
        (directory / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> ArticleEmbeddings:
        if not (directory / "manifest.json").is_file():
            raise FileNotFoundError("Run `eie search embed` to build embeddings")
        index = pd.read_parquet(directory / "articles.parquet")
        return cls(
            article_ids=index["article_id"].to_numpy(),
            image=np.load(directory / "image.npy").astype(np.float32),
            text=np.load(directory / "text.npy").astype(np.float32),
            has_image=index["has_image"].to_numpy(dtype=bool),
        )


def _load_image(path: Path) -> Image.Image | None:
    if not path.is_file():
        return None
    with Image.open(path) as image:
        return image.convert("RGB")


def build_embeddings(
    paths: ProjectPaths,
    encoder: Encoder,
    *,
    log: Callable[[str], None] = print,
) -> dict:
    articles = pd.read_parquet(
        paths.table("articles"),
        columns=[
            "article_id",
            "prod_name",
            "colour_group_name",
            "product_type_name",
            "index_group_name",
            "detail_desc",
        ],
    ).sort_values("article_id", ignore_index=True)
    ids = articles["article_id"].to_numpy()
    count = len(ids)
    started = time.perf_counter()

    texts = [article_text(row) for row in articles.itertuples(index=False)]
    text = np.zeros((count, encoder.dim), dtype=np.float32)
    for start in range(0, count, TEXT_BATCH):
        text[start : start + TEXT_BATCH] = encoder.encode_texts(
            texts[start : start + TEXT_BATCH]
        )
    text_seconds = time.perf_counter() - started
    log(f"text embeddings: {count:,} articles in {text_seconds:.1f}s")

    image = np.zeros((count, encoder.dim), dtype=np.float32)
    has_image = np.zeros(count, dtype=bool)
    files = [thumbnail_path(paths.thumbnails, article_id) for article_id in ids]
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as pool:
        for start in range(0, count, IMAGE_BATCH):
            loaded = list(pool.map(_load_image, files[start : start + IMAGE_BATCH]))
            present = [i for i, img in enumerate(loaded) if img is not None]
            if present:
                vectors = encoder.encode_images([loaded[i] for i in present])
                rows = start + np.asarray(present)
                image[rows] = vectors
                has_image[rows] = True
            if (start // IMAGE_BATCH) % 40 == 0:
                log(f"  images {min(start + IMAGE_BATCH, count):,}/{count:,}")
    image_seconds = time.perf_counter() - started
    log(f"image embeddings: {int(has_image.sum()):,} photos in {image_seconds:.1f}s")

    embeddings = ArticleEmbeddings(ids, image, normalize(text), has_image)
    # How often a photo's nearest text (among a sample) is its own description.
    sample = np.flatnonzero(has_image)[:: max(1, int(has_image.sum()) // 2000)]
    scores = embeddings.image[sample] @ embeddings.text[sample].T
    self_match = float(np.mean(scores.argmax(axis=1) == np.arange(len(sample))))
    manifest = {
        "model": getattr(encoder, "name", "unknown"),
        "model_revision": getattr(encoder, "revision", None),
        "dim": encoder.dim,
        "articles": count,
        "with_image": int(has_image.sum()),
        "text_seconds": round(text_seconds, 1),
        "image_seconds": round(image_seconds, 1),
        "image_to_text_top1_on_sample": round(self_match, 4),
        "sample_size": len(sample),
    }
    embeddings.save(paths.embeddings, manifest)
    return manifest
