"""Exact inner-product search over unit-length article embeddings."""

from __future__ import annotations

from dataclasses import dataclass, field

import faiss
import numpy as np

from ecommerce_intelligence.search.embeddings import ArticleEmbeddings
from ecommerce_intelligence.search.encoder import Encoder, normalize

# Text queries are scored against both the photo and the description. Each
# modality is mean-centred first: raw CLIP text-to-image scores suffer from
# "hub" articles that sit close to every query. Weights were chosen on the
# query benchmark in search/evaluate.py (precision@10 0.95 vs 0.22 uncentred).
IMAGE_WEIGHT = 0.7
TEXT_WEIGHT = 0.3


@dataclass
class SearchHit:
    article_id: str
    score: float


@dataclass
class ProductIndex:
    embeddings: ArticleEmbeddings
    position: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        e = self.embeddings
        self.position = {aid: i for i, aid in enumerate(e.article_ids)}
        self.items = np.ascontiguousarray(e.item, dtype=np.float32)
        self.similar_index = faiss.IndexFlatIP(self.items.shape[1])
        self.similar_index.add(self.items)
        # Concatenating weighted halves makes one inner product equal to
        # IMAGE_WEIGHT * cos(query, photo) + TEXT_WEIGHT * cos(query, text).
        image = np.where(e.has_image[:, None], e.image, e.text)
        self.image_mean = image.mean(axis=0)
        self.text_mean = e.text.mean(axis=0)
        joint = np.hstack(
            [
                normalize(image - self.image_mean) * np.sqrt(IMAGE_WEIGHT),
                normalize(e.text - self.text_mean) * np.sqrt(TEXT_WEIGHT),
            ]
        ).astype(np.float32)
        self.text_index = faiss.IndexFlatIP(joint.shape[1])
        self.text_index.add(np.ascontiguousarray(joint))

    def similar(self, article_id: str, k: int = 12) -> list[SearchHit]:
        """Articles that look most like `article_id`, excluding itself."""
        if article_id not in self.position:
            raise KeyError(article_id)
        query = self.items[[self.position[article_id]]]
        scores, rows = self.similar_index.search(query, k + 1)
        return [
            SearchHit(str(self.embeddings.article_ids[row]), float(score))
            for score, row in zip(scores[0], rows[0], strict=True)
            if row >= 0 and self.embeddings.article_ids[row] != article_id
        ][:k]

    def search_vector(self, query: np.ndarray, k: int = 12) -> list[SearchHit]:
        query = np.asarray(query, dtype=np.float32).reshape(1, -1)
        query = normalize(query - self.text_mean)
        joint = np.hstack([query * np.sqrt(IMAGE_WEIGHT), query * np.sqrt(TEXT_WEIGHT)])
        scores, rows = self.text_index.search(np.ascontiguousarray(joint), k)
        return [
            SearchHit(str(self.embeddings.article_ids[row]), float(score))
            for score, row in zip(scores[0], rows[0], strict=True)
            if row >= 0
        ]

    def search_text(self, encoder: Encoder, query: str, k: int = 12) -> list[SearchHit]:
        if not query.strip():
            raise ValueError("Search text is empty")
        return self.search_vector(encoder.encode_texts([query.strip()])[0], k)
