"""Filesystem layout for local data, generated artifacts, and reports."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _default_root() -> Path:
    configured = os.getenv("EIE_PROJECT_ROOT")
    if configured:
        return Path(configured).resolve()
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ProjectPaths:
    root: Path

    @classmethod
    def default(cls) -> ProjectPaths:
        return cls(_default_root())

    @property
    def raw(self) -> Path:
        return self.root / "data" / "raw" / "hm"

    @property
    def archive(self) -> Path:
        return self.raw / "h-and-m-personalized-fashion-recommendations.zip"

    @property
    def csv(self) -> Path:
        return self.raw / "csv"

    @property
    def parquet(self) -> Path:
        return self.root / "data" / "parquet"

    @property
    def thumbnails(self) -> Path:
        return self.root / "data" / "thumbnails"

    @property
    def embeddings(self) -> Path:
        return self.root / "data" / "embeddings"

    @property
    def warehouse(self) -> Path:
        return self.root / "data" / "warehouse.duckdb"

    @property
    def duckdb_temp(self) -> Path:
        return self.root / "data" / "duckdb_tmp"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    def table(self, name: str) -> Path:
        return self.parquet / f"{name}.parquet"
