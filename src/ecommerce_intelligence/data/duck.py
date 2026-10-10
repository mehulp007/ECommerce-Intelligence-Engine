"""DuckDB connections tuned for larger-than-memory work on a laptop."""

from __future__ import annotations

import os
from pathlib import Path

import duckdb

from ecommerce_intelligence.paths import ProjectPaths


def sql_path(path: Path) -> str:
    """Quote a filesystem path as a DuckDB string literal."""
    return "'" + path.resolve().as_posix().replace("'", "''") + "'"


def connect(
    paths: ProjectPaths,
    database: Path | str = ":memory:",
    *,
    read_only: bool = False,
) -> duckdb.DuckDBPyConnection:
    paths.duckdb_temp.mkdir(parents=True, exist_ok=True)
    connection = duckdb.connect(str(database), read_only=read_only)
    memory = os.getenv("EIE_DUCKDB_MEMORY", "6GB")
    threads = int(
        os.getenv("EIE_DUCKDB_THREADS", str(max(1, (os.cpu_count() or 2) - 2)))
    )
    connection.execute(f"SET memory_limit = '{memory}'")
    connection.execute(f"SET threads = {threads}")
    connection.execute(f"SET temp_directory = {sql_path(paths.duckdb_temp)}")
    connection.execute("SET preserve_insertion_order = false")
    return connection
