"""Read-only SQL for the copilot, enforced in layers.

1. The statement must parse as exactly one SELECT (DuckDB serializes only
   SELECT statements to JSON, so anything else is rejected).
2. Every table reference must be an unqualified, allow-listed table or a CTE
   defined in the query; table functions (read_csv, glob, ...) and
   environment or file functions are refused.
3. The connection itself attaches the data read-only, then disables file and
   network access and locks its configuration, so a query that slipped past
   the checks still could not write or read anything else.
4. Each query has a time limit and a row cap.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import duckdb

from ecommerce_intelligence.data.duck import sql_path
from ecommerce_intelligence.paths import ProjectPaths

MAX_ROWS = 500
TIMEOUT_SECONDS = 20.0
SAFE_TABLE_FUNCTIONS = {"unnest", "range", "generate_series"}
BLOCKED_FUNCTIONS = {
    "getenv",
    "current_setting",
    "query",
    "query_table",
    "read_text",
    "read_blob",
    "glob",
    "sniff_csv",
    "which_secret",
    "duckdb_secrets",
    "duckdb_settings",
}
# Table name -> SQL defining it over the attached, read-only databases.
TABLES = {
    "sales": "SELECT * FROM wh.main.sales",
    "articles": "SELECT * FROM wh.main.articles",
    "customers": (
        "SELECT customer_id, age, club_member_status, fashion_news_frequency, "
        "active AS account_active, fashion_news_subscriber FROM wh.main.customers"
    ),
    "customer_scores": "SELECT * FROM cp.main.customer_scores",
    "recommendations": "SELECT * FROM cp.main.recommendations",
}


class UnsafeQuery(ValueError):
    """The query is not a single read-only SELECT over allowed tables."""


@dataclass
class QueryResult:
    sql: str
    columns: list[str]
    rows: list[list]
    truncated: bool
    seconds: float
    types: list[str] = field(default_factory=list)


def copilot_database(paths: ProjectPaths) -> Path:
    return paths.root / "data" / "serving" / "copilot.duckdb"


def build_copilot_database(paths: ProjectPaths) -> Path:
    """Copy model outputs into a small DuckDB file the copilot can attach."""
    target = copilot_database(paths)
    partial = target.with_suffix(".duckdb.part")
    partial.unlink(missing_ok=True)
    serving = target.parent
    with duckdb.connect(str(partial)) as con:
        for name in ("customer_scores", "recommendations"):
            con.execute(
                f"CREATE TABLE {name} AS SELECT * FROM "
                f"read_parquet({sql_path(serving / f'{name}.parquet')})"
            )
        con.execute("CHECKPOINT")
    target.unlink(missing_ok=True)
    partial.replace(target)
    return target


def _references(node, tables: list, functions: list, ctes: set) -> None:
    if isinstance(node, dict):
        kind = node.get("type")
        if kind in ("BASE_TABLE", "TABLE_FUNCTION"):
            tables.append(node)
        if node.get("class") == "FUNCTION" or kind == "FUNCTION":
            name = node.get("function_name")
            if name:
                functions.append(name.lower())
        for key, value in node.items():
            if key == "cte_map" and isinstance(value, dict):
                for entry in value.get("map", []):
                    ctes.add(str(entry.get("key", "")).lower())
            _references(value, tables, functions, ctes)
    elif isinstance(node, list):
        for item in node:
            _references(item, tables, functions, ctes)


def check(sql: str, con: duckdb.DuckDBPyConnection | None = None) -> None:
    """Raise UnsafeQuery unless `sql` is one SELECT over allowed tables."""
    if not sql or not sql.strip():
        raise UnsafeQuery("The query is empty")
    con = con or duckdb.connect()
    tree = json.loads(con.execute("SELECT json_serialize_sql(?)", [sql]).fetchone()[0])
    if tree.get("error"):
        raise UnsafeQuery(
            f"Only a single SELECT is allowed: {tree.get('error_message')}"
        )
    if len(tree["statements"]) != 1:
        raise UnsafeQuery("Exactly one statement is allowed")
    tables: list[dict] = []
    functions: list[str] = []
    ctes: set[str] = set()
    _references(tree, tables, functions, ctes)
    for ref in tables:
        if ref["type"] == "TABLE_FUNCTION":
            name = (ref.get("function") or {}).get("function_name", "").lower()
            if name not in SAFE_TABLE_FUNCTIONS:
                raise UnsafeQuery(f"Table function {name or '?'} is not allowed")
            continue
        name = (ref.get("table_name") or "").lower()
        if ref.get("catalog_name") or ref.get("schema_name"):
            raise UnsafeQuery(
                f"Use unqualified table names, not {name} in another schema"
            )
        if name not in TABLES and name not in ctes:
            raise UnsafeQuery(
                f"Unknown table {name}; allowed: {', '.join(sorted(TABLES))}"
            )
    blocked = sorted(
        {f for f in functions if f in BLOCKED_FUNCTIONS or f.startswith("read_")}
    )
    if blocked:
        raise UnsafeQuery(f"Function {blocked[0]} is not allowed")


class Sandbox:
    """A locked-down DuckDB connection exposing only the allow-listed tables."""

    def __init__(self, paths: ProjectPaths, *, memory: str = "2GB"):
        warehouse, scores = paths.warehouse, copilot_database(paths)
        for path in (warehouse, scores):
            if not path.is_file():
                raise FileNotFoundError(f"{path.name} missing; run `eie serve prepare`")
        self._con = duckdb.connect()
        self._con.execute(f"ATTACH {sql_path(warehouse)} AS wh (READ_ONLY)")
        self._con.execute(f"ATTACH {sql_path(scores)} AS cp (READ_ONLY)")
        for name, definition in TABLES.items():
            self._con.execute(f"CREATE VIEW {name} AS {definition}")
        self._con.execute(f"SET memory_limit = '{memory}'")
        self._con.execute("SET threads = 4")
        self._con.execute("SET enable_external_access = false")
        self._con.execute("SET lock_configuration = true")
        self._lock = threading.Lock()

    def schema(self) -> dict[str, list[tuple[str, str]]]:
        return {
            name: [
                (row[0], row[1])
                for row in self._con.execute(f"DESCRIBE {name}").fetchall()
            ]
            for name in TABLES
        }

    def run(
        self, sql: str, *, max_rows: int = MAX_ROWS, timeout: float = TIMEOUT_SECONDS
    ) -> QueryResult:
        sql = sql.strip().rstrip(";").strip()
        with self._lock:
            cursor = self._con.cursor()
        try:
            check(sql, cursor)
            timer = threading.Timer(timeout, cursor.interrupt)
            started = time.perf_counter()
            timer.start()
            try:
                relation = cursor.execute(sql)
                columns = [d[0] for d in relation.description]
                types = [str(d[1]) for d in relation.description]
                rows = relation.fetchmany(max_rows + 1)
            except duckdb.InterruptException:
                raise UnsafeQuery(
                    f"The query took longer than {timeout:.0f} s"
                ) from None
            finally:
                timer.cancel()
            return QueryResult(
                sql=sql,
                columns=columns,
                rows=[[_jsonable(v) for v in row] for row in rows[:max_rows]],
                truncated=len(rows) > max_rows,
                seconds=round(time.perf_counter() - started, 3),
                types=types,
            )
        finally:
            cursor.close()


def _jsonable(value):
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if isinstance(value, float):
        return round(value, 6)
    if isinstance(value, (bytes, bytearray)):
        return value.hex()
    if hasattr(value, "item"):
        return value.item()
    return value
