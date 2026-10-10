"""Command-line entry point: `eie <group> <command>`."""

from __future__ import annotations

import time

import typer

from ecommerce_intelligence.paths import ProjectPaths

app = typer.Typer(
    help="E-Commerce Intelligence Engine pipelines.", no_args_is_help=True
)
data_app = typer.Typer(
    help="Download, ingest, and validate the H&M dataset.", no_args_is_help=True
)
app.add_typer(data_app, name="data")


def _paths() -> ProjectPaths:
    return ProjectPaths.default()


def _done(label: str, started: float) -> None:
    typer.echo(f"{label} finished in {time.perf_counter() - started:,.1f}s")


@data_app.command("download")
def data_download(force: bool = typer.Option(False, help="Download again.")) -> None:
    """Download the competition archive (~29 GB) with your Kaggle token."""
    from ecommerce_intelligence.data.source import download

    started = time.perf_counter()
    archive = download(_paths(), force=force)
    typer.echo(f"Archive: {archive} ({archive.stat().st_size / 1e9:,.2f} GB)")
    _done("Download", started)


@data_app.command("ingest")
def data_ingest() -> None:
    """Extract the CSV tables and convert them to typed Parquet."""
    from ecommerce_intelligence.data.ingest import ingest
    from ecommerce_intelligence.data.source import extract_csvs

    paths = _paths()
    started = time.perf_counter()
    for name, size in extract_csvs(paths).items():
        typer.echo(f"CSV {name}: {size / 1e6:,.1f} MB")
    for name, rows in ingest(paths).items():
        typer.echo(f"Parquet {name}: {rows:,} rows")
    _done("Ingest", started)


@data_app.command("warehouse")
def data_warehouse() -> None:
    """Build data/warehouse.duckdb from the Parquet tables."""
    from ecommerce_intelligence.data.warehouse import build_warehouse

    started = time.perf_counter()
    for name, rows in build_warehouse(_paths()).items():
        typer.echo(f"Table {name}: {rows:,} rows")
    _done("Warehouse", started)


@data_app.command("quality")
def data_quality() -> None:
    """Profile the tables and fail if integrity checks do not pass."""
    from ecommerce_intelligence.data.quality import write_report

    started = time.perf_counter()
    target, report, failures = write_report(_paths())
    t = report["transactions"]
    typer.echo(
        f"{t['rows']:,} transactions, {t['customers']:,} buyers, "
        f"{t['articles']:,} articles sold, {t['first_date']} to {t['last_date']}"
    )
    typer.echo(f"Report: {target}")
    _done("Quality check", started)
    if failures:
        for failure in failures:
            typer.echo(f"FAILED: {failure}", err=True)
        raise typer.Exit(code=1)
    typer.echo("All integrity checks passed.")


@data_app.command("thumbnails")
def data_thumbnails(
    workers: int = typer.Option(0, help="Worker processes (0 = automatic)."),
    limit: int = typer.Option(0, help="Only process the first N images (0 = all)."),
) -> None:
    """Resize product photos into 256x384 WebP thumbnails."""
    from ecommerce_intelligence.data.images import build_thumbnails

    started = time.perf_counter()
    summary = build_thumbnails(_paths(), workers=workers or None, limit=limit or None)
    typer.echo(
        f"{summary['created']:,} created, {summary['skipped_existing']:,} already "
        f"present, {summary['failed']:,} failed; {summary['thumbnail_files']:,} "
        f"thumbnails using {summary['thumbnail_bytes'] / 1e9:,.2f} GB"
    )
    _done("Thumbnails", started)


@data_app.command("all")
def data_all() -> None:
    """Run download, ingest, warehouse, quality, and thumbnails in order."""
    data_download(force=False)
    data_ingest()
    data_warehouse()
    data_quality()
    data_thumbnails(workers=0, limit=0)


if __name__ == "__main__":
    app()
