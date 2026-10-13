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
recsys_app = typer.Typer(
    help="Train, evaluate, and run the product recommender.", no_args_is_help=True
)
app.add_typer(recsys_app, name="recsys")
search_app = typer.Typer(
    help="Build product embeddings and search the catalog.", no_args_is_help=True
)
app.add_typer(search_app, name="search")
customers_app = typer.Typer(
    help="Churn, lifetime value, segments, and retention campaigns.",
    no_args_is_help=True,
)
app.add_typer(customers_app, name="customers")
serve_app = typer.Typer(
    help="Build serving tables and run the API.", no_args_is_help=True
)
app.add_typer(serve_app, name="serve")
copilot_app = typer.Typer(help="Ask the Gemini analyst copilot.", no_args_is_help=True)
app.add_typer(copilot_app, name="copilot")


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


@recsys_app.command("train")
def recsys_train(
    ablations: bool = typer.Option(True, help="Measure each source's contribution."),
    score: bool = typer.Option(True, help="Write top-12 lists for every customer."),
) -> None:
    """Evaluate on the final week, train the production ranker, score customers."""
    from ecommerce_intelligence.recsys.pipeline import run

    started = time.perf_counter()
    report = run(_paths(), ablations=ablations, score_all=score, log=typer.echo)
    ranker = report["results"]["ranker"]
    typer.echo(
        f"Validation MAP@12 {ranker['map_at_12']:.5f} "
        f"(popularity baseline {report['results']['popular']['map_at_12']:.5f})"
    )
    typer.echo("Report: reports/recommender_metrics.json")
    _done("Recommender", started)


@search_app.command("evaluate")
def search_evaluate() -> None:
    """Measure text-search precision@10 on a labelled query set."""
    from ecommerce_intelligence.search.embeddings import ArticleEmbeddings
    from ecommerce_intelligence.search.encoder import FashionClipEncoder
    from ecommerce_intelligence.search.evaluate import evaluate
    from ecommerce_intelligence.search.index import ProductIndex

    paths = _paths()
    index = ProductIndex(ArticleEmbeddings.load(paths.embeddings))
    report = evaluate(paths, index, FashionClipEncoder())
    for query, score in report["per_query"].items():
        typer.echo(f"  {score:.1f}  {query}")
    typer.echo(f"Precision@10 {report['precision_at_10']:.3f}")
    typer.echo("Report: reports/search_quality.json")


@customers_app.command("train")
def customers_train(
    trials: int = typer.Option(25, help="Optuna tuning trials for the churn model."),
) -> None:
    """Evaluate churn and lifetime value, fit segments, and score every customer."""
    from ecommerce_intelligence.customers.pipeline import run

    started = time.perf_counter()
    report = run(_paths(), trials=trials, log=typer.echo)
    best = report["churn"]["test_results"]["xgboost_tuned"]
    typer.echo(
        f"Churn test ROC AUC {best['roc_auc']:.3f}, Brier {best['brier']:.3f}; "
        f"{report['segments']['chosen_k']} segments; "
        f"{report['scores']['customers']:,} customers scored"
    )
    typer.echo("Report: reports/customer_metrics.json")
    _done("Customer intelligence", started)


@customers_app.command("campaign")
def customers_campaign(
    budget: float = typer.Option(500.0, help="Total budget in price units."),
    cost: float = typer.Option(0.005, help="Cost per contacted customer."),
    uplift: float = typer.Option(
        0.10, help="Share of would-be-inactive customers retained."
    ),
) -> None:
    """Plan a retention campaign from the latest customer scores."""
    import pandas as pd

    from ecommerce_intelligence.customers.campaign import plan

    scores = pd.read_parquet(
        _paths().root / "data" / "serving" / "customer_scores.parquet"
    )
    result = plan(scores, budget=budget, cost_per_contact=cost, uplift=uplift)
    typer.echo(
        f"Contact {result.contacted:,} customers for {result.spend:,.2f}: "
        "expected value "
        f"saved {result.expected_value_saved:,.2f}, net {result.expected_net:,.2f}, "
        f"ROI {result.roi:.1%}"
    )
    if result.contacted:
        top = (
            result.customers.groupby("segment_name").size().sort_values(ascending=False)
        )
        for name, count in top.head(5).items():
            typer.echo(f"  {count:>9,}  {name}")


@serve_app.command("prepare")
def serve_prepare() -> None:
    """Precompute customer history, product statistics, and dashboard aggregates."""
    from ecommerce_intelligence.serving.prepare import build

    started = time.perf_counter()
    manifest = build(_paths(), log=typer.echo)
    for name, info in manifest["files"].items():
        typer.echo(f"  {name}: {info['bytes'] / 1e6:,.1f} MB")
    _done("Serving build", started)


@serve_app.command("run")
def serve_run(
    host: str = typer.Option("127.0.0.1", help="Interface to bind; keep local."),
    port: int = typer.Option(8000, help="Port for the API."),
    reload: bool = typer.Option(False, help="Restart on code changes."),
) -> None:
    """Start the API (Swagger docs at /docs)."""
    import uvicorn

    typer.echo(f"API at http://{host}:{port} (docs at /docs)")
    uvicorn.run(
        "ecommerce_intelligence.api.app:app", host=host, port=port, reload=reload
    )


@serve_app.command("openapi")
def serve_openapi(
    output: str = typer.Option(
        "web/lib/openapi.json", help="Where to write the OpenAPI schema."
    ),
) -> None:
    """Write the API's OpenAPI schema without starting it (for the web types)."""
    import json
    from pathlib import Path

    from ecommerce_intelligence.api.app import create_app

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    schema = create_app().openapi()
    path.write_text(json.dumps(schema, indent=2) + "\n", encoding="utf-8")
    typer.echo(f"Wrote {path} ({len(schema['paths'])} paths)")


@copilot_app.command("ask")
def copilot_ask(question: str) -> None:
    """Answer one question with the copilot and print each step."""
    from ecommerce_intelligence.api.settings import Settings
    from ecommerce_intelligence.copilot.agent import Agent, GeminiChat
    from ecommerce_intelligence.copilot.prompt import system_prompt
    from ecommerce_intelligence.copilot.sandbox import Sandbox
    from ecommerce_intelligence.copilot.tools import DECLARATIONS, Toolbox
    from ecommerce_intelligence.search.embeddings import ArticleEmbeddings
    from ecommerce_intelligence.search.encoder import FashionClipEncoder
    from ecommerce_intelligence.search.index import ProductIndex
    from ecommerce_intelligence.serving.store import ServingStore

    settings = Settings()
    if not settings.gemini_api_key:
        typer.echo("Set GEMINI_API_KEY in .env first.", err=True)
        raise typer.Exit(code=1)
    paths = _paths()
    sandbox = Sandbox(paths)
    encoders = {}

    def encoder():
        if "clip" not in encoders:
            encoders["clip"] = FashionClipEncoder(device="cpu")
        return encoders["clip"]

    toolbox = Toolbox(
        sandbox,
        ServingStore(paths),
        ProductIndex(ArticleEmbeddings.load(paths.embeddings)),
        encoder,
    )
    chat = GeminiChat(
        settings.gemini_api_key.get_secret_value(),
        settings.copilot_model,
        settings.copilot_fallback_model,
    )
    agent = Agent(chat, toolbox, system_prompt(sandbox.schema()), DECLARATIONS)
    for event in agent.run(question):
        kind = event["type"]
        if kind == "tool_call":
            typer.echo(f"> {event['name']}({event['args']})")
        elif kind == "table":
            typer.echo(f"  {event['row_count']} rows: {', '.join(event['columns'])}")
        elif kind in ("chart", "products", "customer", "campaign"):
            typer.echo(f"  [{kind}] {event.get('title', '')}")
        elif kind == "tool_error":
            typer.echo(f"  tool error: {event['message']}")
        elif kind in ("answer", "thinking"):
            typer.echo()
            typer.echo(event["text"])
        elif kind == "error":
            typer.echo(f"ERROR: {event['message']}", err=True)
        elif kind == "done":
            typer.echo()
            typer.echo(f"({event['model']}, {event['seconds']} s, {event['usage']})")


def _describe(hits) -> None:
    import duckdb

    paths = _paths()
    ids = [hit.article_id for hit in hits]
    with duckdb.connect(str(paths.warehouse), read_only=True) as con:
        names = {
            row[0]: row[1:]
            for row in con.execute(
                """
                SELECT article_id, prod_name, colour_group_name, product_type_name
                FROM articles WHERE list_contains(?, article_id)
                """,
                [ids],
            ).fetchall()
        }
    for rank, hit in enumerate(hits, start=1):
        name, colour, kind = names.get(hit.article_id, ("?", "?", "?"))
        typer.echo(
            f"{rank:>2}. {hit.article_id}  {hit.score:.3f}  {name} ({colour} {kind})"
        )


@search_app.command("embed")
def search_embed(
    device: str = typer.Option("", help="cuda or cpu (default: automatic)."),
) -> None:
    """Embed every article photo and description with FashionCLIP."""
    from ecommerce_intelligence.search.embeddings import build_embeddings
    from ecommerce_intelligence.search.encoder import FashionClipEncoder

    started = time.perf_counter()
    encoder = FashionClipEncoder(device=device or None)
    typer.echo(f"Model {encoder.name} on {encoder.device}")
    manifest = build_embeddings(_paths(), encoder, log=typer.echo)
    typer.echo(
        f"{manifest['articles']:,} articles, {manifest['with_image']:,} with photos; "
        f"photo-to-own-description top-1 on a "
        f"{manifest['sample_size']:,}-article sample: "
        f"{manifest['image_to_text_top1_on_sample']:.1%}"
    )
    _done("Embeddings", started)


@search_app.command("similar")
def search_similar(
    article_id: str, k: int = typer.Option(10, help="Results to show.")
) -> None:
    """Show the articles that look most like ARTICLE_ID."""
    from ecommerce_intelligence.search.embeddings import ArticleEmbeddings
    from ecommerce_intelligence.search.index import ProductIndex, SearchHit

    article_id = article_id.zfill(10)
    index = ProductIndex(ArticleEmbeddings.load(_paths().embeddings))
    _describe([SearchHit(article_id, 1.0)])
    typer.echo("Most similar:")
    _describe(index.similar(article_id, k))


@search_app.command("query")
def search_query(text: str, k: int = typer.Option(10, help="Results to show.")) -> None:
    """Find articles matching a natural-language description."""
    from ecommerce_intelligence.search.embeddings import ArticleEmbeddings
    from ecommerce_intelligence.search.encoder import FashionClipEncoder
    from ecommerce_intelligence.search.index import ProductIndex

    index = ProductIndex(ArticleEmbeddings.load(_paths().embeddings))
    _describe(index.search_text(FashionClipEncoder(device="cpu"), text, k))


if __name__ == "__main__":
    app()
