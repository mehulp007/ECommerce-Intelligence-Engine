from __future__ import annotations

from typer.testing import CliRunner

from ecommerce_intelligence import cli


def test_build_runs_every_pipeline_in_dependency_order(monkeypatch) -> None:
    calls = []
    for name in (
        "data_all",
        "search_embed",
        "search_evaluate",
        "search_galaxy",
        "recsys_train",
        "customers_train",
        "serve_prepare",
    ):
        monkeypatch.setattr(
            cli, name, lambda *a, _name=name, **k: calls.append((_name, k))
        )
    result = CliRunner().invoke(cli.app, ["build", "--trials", "3", "--no-ablations"])
    assert result.exit_code == 0, result.output
    # Embeddings feed the map, recommender, and segments; scores feed serving.
    assert [name for name, _ in calls] == [
        "data_all",
        "search_embed",
        "search_evaluate",
        "search_galaxy",
        "recsys_train",
        "customers_train",
        "serve_prepare",
    ]
    assert dict(calls)["customers_train"] == {"trials": 3}
    assert dict(calls)["recsys_train"] == {"ablations": False, "score": True}
