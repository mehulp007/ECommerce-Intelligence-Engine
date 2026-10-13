from __future__ import annotations

import json
from types import SimpleNamespace

import duckdb
import pandas as pd
import pytest
from conftest import Encoder16
from fastapi.testclient import TestClient
from google.genai import types
from pydantic import SecretStr

from ecommerce_intelligence.api.app import create_app
from ecommerce_intelligence.api.settings import Settings
from ecommerce_intelligence.copilot.agent import (
    MAX_STEPS,
    Agent,
    Call,
    GeminiChat,
    GeminiSession,
    Result,
    Turn,
)
from ecommerce_intelligence.copilot.prompt import system_prompt
from ecommerce_intelligence.copilot.sandbox import Sandbox, UnsafeQuery, check
from ecommerce_intelligence.copilot.tools import DECLARATIONS, Toolbox, ToolError
from ecommerce_intelligence.search.embeddings import ArticleEmbeddings
from ecommerce_intelligence.search.index import ProductIndex
from ecommerce_intelligence.serving.store import ServingStore

KEY = "test-key"
AUTH = {"Authorization": f"Bearer {KEY}"}

SAFE = [
    "SELECT count(*) FROM sales",
    "select sum(price) from sales where sales_channel = 'online';",
    "WITH t AS (SELECT article_id, count(*) n FROM sales GROUP BY 1) "
    "SELECT * FROM t JOIN articles USING (article_id)",
    "SELECT * FROM (SELECT customer_id FROM customer_scores) s",
    "SELECT * FROM unnest([1, 2, 3])",
]
UNSAFE = {
    "DELETE FROM sales": "single SELECT",
    "DROP TABLE articles": "single SELECT",
    "CREATE TABLE x AS SELECT 1": "single SELECT",
    "SELECT 1; SELECT 2": "one statement",
    "ATTACH 'x.db' AS x": "single SELECT",
    "COPY (SELECT 1) TO 'out.csv'": "single SELECT",
    "PRAGMA database_list": "single SELECT",
    "SET enable_external_access = true": "single SELECT",
    "SELECT * FROM read_csv('secret.csv')": "read_csv",
    "SELECT (SELECT max(x) FROM read_parquet('a.parquet'))": "read_parquet",
    "SELECT * FROM glob('*')": "glob",
    "SELECT getenv('GEMINI_API_KEY')": "getenv",
    "SELECT upper(getenv('PATH')) FROM sales": "getenv",
    "SELECT * FROM wh.main.customers": "unqualified",
    "SELECT * FROM transactions": "Unknown table",
    "": "empty",
}


@pytest.mark.parametrize("sql", SAFE)
def test_guard_allows_read_only_selects(sql) -> None:
    check(sql.rstrip(";"))


@pytest.mark.parametrize("sql, message", UNSAFE.items())
def test_guard_refuses_anything_else(sql, message) -> None:
    with pytest.raises(UnsafeQuery, match=message):
        check(sql)


@pytest.fixture(scope="module")
def sandbox(world) -> Sandbox:
    return Sandbox(world)


def test_sandbox_runs_queries_with_caps_and_defences(sandbox) -> None:
    result = sandbox.run("SELECT index_group_name, count(*) AS n FROM sales GROUP BY 1")
    assert result.columns == ["index_group_name", "n"] and len(result.rows) == 5
    capped = sandbox.run("SELECT * FROM sales", max_rows=7)
    assert len(capped.rows) == 7 and capped.truncated
    with pytest.raises(duckdb.BinderException):
        sandbox.run("SELECT postal_code FROM customers")  # Column not exposed.
    with pytest.raises(duckdb.PermissionException):
        sandbox._con.execute("SELECT * FROM read_csv('pyproject.toml')")
    with pytest.raises(duckdb.InvalidInputException):
        sandbox._con.execute("SET enable_external_access = true")
    with pytest.raises(UnsafeQuery, match="longer than"):
        sandbox.run("SELECT count(*) FROM sales a, sales b, sales c", timeout=0.2)


def test_prompt_lists_every_table_and_column(sandbox) -> None:
    prompt = system_prompt(sandbox.schema())
    for table in (
        "sales",
        "articles",
        "customers",
        "customer_scores",
        "recommendations",
    ):
        assert f"- {table}:" in prompt
    assert "churn_probability" in prompt and "postal_code" not in prompt


@pytest.fixture(scope="module")
def toolbox_factory(world, sandbox):
    store = ServingStore(world)
    index = ProductIndex(ArticleEmbeddings.load(world.embeddings))
    return lambda: Toolbox(sandbox, store, index, Encoder16)


def test_sql_then_chart_events(toolbox_factory) -> None:
    box = toolbox_factory()
    payload, events = box.call(
        "run_sql",
        {
            "sql": "SELECT date_trunc('month', purchase_date) AS month, "
            "count(*) AS purchases FROM sales GROUP BY 1 ORDER BY 1",
            "purpose": "Monthly purchases",
        },
    )
    assert payload["result_id"] == "r1" and events[0]["type"] == "table"
    _, chart = box.call(
        "create_chart",
        {
            "result_id": "r1",
            "kind": "line",
            "x": "month",
            "y": ["purchases"],
            "title": "T",
        },
    )
    assert chart[0]["type"] == "chart" and len(chart[0]["rows"]) == payload["row_count"]
    with pytest.raises(ToolError, match="invalid"):
        box.call(
            "create_chart",
            {
                "result_id": "r1",
                "kind": "bar",
                "x": "nope",
                "y": ["purchases"],
                "title": "T",
            },
        )
    with pytest.raises(ToolError, match="No result"):
        box.call(
            "create_chart",
            {"result_id": "r9", "kind": "bar", "x": "a", "y": ["b"], "title": "T"},
        )
    with pytest.raises(ToolError, match="not allowed|single SELECT"):
        box.call("run_sql", {"sql": "DROP TABLE sales", "purpose": "x"})


def test_customer_search_campaign_tools(toolbox_factory, world) -> None:
    box = toolbox_factory()
    scores = pd.read_parquet(
        world.root / "data" / "serving" / "customer_scores.parquet"
    )
    customer_id = scores.dropna(subset=["churn_probability"])["customer_id"].iloc[0]
    payload, events = box.call("get_customer", {"customer_id": customer_id})
    assert (
        payload["customer_id"] == customer_id
        and len(payload["top_recommendations"]) == 6
    )
    assert events[0]["type"] == "customer"
    with pytest.raises(ToolError, match="64 lowercase hex"):
        box.call("get_customer", {"customer_id": "abc"})
    payload, events = box.call("search_products", {"query": "black dress", "k": 3})
    assert len(payload["products"]) == 3 and events[0]["type"] == "products"
    plan, events = box.call(
        "plan_campaign", {"budget": 0.2, "cost_per_contact": 0.005, "uplift": 0.1}
    )
    assert plan["contacted"] <= 40 and events[0]["type"] == "campaign"
    with pytest.raises(ToolError, match="Unknown tool"):
        box.call("delete_everything", {})


class ScriptedModel:
    """A fake model that replays a list of turns and records what it was sent."""

    name = "scripted"

    def __init__(self, turns):
        self.turns = list(turns)
        self.sent: list[dict] = []

    def start(self, system, tools, history):
        assert "Rules:" in system and {t["name"] for t in tools} >= {"run_sql"}
        model = self

        class Session:
            def send(self, message=None, results=None):
                model.sent.append({"message": message, "results": results})
                turn = model.turns.pop(0)
                if isinstance(turn, Exception):
                    raise turn
                return turn

        return Session()


def _agent(toolbox_factory, turns) -> tuple[Agent, ScriptedModel]:
    model = ScriptedModel(turns)
    return Agent(model, toolbox_factory(), "Rules: test", DECLARATIONS), model


def test_agent_runs_tools_then_answers(toolbox_factory) -> None:
    sql = "SELECT sales_channel, count(*) AS n FROM sales GROUP BY 1"
    agent, model = _agent(
        toolbox_factory,
        [
            Turn(calls=[Call("run_sql", {"sql": sql, "purpose": "Channel mix"}, "c1")]),
            Turn(
                calls=[
                    Call(
                        "create_chart",
                        {
                            "result_id": "r1",
                            "kind": "bar",
                            "x": "sales_channel",
                            "y": ["n"],
                            "title": "Channels",
                        },
                    )
                ]
            ),
            Turn(text="Online leads.", usage={"input_tokens": 5}),
        ],
    )
    events = list(agent.run("Which channel sells more?"))
    kinds = [e["type"] for e in events]
    assert kinds == ["tool_call", "table", "tool_call", "chart", "answer", "done"]
    assert (
        events[-2]["text"] == "Online leads."
        and events[-1]["usage"]["input_tokens"] == 5
    )
    first_results = model.sent[1]["results"]
    assert (
        first_results[0].call.id == "c1" and first_results[0].payload["row_count"] == 2
    )


def test_agent_reports_tool_errors_so_the_model_can_retry(toolbox_factory) -> None:
    agent, model = _agent(
        toolbox_factory,
        [
            Turn(
                calls=[
                    Call("run_sql", {"sql": "SELECT nope FROM sales", "purpose": "x"})
                ]
            ),
            Turn(text="Fixed."),
        ],
    )
    events = list(agent.run("q"))
    assert [e["type"] for e in events] == ["tool_call", "tool_error", "answer", "done"]
    assert "error" in model.sent[1]["results"][0].payload


def test_agent_stops_after_the_step_limit_and_on_provider_errors(
    toolbox_factory,
) -> None:
    looping = [Turn(calls=[Call("search_products", {"query": "red"})])] * (
        MAX_STEPS + 1
    )
    agent, _ = _agent(toolbox_factory, looping)
    events = list(agent.run("q"))
    assert events[-2]["type"] == "error" and "Stopped after" in events[-2]["message"]
    agent, _ = _agent(toolbox_factory, [RuntimeError("quota exceeded")])
    events = list(agent.run("q"))
    assert [e["type"] for e in events] == ["error", "done"]
    assert "quota exceeded" in events[0]["message"]


def test_gemini_session_keeps_model_content_and_answers_by_call_id() -> None:
    model_content = types.Content(
        role="model",
        parts=[
            types.Part(text="thinking", thought=True),
            types.Part(
                function_call=types.FunctionCall(
                    id="call-7", name="run_sql", args={"sql": "S"}
                ),
                thought_signature=b"sig",
            ),
        ],
    )
    responses = [
        SimpleNamespace(
            candidates=[SimpleNamespace(content=model_content)],
            usage_metadata=SimpleNamespace(
                prompt_token_count=10, candidates_token_count=2, thoughts_token_count=3
            ),
        ),
        SimpleNamespace(
            candidates=[
                SimpleNamespace(
                    content=types.Content(role="model", parts=[types.Part(text="Done")])
                )
            ],
            usage_metadata=None,
        ),
    ]
    seen = []

    def generate_content(model, contents, config):
        seen.append(list(contents))
        return responses.pop(0)

    chat = _fake_chat(generate_content, "gemini-test", None)
    session = GeminiSession(
        chat,
        "system",
        DECLARATIONS,
        [{"role": "assistant", "content": "hi"}],
        "gemini-test",
    )
    turn = session.send(message="question")
    assert turn.text == "" and turn.calls[0].id == "call-7"
    assert turn.usage == {"input_tokens": 10, "output_tokens": 2, "thinking_tokens": 3}
    final = session.send(results=[Result(turn.calls[0], {"rows": [[1]]})])
    assert final.text == "Done"
    second_request = seen[1]
    assert second_request[0].role == "model"  # History mapped from "assistant".
    assert second_request[2].parts[1].thought_signature == b"sig"  # Preserved.
    response_part = second_request[3].parts[0].function_response
    assert response_part.id == "call-7" and response_part.response == {"rows": [[1]]}


def _client(world, factory):
    settings = Settings(eie_api_key=SecretStr(KEY), gemini_api_key=None, _env_file=None)
    return TestClient(
        create_app(
            settings, world, encoder_factory=Encoder16, chat_model_factory=factory
        )
    )


def test_chat_endpoint_streams_events(world) -> None:
    turns = [
        Turn(
            calls=[
                Call(
                    "run_sql",
                    {"sql": "SELECT count(*) AS n FROM sales", "purpose": "n"},
                )
            ]
        ),
        Turn(text="There are many purchases."),
    ]
    with _client(world, lambda: ScriptedModel(turns)) as client:
        status = client.get("/api/copilot/status", headers=AUTH).json()
        assert status["configured"] is True and "sales" in status["tables"]
        response = client.post(
            "/api/copilot/chat",
            json={"messages": [{"role": "user", "content": "How many purchases?"}]},
            headers=AUTH,
        )
        assert response.headers["content-type"].startswith("text/event-stream")
        events = [
            json.loads(line[len("data: ") :])
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]
        assert [e["type"] for e in events] == ["tool_call", "table", "answer", "done"]
        bad = client.post(
            "/api/copilot/chat",
            json={"messages": [{"role": "assistant", "content": "hi"}]},
            headers=AUTH,
        )
        assert bad.status_code == 422
        assert (
            client.post(
                "/api/copilot/chat", json={"messages": []}, headers=AUTH
            ).status_code
            == 422
        )


def test_chat_endpoint_requires_a_configured_model(world) -> None:
    with _client(world, None) as client:
        assert (
            client.get("/api/copilot/status", headers=AUTH).json()["configured"]
            is False
        )
        response = client.post(
            "/api/copilot/chat",
            json={"messages": [{"role": "user", "content": "hi"}]},
            headers=AUTH,
        )
        assert (
            response.status_code == 503
            and "GEMINI_API_KEY" in response.json()["detail"]
        )
        assert (
            client.post("/api/copilot/chat", json={"messages": []}).status_code == 401
        )


def _busy(code: int = 503):
    from google.genai import errors

    return errors.ServerError(code, {"error": {"code": code, "message": "busy"}})


def _session_with(outcomes, fallback="fallback-model"):
    """A Gemini session whose fake client replays errors or text answers."""
    calls = []

    def generate_content(model, contents, config):
        calls.append(model)
        outcome = outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        content = types.Content(role="model", parts=[types.Part(text=outcome)])
        return SimpleNamespace(
            candidates=[SimpleNamespace(content=content)], usage_metadata=None
        )

    chat = _fake_chat(generate_content, fallback=fallback)
    return chat.start("system", DECLARATIONS, []), calls


def test_busy_primary_switches_to_the_fallback_before_the_first_answer() -> None:
    session, calls = _session_with([_busy(), "From fallback"])
    assert session.send(message="q").text == "From fallback"
    assert calls == ["primary-model", "fallback-model"]
    assert session.model_name == "fallback-model"


def test_after_the_first_answer_the_model_never_changes() -> None:
    session, calls = _session_with(["first", _busy(), _busy(), "recovered"])
    session.send(message="q")
    assert session.send(message="again").text == "recovered"
    assert calls == ["primary-model"] * 4


def test_non_busy_errors_and_exhausted_retries_are_raised() -> None:
    from google.genai import errors

    bad_request = errors.ClientError(400, {"error": {"code": 400, "message": "bad"}})
    session, calls = _session_with([bad_request])
    with pytest.raises(errors.ClientError):
        session.send(message="q")
    assert calls == ["primary-model"]
    session, calls = _session_with([_busy()] * 6, fallback=None)
    with pytest.raises(errors.ServerError):
        session.send(message="q")
    assert len(calls) == 4  # First call plus three patient retries.


def _fake_chat(generate_content, name="primary-model", fallback="fallback-model"):
    """A GeminiChat whose client, sleep, and clock are test doubles."""
    chat = GeminiChat.__new__(GeminiChat)
    chat.client = SimpleNamespace(
        models=SimpleNamespace(generate_content=generate_content)
    )
    chat.name, chat.fallback, chat.temperature = name, fallback, 0.0
    chat.unavailable_until = {}
    chat.sleep = lambda seconds: None
    chat.clock = lambda: 1000.0
    return chat


def _quota(seconds: float = 37888.0):
    from google.genai import errors

    detail = {
        "@type": "type.googleapis.com/google.rpc.RetryInfo",
        "retryDelay": f"{seconds:.0f}s",
    }
    return errors.ClientError(
        429, {"error": {"code": 429, "message": "quota", "details": [detail]}}
    )


def test_exhausted_daily_quota_marks_the_model_and_switches() -> None:
    from ecommerce_intelligence.copilot.agent import retry_after_seconds

    assert retry_after_seconds(_quota(37888)) == 37888.0
    session, calls = _session_with([_quota(), "From fallback"])
    assert session.send(message="q").text == "From fallback"
    assert calls == ["primary-model", "fallback-model"]
    chat = session.chat
    assert chat.unavailable_until["primary-model"] == 1000.0 + 37888.0
    # The next conversation starts on the fallback directly...
    assert chat.start("system", DECLARATIONS, []).model_name == "fallback-model"
    # ...until the primary's quota resets.
    chat.clock = lambda: 1000.0 + 40000.0
    assert chat.start("system", DECLARATIONS, []).model_name == "primary-model"


def test_quota_exhausted_mid_answer_raises_model_unavailable() -> None:
    from ecommerce_intelligence.copilot.agent import ModelUnavailable

    session, calls = _session_with(["first", _quota()])
    session.send(message="q")
    with pytest.raises(ModelUnavailable, match="quota exhausted"):
        session.send(message="again")
    assert calls == ["primary-model", "primary-model"]  # No retries, no switch.
    session.chat.unavailable_until["fallback-model"] = 9e9
    with pytest.raises(ModelUnavailable):
        session.chat.start("system", DECLARATIONS, [])


def test_agent_restarts_once_on_another_model_when_quota_runs_out(
    toolbox_factory,
) -> None:
    from ecommerce_intelligence.copilot.agent import ModelUnavailable

    scripts = [
        (
            "m1",
            [
                Turn(calls=[Call("search_products", {"query": "red"})]),
                ModelUnavailable("m1", 9e3),
            ],
        ),
        ("m2", [Turn(text="Answered on the fallback.")]),
    ]

    class Switching:
        name = "m1"

        def start(self, system, tools, history):
            model, turns = scripts.pop(0)

            class Session:
                model_name = model

                def send(self, message=None, results=None):
                    turn = turns.pop(0)
                    if isinstance(turn, Exception):
                        raise turn
                    return turn

            return Session()

    agent = Agent(Switching(), toolbox_factory(), "Rules: test", DECLARATIONS)
    events = list(agent.run("q"))
    kinds = [e["type"] for e in events]
    assert kinds == ["tool_call", "products", "restart", "answer", "done"]
    assert events[-1]["model"] == "m2"


def test_agent_reports_once_when_every_model_is_exhausted(toolbox_factory) -> None:
    outcomes = [_quota(), _quota()]

    def generate_content(model, contents, config):
        raise outcomes.pop(0)

    agent = Agent(
        _fake_chat(generate_content), toolbox_factory(), "Rules: test", DECLARATIONS
    )
    events = list(agent.run("q"))
    # No restart is promised when no other model is left.
    assert [e["type"] for e in events] == ["error", "done"]
    assert "primary-model and fallback-model are unavailable" in events[0]["message"]
