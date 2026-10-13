"""The tool-calling loop, independent of the language model provider."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol

from ecommerce_intelligence.copilot.tools import Toolbox, ToolError

MAX_STEPS = 8
DEFAULT_MODEL = "gemini-3.8-flash"
DEFAULT_FALLBACK_MODEL = "gemini-3.6-flash"
BUSY_CODES = (429, 500, 502, 503, 504)
PATIENT_DELAYS = (4.0, 8.0, 16.0)
# A rate limit asking us to wait longer than this is treated as an exhausted
# quota (for example the free tier's daily limit), not a brief spike.
LONG_WAIT_SECONDS = 120.0


@dataclass
class Call:
    name: str
    args: dict[str, Any]
    id: str | None = None


@dataclass
class Result:
    call: Call
    payload: dict


@dataclass
class Turn:
    text: str = ""
    calls: list[Call] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)


class ModelUnavailable(RuntimeError):
    """A model cannot answer for a long time, e.g. its daily quota is used up."""

    def __init__(self, model: str, retry_after: float | None):
        self.model = model
        self.retry_after = retry_after
        wait = f" for about {retry_after / 3600:.1f} hours" if retry_after else ""
        verb = "are" if " and " in model else "is"
        super().__init__(f"{model} {verb} unavailable{wait} (quota exhausted)")


class ChatSession(Protocol):
    model_name: str

    def send(
        self, message: str | None = None, results: list[Result] | None = None
    ) -> Turn: ...


class ChatModel(Protocol):
    name: str

    def start(
        self, system: str, tools: list[dict], history: list[dict[str, str]]
    ) -> ChatSession: ...


class Agent:
    def __init__(
        self, model: ChatModel, toolbox: Toolbox, system: str, tools: list[dict]
    ):
        self.model = model
        self.toolbox = toolbox
        self.system = system
        self.tools = tools

    def run(
        self, question: str, history: list[dict[str, str]] | None = None
    ) -> Iterator[dict]:
        """Yield interface events until the model gives a final answer.

        If the model's quota runs out mid-answer, the question is asked once
        more from the start on another model, after a `restart` event.
        """
        started = time.perf_counter()
        usage: dict[str, int] = {}
        session = None
        for attempt in range(2):
            try:
                session = self.model.start(self.system, self.tools, history or [])
                yield from self._answer(session, question, usage)
                break
            except ModelUnavailable as error:
                if attempt == 0:
                    yield {
                        "type": "restart",
                        "message": f"{error}; answering again with another model.",
                    }
                    continue
                yield {"type": "error", "message": str(error)}
            except Exception as error:  # Provider failures end the stream cleanly.
                yield {"type": "error", "message": f"{type(error).__name__}: {error}"}
                break
        yield {
            "type": "done",
            # The model that last answered (sessions may switch to a fallback).
            "model": getattr(session, "model_name", None) or self.model.name,
            "seconds": round(time.perf_counter() - started, 2),
            "usage": usage,
        }

    def _answer(
        self, session: ChatSession, question: str, usage: dict[str, int]
    ) -> Iterator[dict]:
        turn = session.send(message=question)
        for _ in range(MAX_STEPS):
            _add(usage, turn.usage)
            if not turn.calls:
                yield {"type": "answer", "text": turn.text.strip()}
                return
            if turn.text.strip():
                yield {"type": "thinking", "text": turn.text.strip()}
            results = []
            for call in turn.calls:
                yield {"type": "tool_call", "name": call.name, "args": call.args}
                try:
                    payload, events = self.toolbox.call(call.name, call.args)
                    yield from events
                except ToolError as error:
                    payload = {"error": str(error)}
                    yield {
                        "type": "tool_error",
                        "name": call.name,
                        "message": str(error),
                    }
                results.append(Result(call, payload))
            turn = session.send(results=results)
        yield {
            "type": "error",
            "message": f"Stopped after {MAX_STEPS} tool steps without an answer.",
        }


def _add(total: dict[str, int], part: dict[str, int]) -> None:
    for key, value in part.items():
        total[key] = total.get(key, 0) + int(value or 0)


def retry_after_seconds(error: Exception) -> float | None:
    """The server's suggested wait, from a google.rpc.RetryInfo detail."""
    match = re.search(r"retryDelay['\"]?\s*:\s*['\"](\d+(?:\.\d+)?)s", str(error))
    return float(match.group(1)) if match else None


class GeminiChat:
    """Google Gemini through the google-genai SDK, with manual function calling.

    Tracks models whose quota is exhausted, so later questions go straight to
    the fallback model until the quota resets.
    """

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        fallback: str | None = DEFAULT_FALLBACK_MODEL,
        temperature: float = 0.2,
    ):
        from google import genai
        from google.genai import types

        # Brief spikes (503) and rate limits (429) are retried by the SDK with
        # jittered backoff; with a fallback model, give up on the primary sooner.
        self.client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(
                timeout=120_000,
                retry_options=types.HttpRetryOptions(
                    attempts=2 if fallback else 5,
                    initial_delay=2.0,
                    max_delay=30.0,
                    exp_base=2.0,
                    jitter=1.0,
                    http_status_codes=list(BUSY_CODES),
                ),
            ),
        )
        self.name = model
        self.fallback = fallback if fallback and fallback != model else None
        self.unavailable_until: dict[str, float] = {}
        self.sleep = time.sleep
        self.clock = time.time
        self.temperature = temperature

    def available_models(self) -> list[str]:
        now = self.clock()
        return [
            m
            for m in (self.name, self.fallback)
            if m and self.unavailable_until.get(m, 0.0) <= now
        ]

    def start(
        self, system: str, tools: list[dict], history: list[dict[str, str]]
    ) -> GeminiSession:
        models = self.available_models()
        if not models:
            soonest = min(self.unavailable_until.values()) - self.clock()
            raise ModelUnavailable(" and ".join(self.unavailable_until), soonest)
        return GeminiSession(self, system, tools, history, models[0])


class GeminiSession:
    def __init__(
        self, chat: GeminiChat, system: str, tools: list[dict], history, model: str
    ):
        from google.genai import types

        self.types = types
        self.chat = chat
        self.model_name = model
        self.tried = {model}
        self.started = False
        self.config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=chat.temperature,
            tools=[
                types.Tool(
                    function_declarations=[
                        types.FunctionDeclaration(**t) for t in tools
                    ]
                )
            ],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        )
        self.contents = [
            types.Content(
                role="model" if m["role"] == "assistant" else "user",
                parts=[types.Part(text=m["content"])],
            )
            for m in history
            if m.get("content")
        ]

    def _switch_model(self) -> bool:
        """Before the first answer only: thought signatures belong to one model."""
        if self.started:
            return False
        for model in self.chat.available_models():
            if model not in self.tried:
                self.model_name = model
                self.tried.add(model)
                return True
        return False

    def _generate(self):
        from google.genai import errors

        delays = iter(PATIENT_DELAYS)
        while True:
            try:
                return self.chat.client.models.generate_content(
                    model=self.model_name, contents=self.contents, config=self.config
                )
            except errors.APIError as error:
                code = getattr(error, "code", None)
                if code not in BUSY_CODES:
                    raise
                wait = retry_after_seconds(error)
                exhausted = (
                    code == 429 and wait is not None and wait > LONG_WAIT_SECONDS
                )
                if exhausted:
                    self.chat.unavailable_until[self.model_name] = (
                        self.chat.clock() + wait
                    )
                if self._switch_model():
                    continue
                if exhausted:
                    raise ModelUnavailable(self.model_name, wait) from error
                delay = next(delays, None)
                if delay is None:
                    raise
                # Committed to this model now, so wait longer between attempts.
                self.chat.sleep(delay)

    def send(
        self, message: str | None = None, results: list[Result] | None = None
    ) -> Turn:
        types = self.types
        if message is not None:
            self.contents.append(
                types.Content(role="user", parts=[types.Part(text=message)])
            )
        if results:
            self.contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            function_response=types.FunctionResponse(
                                id=r.call.id,
                                name=r.call.name,
                                response=json.loads(json.dumps(r.payload, default=str)),
                            )
                        )
                        for r in results
                    ],
                )
            )
        response = self._generate()
        self.started = True
        candidate = response.candidates[0] if response.candidates else None
        if candidate is None or candidate.content is None:
            reason = getattr(candidate, "finish_reason", None) if candidate else None
            raise RuntimeError(f"Gemini returned no content (finish reason {reason})")
        # Keeping the model's own content preserves its thought signatures,
        # which Gemini needs to continue multi-step tool calls.
        self.contents.append(candidate.content)
        text, calls = [], []
        for part in candidate.content.parts or []:
            if part.function_call is not None:
                calls.append(
                    Call(
                        name=part.function_call.name,
                        args=dict(part.function_call.args or {}),
                        id=part.function_call.id,
                    )
                )
            elif part.text and not part.thought:
                text.append(part.text)
        meta = response.usage_metadata
        usage = (
            {
                "input_tokens": meta.prompt_token_count or 0,
                "output_tokens": meta.candidates_token_count or 0,
                "thinking_tokens": meta.thoughts_token_count or 0,
            }
            if meta
            else {}
        )
        return Turn(text="".join(text), calls=calls, usage=usage)
