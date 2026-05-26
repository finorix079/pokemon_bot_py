"""tests/conftest.py — shared pytest fixtures for pokemon_bot smoke tests.

We never make real LLM or PokéAPI calls in tests. All three LLM client
functions (`openai_chat_completion`, `kimi_chat_completion`,
`anthropic_chat_completion` / `anthropic_stream_text`) are replaced with
in-memory async stubs that the tests configure per case.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Awaitable, Callable

import pytest


@pytest.fixture(autouse=True)
def _stub_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide fake API keys so `pokemon_bot.config.load()` doesn't raise.

    Post-migration (2026-05-26): `ANTHROPIC_API_KEY` is the only required
    key for the chat pipeline. `OPENAI_API_KEY` is kept as an empty
    optional for back-compat with code that still reads the field on the
    `Config` dataclass; `AI_PROVIDER` is no longer consulted.
    """
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-anthropic-key")
    monkeypatch.setenv("KIMI_API_KEY", "test-kimi-key")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AI_PROVIDER", raising=False)


class LLMRecorder:
    """Records every call made through the stubbed LLM clients so tests can
    assert what messages were sent. Defaults to a noop string response; per-
    test customisation via `recorder.set_response(name, value_or_callable)`.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self._responses: dict[str, Any] = {}

    def set_response(self, name: str, value: Any) -> None:
        self._responses[name] = value

    def _resolve(self, name: str, kwargs: dict[str, Any]) -> str:
        resp = self._responses.get(name, "")
        if callable(resp):
            return resp(kwargs)
        return resp

    async def stub(self, name: str, **kwargs: Any) -> str:
        self.calls.append({"name": name, "kwargs": kwargs})
        return self._resolve(name, kwargs)


@pytest.fixture
def llm_recorder(monkeypatch: pytest.MonkeyPatch) -> LLMRecorder:
    """Replace LLM functions in every consumer module with `LLMRecorder.stub`."""
    rec = LLMRecorder()

    async def openai_stub(**kw: Any) -> str:
        return await rec.stub("openai", **kw)

    async def kimi_stub(**kw: Any) -> str:
        return await rec.stub("kimi", **kw)

    async def anthropic_stub(**kw: Any) -> str:
        return await rec.stub("anthropic", **kw)

    async def anthropic_stream_stub(
        *,
        messages: list[dict[str, str]],
        model: str = "claude-sonnet-4-5-20250929",
        on_delta: Callable[[str], None] | None = None,
    ) -> tuple[str, int]:
        text = await rec.stub(
            "anthropic_stream", messages=messages, model=model
        )
        if on_delta is not None:
            for token in text:
                on_delta(token)
        return text, len(text)

    # Patch each consumer module's imported alias
    import pokemon_bot.utils.ai_handler as ai_handler

    monkeypatch.setattr(ai_handler, "openai_chat_completion", openai_stub)
    monkeypatch.setattr(ai_handler, "kimi_chat_completion", kimi_stub)
    monkeypatch.setattr(ai_handler, "anthropic_chat_completion", anthropic_stub)
    monkeypatch.setattr(ai_handler, "anthropic_stream_text", anthropic_stream_stub)

    for mod_name in (
        "pokemon_bot.utils.query_refinement",
        "pokemon_bot.chat.planner",
        "pokemon_bot.chat.planner_utils",
        "pokemon_bot.chat.validators",
        "pokemon_bot.chat.executor",
        "pokemon_bot.chat.message_utils",
        "pokemon_bot.chat.handler",
        "pokemon_bot.services.chat_planner_service",
    ):
        try:
            mod = __import__(mod_name, fromlist=["*"])
        except ImportError:
            continue
        if hasattr(mod, "openai_chat_completion"):
            monkeypatch.setattr(mod, "openai_chat_completion", openai_stub)
        if hasattr(mod, "kimi_chat_completion"):
            monkeypatch.setattr(mod, "kimi_chat_completion", kimi_stub)
        if hasattr(mod, "anthropic_chat_completion"):
            monkeypatch.setattr(mod, "anthropic_chat_completion", anthropic_stub)
        if hasattr(mod, "anthropic_stream_text"):
            monkeypatch.setattr(mod, "anthropic_stream_text", anthropic_stream_stub)

    return rec


@pytest.fixture
def stub_pokeapi(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace the `apiService` tool + `fetch_pokemon_details_tool` with
    in-memory stubs. Returns a dict the test can mutate to set per-key
    responses.
    """
    responses: dict[str, Any] = {}

    async def fake_api_service(input_: dict[str, Any]) -> Any:
        schema = input_.get("schema") or {}
        key = f"{schema.get('method', '').lower()} {schema.get('path', '')}"
        return responses.get(key, {"success": True, "result": {}})

    async def fake_fetch_pokemon_details(id: Any) -> Any:
        return responses.get(f"pokemon:{id}", {"success": True, "result": {"id": id}})

    import pokemon_bot.tools.pokemon_tools as pt

    monkeypatch.setattr(pt, "apiService", fake_api_service)
    monkeypatch.setattr(pt, "fetchPokemonDetails", fake_fetch_pokemon_details)

    # Force the agent_tools cache to rebuild with the new attributes
    import pokemon_bot.utils.ai_handler as ai_handler

    ai_handler._agent_tools_cache = None  # type: ignore[attr-defined]

    return responses
