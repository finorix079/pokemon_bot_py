"""tools/pokemon_tools.py — named-tool registry for the chatbot pipeline.

Ported from open-react-template/ed_tools.ts.

The TypeScript file wraps each tool with `wrapTool(name, fn)` so ElasticDash
can intercept calls for telemetry / mocking. In the Python CLI port we use
a passthrough `wrap_tool(name, fn)` that just returns `fn` unchanged — the
shape of every tool function (single `input` dict arg, awaited result) is
preserved so `agent_tools` in `utils/ai_handler.py` can call them through
the same interface as the TS code.

The seven exported names (`apiService`, `queryRefinement`, `searchPokemon`,
`fetchPokemonDetails`, `searchMove`, `searchBerry`, `searchAbility`) keep
their camelCase TS spelling so anything that looks them up by string key
(e.g. the agent_tools map, planner LLM output) continues to match.
"""

from __future__ import annotations

import inspect
from typing import Any, Awaitable, Callable, TypeVar

from ..services.api_service import dynamic_api_request
from ..services.pokemon_service import (
    fetch_pokemon_details_tool,
    search_ability_tool,
    search_berry_tool,
    search_move_tool,
    search_pokemon_tool,
)
from ..utils.query_refinement import clarify_and_refine_user_input

F = TypeVar("F", bound=Callable[..., Any])


try:
    from elasticdash_test import ed_tool as _ed_tool  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover — SDK is a runtime dep but stay resilient
    _ed_tool = None  # type: ignore[assignment]


def wrap_tool(name: str, fn: F) -> F:
    """Register a tool with the ElasticDash SDK.

    Uses `elasticdash_test.ed_tool(name=…)` which:
    1. Adds the function to the global rerun registry so the MCP server
       and `elasticdash run-tool <name> --input '<json>'` CLI can locate
       it for behaviour validation.
    2. Wraps it with telemetry — every call emits a `tool` WorkflowEvent
       inside the active observability context (`init_observability` +
       `start_trace` are wired in `pokemon_bot/__main__.py`).

    With no observability context the wrapper is a passthrough. Falls
    back to a passthrough if the SDK import fails entirely.
    """
    if _ed_tool is None:
        return fn
    return _ed_tool(name=name)(fn)  # type: ignore[return-value]


async def maybe_await(value: Any) -> Any:
    """Await `value` if it's a coroutine/awaitable, otherwise return as-is.

    Used by `pokemon_bot.utils.ai_handler._build_agent_tools` to support
    both sync and async tool implementations behind a uniform interface.
    """
    if inspect.isawaitable(value):
        return await value
    return value


# ---------------------------------------------------------------------------
# Tool implementations — one wrapper per TS export.
# Each accepts a single dict input and forwards to the relevant service.
# ---------------------------------------------------------------------------


async def _apiService_impl(input_: dict[str, Any]) -> Any:
    """Dynamic API request tool. Mirrors TS `apiService` in ed_tools.ts."""
    base_url = input_.get("baseUrl", "")
    schema = input_.get("schema") or {}
    return await dynamic_api_request(base_url, schema)


async def _queryRefinement_impl(input_: dict[str, Any]) -> dict[str, Any]:
    """Refine and clarify user queries. Mirrors TS `queryRefinement` in ed_tools.ts.

    The TS version forcibly coerces `apiNeeds`, `concepts`, `entities` to
    arrays before returning ("guards against old recorded snapshots where
    these fields may have been missing"). The same guard is preserved here.
    """
    print(f"queryRefinement input: {input_}")
    user_input = input_.get("userInput", "")
    user_token = input_.get("userToken")
    try:
        res = await clarify_and_refine_user_input(user_input, user_token)
    except Exception as error:
        print(f"Error in clarifyAndRefineUserInput: {error}")
        raise
    print(f"queryRefinement output: {res}")
    output: dict[str, Any] = dict(res)
    output["apiNeeds"] = (
        output["apiNeeds"] if isinstance(output.get("apiNeeds"), list) else []
    )
    output["concepts"] = (
        output["concepts"] if isinstance(output.get("concepts"), list) else []
    )
    output["entities"] = (
        output["entities"] if isinstance(output.get("entities"), list) else []
    )
    return output


async def _searchPokemon_impl(input_: dict[str, Any]) -> Any:
    """Search Pokémon by name or list by page.

    Calls PokéAPI GET /pokemon/{name} or /pokemon/?limit=20&offset=…
    """
    return await search_pokemon_tool(**_normalise_kwargs(input_, allowed={"searchterm", "page", "sortby", "filter"}))


async def _fetchPokemonDetails_impl(input_: dict[str, Any]) -> Any:
    """Fetch full Pokémon details (stats, types, abilities, moves, flavor text).

    Calls PokéAPI GET /pokemon/{id} + /pokemon-species/{id} + per-ability details.
    """
    return await fetch_pokemon_details_tool(input_.get("id"))


async def _searchMove_impl(input_: dict[str, Any]) -> Any:
    """Search moves by name or list by page.

    Calls PokéAPI GET /move/{name} or /move/?limit=20&offset=…
    """
    return await search_move_tool(**_normalise_kwargs(input_, allowed={"searchterm", "page"}))


async def _searchBerry_impl(input_: dict[str, Any]) -> Any:
    """Search berries by name or list by page.

    Calls PokéAPI GET /berry/{name} or /berry/?limit=20&offset=…
    """
    return await search_berry_tool(**_normalise_kwargs(input_, allowed={"query", "page"}))


async def _searchAbility_impl(input_: dict[str, Any]) -> Any:
    """Search abilities by name or list by page.

    Calls PokéAPI GET /ability/{name} or /ability/?limit=20&offset=…
    """
    return await search_ability_tool(**_normalise_kwargs(input_, allowed={"searchterm", "page"}))


def _normalise_kwargs(
    input_: dict[str, Any], *, allowed: set[str]
) -> dict[str, Any]:
    """Drop keys not in `allowed`, mirroring TS destructuring with defaults.

    The TS code uses `({searchterm = '', page = 0}: {...}) => ...` which
    silently ignores extra keys; Python kwargs would raise `TypeError`, so
    we filter here.
    """
    return {k: v for k, v in input_.items() if k in allowed}


# ---------------------------------------------------------------------------
# Public names — match the TS `ed_tools` exports exactly (camelCase).
# `agent_tools` in ai_handler.py looks tools up by these names.
# ---------------------------------------------------------------------------

apiService = wrap_tool("apiService", _apiService_impl)
queryRefinement = wrap_tool("queryRefinement", _queryRefinement_impl)
searchPokemon = wrap_tool("searchPokemon", _searchPokemon_impl)
fetchPokemonDetails = wrap_tool("fetchPokemonDetails", _fetchPokemonDetails_impl)
searchMove = wrap_tool("searchMove", _searchMove_impl)
searchBerry = wrap_tool("searchBerry", _searchBerry_impl)
searchAbility = wrap_tool("searchAbility", _searchAbility_impl)


__all__ = [
    "wrap_tool",
    "maybe_await",
    "apiService",
    "queryRefinement",
    "searchPokemon",
    "fetchPokemonDetails",
    "searchMove",
    "searchBerry",
    "searchAbility",
]
