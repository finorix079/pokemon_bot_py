"""services/pokemon_service.py — PokéAPI tool implementations.

Ported from open-react-template/services/pokemonService.ts.

All five active tools (`search_pokemon_tool`, `fetch_pokemon_details_tool`,
`search_move_tool`, `search_berry_tool`, `search_ability_tool`) keep the
**exact same return shapes** as their TypeScript counterparts so the rest
of the planner/executor pipeline can consume them unchanged.

Watchlist tools are kept as stubs to mirror the TS surface even though the
chat pipeline never invokes them (PokéAPI has no user-specific storage).

Networking uses `httpx.AsyncClient` so all the "parallel `Promise.all`"
patterns from the TS code map naturally to `asyncio.gather`.
"""

from __future__ import annotations

import asyncio
import math
from typing import Any, Optional
from urllib.parse import quote

import httpx

POKEAPI_BASE = "https://pokeapi.co/api/v2"
PAGE_SIZE = 20

# Shared HTTPX client per event loop. Lazy-created so importing this module
# doesn't open sockets unnecessarily (e.g. during pytest collection).
_client_cache: dict[int, httpx.AsyncClient] = {}


def _get_client() -> httpx.AsyncClient:
    """Return an AsyncClient bound to the current event loop."""
    loop = asyncio.get_event_loop()
    key = id(loop)
    client = _client_cache.get(key)
    if client is None or client.is_closed:
        client = httpx.AsyncClient(timeout=20.0)
        _client_cache[key] = client
    return client


async def _fetch_json(url: str, *, error_message: str) -> Any:
    """GET `url` and decode JSON. Raises `RuntimeError(error_message)` on
    non-2xx, matching the TS `throw new Error('...')` pattern."""
    client = _get_client()
    response = await client.get(url)
    if response.status_code >= 400:
        raise RuntimeError(error_message)
    return response.json()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_english_name(names: list[dict[str, Any]]) -> str:
    """Find the English entry in a PokéAPI `names` array. Returns "" if
    no English entry exists (caller falls back to the resource's `name`)."""
    for entry in names:
        if entry.get("language", {}).get("name") == "en":
            return entry.get("name", "") or ""
    return ""


def _get_english_short_effect(entries: list[dict[str, Any]]) -> str:
    """Find the English `short_effect` in an `effect_entries` array."""
    for entry in entries:
        if entry.get("language", {}).get("name") == "en":
            return entry.get("short_effect", "") or ""
    return ""


def _transform_pokemon_list_item(raw: dict[str, Any]) -> dict[str, Any]:
    """Transform `/pokemon/{id}` payload → list-item shape."""
    sprite = (raw.get("sprites") or {}).get("front_default")
    if not sprite:
        sprite = (
            f"https://raw.githubusercontent.com/PokeAPI/sprites/master/"
            f"sprites/pokemon/{raw['id']}.png"
        )
    return {
        "id": raw["id"],
        "identifier": raw["name"],
        "sprite": sprite,
        "types": [t["type"]["name"] for t in (raw.get("types") or [])],
    }


def _transform_move_list_item(raw: dict[str, Any]) -> dict[str, Any]:
    """Transform `/move/{id}` payload → move list-item shape."""
    localized = _get_english_name(raw.get("names") or []) or raw.get("name", "")
    return {
        "localized_name": localized,
        "type_name": (raw.get("type") or {}).get("name", "") or "",
        "power": raw.get("power"),
        "accuracy": raw.get("accuracy"),
    }


def _transform_ability_list_item(raw: dict[str, Any]) -> dict[str, Any]:
    """Transform `/ability/{id}` payload → ability list-item shape."""
    localized = _get_english_name(raw.get("names") or []) or raw.get("name", "")
    return {
        "id": raw["id"],
        "localized_name": localized,
        "short_effect": _get_english_short_effect(raw.get("effect_entries") or []),
    }


# ---------------------------------------------------------------------------
# Pokémon search
# ---------------------------------------------------------------------------


async def search_pokemon_tool(
    searchterm: str = "",
    page: int = 0,
    sortby: Optional[int] = None,
    filter: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """Search Pokémon by name (direct lookup) or return a paginated list.

    Ported from TS `searchPokemonTool`. Returns
    `{success, result: {results, totalPage}}` (camelCase preserved for
    pipeline compatibility).
    """
    _ = sortby, filter  # accepted for TS signature parity, currently unused
    try:
        if searchterm.strip():
            raw = await _fetch_json(
                f"{POKEAPI_BASE}/pokemon/{quote(searchterm.strip().lower())}",
                error_message="Pokémon not found",
            )
            return {
                "success": True,
                "result": {
                    "results": [_transform_pokemon_list_item(raw)],
                    "totalPage": 1,
                },
            }

        list_data = await _fetch_json(
            f"{POKEAPI_BASE}/pokemon/?limit={PAGE_SIZE}&offset={page * PAGE_SIZE}",
            error_message="Failed to fetch Pokémon list",
        )
        total_page = math.ceil(list_data["count"] / PAGE_SIZE)
        urls = [item["url"] for item in list_data.get("results", [])]
        raws = await asyncio.gather(
            *(_fetch_json(url, error_message="Failed to fetch Pokémon list") for url in urls)
        )
        results = [_transform_pokemon_list_item(r) for r in raws]
        return {"success": True, "result": {"results": results, "totalPage": total_page}}
    except Exception as error:  # noqa: BLE001
        print(f"Error searching Pokémon: {error}")
        raise


# ---------------------------------------------------------------------------
# Move search
# ---------------------------------------------------------------------------


async def search_move_tool(
    searchterm: str = "",
    page: int = 0,
) -> dict[str, Any]:
    """Search moves by name or return a paginated list.

    Ported from TS `searchMoveTool`.
    """
    try:
        if searchterm.strip():
            raw = await _fetch_json(
                f"{POKEAPI_BASE}/move/{quote(searchterm.strip().lower())}",
                error_message="Move not found",
            )
            return {
                "success": True,
                "result": {
                    "results": [_transform_move_list_item(raw)],
                    "totalPage": 1,
                },
            }

        list_data = await _fetch_json(
            f"{POKEAPI_BASE}/move/?limit={PAGE_SIZE}&offset={page * PAGE_SIZE}",
            error_message="Failed to fetch move list",
        )
        total_page = math.ceil(list_data["count"] / PAGE_SIZE)
        urls = [item["url"] for item in list_data.get("results", [])]
        raws = await asyncio.gather(
            *(_fetch_json(url, error_message="Failed to fetch move list") for url in urls)
        )
        results = [_transform_move_list_item(r) for r in raws]
        return {"success": True, "result": {"results": results, "totalPage": total_page}}
    except Exception as error:  # noqa: BLE001
        print(f"Error searching Move: {error}")
        raise


# ---------------------------------------------------------------------------
# Berry search
# ---------------------------------------------------------------------------


async def search_berry_tool(
    query: str = "",
    page: int = 0,
) -> dict[str, Any]:
    """Search berries by name or return a paginated list.

    Ported from TS `searchBerryTool`. The TS signature takes `{query, page}`
    (not `{searchterm, page}` like the other tools) — preserved verbatim.
    """
    try:
        if query.strip():
            raw = await _fetch_json(
                f"{POKEAPI_BASE}/berry/{quote(query.strip().lower())}",
                error_message="Berry not found",
            )
            return {"success": True, "result": {"results": [raw], "totalPage": 1}}

        list_data = await _fetch_json(
            f"{POKEAPI_BASE}/berry/?limit={PAGE_SIZE}&offset={page * PAGE_SIZE}",
            error_message="Failed to fetch berry list",
        )
        total_page = math.ceil(list_data["count"] / PAGE_SIZE)
        urls = [item["url"] for item in list_data.get("results", [])]
        results = await asyncio.gather(
            *(_fetch_json(url, error_message="Failed to fetch berry list") for url in urls)
        )
        return {"success": True, "result": {"results": results, "totalPage": total_page}}
    except Exception as error:  # noqa: BLE001
        print(f"Error searching Berry: {error}")
        raise


# ---------------------------------------------------------------------------
# Ability search
# ---------------------------------------------------------------------------


async def search_ability_tool(
    searchterm: str = "",
    page: int = 0,
) -> dict[str, Any]:
    """Search abilities by name or return a paginated list.

    Ported from TS `searchAbilityTool`.
    """
    try:
        if searchterm.strip():
            raw = await _fetch_json(
                f"{POKEAPI_BASE}/ability/{quote(searchterm.strip().lower())}",
                error_message="Ability not found",
            )
            return {
                "success": True,
                "result": {
                    "results": [_transform_ability_list_item(raw)],
                    "totalPage": 1,
                },
            }

        list_data = await _fetch_json(
            f"{POKEAPI_BASE}/ability/?limit={PAGE_SIZE}&offset={page * PAGE_SIZE}",
            error_message="Failed to fetch ability list",
        )
        total_page = math.ceil(list_data["count"] / PAGE_SIZE)
        urls = [item["url"] for item in list_data.get("results", [])]
        raws = await asyncio.gather(
            *(_fetch_json(url, error_message="Failed to fetch ability list") for url in urls)
        )
        results = [_transform_ability_list_item(r) for r in raws]
        return {"success": True, "result": {"results": results, "totalPage": total_page}}
    except Exception as error:  # noqa: BLE001
        print(f"Error searching Ability: {error}")
        raise


# ---------------------------------------------------------------------------
# Pokémon details
# ---------------------------------------------------------------------------

# Map PokéAPI stat names to camelCase keys used by the UI / downstream prompts.
_STAT_KEY_MAP: dict[str, str] = {
    "hp": "hp",
    "attack": "attack",
    "defense": "defense",
    "special-attack": "specialAttack",
    "special-defense": "specialDefense",
    "speed": "speed",
}


async def fetch_pokemon_details_tool(id: int | str) -> dict[str, Any]:
    """Fetch full Pokémon details for the detail page.

    Mirrors TS `fetchPokemonDetailsTool`. Returns
    `{success: True, result: {id, name, sprite, types, abilities,
    hp/attack/.../speed, height, weight, baseExperience, flavorText, moves}}`.

    All keys are camelCase to match the TS output exactly — the executor's
    extraction prompt expects this shape (see trace event 1084085).
    """
    try:
        client = _get_client()
        pokemon_url = f"{POKEAPI_BASE}/pokemon/{id}"
        species_url = f"{POKEAPI_BASE}/pokemon-species/{id}"
        pokemon_res, species_res = await asyncio.gather(
            client.get(pokemon_url), client.get(species_url)
        )

        if pokemon_res.status_code >= 400:
            raise RuntimeError("Failed to fetch Pokémon details")
        raw = pokemon_res.json()
        species = species_res.json() if species_res.status_code < 400 else None

        # Fetch ability details in parallel for English short_effect.
        abilities_raw = raw.get("abilities") or []
        ability_responses = await asyncio.gather(
            *(client.get(a["ability"]["url"]) for a in abilities_raw)
        )
        ability_details = [resp.json() for resp in ability_responses]

        abilities = [
            {
                "ability_name": a["ability"]["name"],
                "short_effect": _get_english_short_effect(
                    (ability_details[i] or {}).get("effect_entries") or []
                ),
            }
            for i, a in enumerate(abilities_raw)
        ]

        stats: dict[str, int] = {}
        for s in raw.get("stats") or []:
            key = _STAT_KEY_MAP.get(s["stat"]["name"])
            if key:
                stats[key] = s["base_stat"]

        # English flavor text — take the latest entry, replacing form-feed chars
        flavor_text = ""
        if species:
            english_entries = [
                e
                for e in (species.get("flavor_text_entries") or [])
                if e.get("language", {}).get("name") == "en"
            ]
            if english_entries:
                flavor_text = english_entries[-1]["flavor_text"].replace("\f", " ")

        moves: list[dict[str, Any]] = []
        for m in raw.get("moves") or []:
            details = m.get("version_group_details") or []
            latest = details[-1] if details else None
            moves.append(
                {
                    "name": m["move"]["name"],
                    "type": None,  # extra per-move fetch; UI defaults to "Normal"
                    "power": None,
                    "accuracy": None,
                    "move_method": (
                        ((latest or {}).get("move_learn_method") or {}).get("name")
                        or "unknown"
                    ),
                    "level": (latest or {}).get("level_learned_at"),
                }
            )

        sprite = (raw.get("sprites") or {}).get("front_default")
        if not sprite:
            sprite = (
                f"https://raw.githubusercontent.com/PokeAPI/sprites/master/"
                f"sprites/pokemon/{raw['id']}.png"
            )

        result: dict[str, Any] = {
            "id": raw["id"],
            "name": raw["name"],
            "sprite": sprite,
            "types": [t["type"]["name"] for t in (raw.get("types") or [])],
            "abilities": abilities,
            **stats,
            # PokéAPI: height in decimetres → metres; weight in hectograms → kg
            "height": (raw.get("height") or 0) / 10,
            "weight": (raw.get("weight") or 0) / 10,
            "baseExperience": raw.get("base_experience"),
            "flavorText": flavor_text,
            "moves": moves,
        }
        return {"success": True, "result": result}
    except Exception as error:  # noqa: BLE001
        print(f"Error fetching Pokémon details: {error}")
        raise


# ---------------------------------------------------------------------------
# Watchlist stubs (PokéAPI has no user-specific storage)
# ---------------------------------------------------------------------------


async def get_watchlist_tool() -> dict[str, Any]:
    """Stub: returns an empty watchlist."""
    return {"success": True, "result": []}


async def add_to_watchlist_tool(_pokemon_id: int) -> None:
    """Stub: no-op add to watchlist."""
    return None


async def remove_from_watchlist_tool(_pokemon_id: int) -> None:
    """Stub: no-op remove from watchlist."""
    return None


# ---------------------------------------------------------------------------
# Backward-compatible aliases (used in TS by dashboard pages and ed_tools.ts)
# ---------------------------------------------------------------------------

search_pokemon = search_pokemon_tool
fetch_pokemon_details = fetch_pokemon_details_tool
search_move = search_move_tool
search_berry = search_berry_tool
search_ability = search_ability_tool
get_watchlist = get_watchlist_tool
add_to_watchlist = add_to_watchlist_tool
remove_from_watchlist = remove_from_watchlist_tool


__all__ = [
    "POKEAPI_BASE",
    "PAGE_SIZE",
    "search_pokemon_tool",
    "search_pokemon",
    "fetch_pokemon_details_tool",
    "fetch_pokemon_details",
    "search_move_tool",
    "search_move",
    "search_berry_tool",
    "search_berry",
    "search_ability_tool",
    "search_ability",
    "get_watchlist_tool",
    "get_watchlist",
    "add_to_watchlist_tool",
    "add_to_watchlist",
    "remove_from_watchlist_tool",
    "remove_from_watchlist",
]
