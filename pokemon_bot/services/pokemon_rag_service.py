"""services/pokemon_rag_service.py — static PokéAPI tool catalog for RAG.

Ported from open-react-template/services/pokemonRagService.ts.

The static tool catalog and its keyword-scoring matcher replace the older
embedding-based retrieval (OpenAI ada-002 + cosine similarity over
vectorised ElasticDash schemas). No external API calls needed — the
catalog is plain Python data.

Tool descriptions / tags are preserved verbatim from the TS source so the
planner LLM sees identical content.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PokemonTool:
    """One entry in the static PokéAPI tool catalog."""

    id: str
    """Unique tool identifier — matches the key in `agent_tools`."""

    summary: str
    """Short human-readable description."""

    tags: list[str]
    """Keywords used for relevance scoring."""

    content: str
    """Full tool description (passed to the planner LLM)."""

    similarity: float = 0.0
    """Relevance score computed by `get_matched_pokemon_tools`."""


# Catalog is preserved byte-for-byte from pokemonRagService.ts:24-89.
_POKEAPI_TOOL_CATALOG: list[dict[str, Any]] = [
    {
        "id": "searchPokemon",
        "summary": "Search Pokémon by name or list all Pokémon paginated",
        "tags": [
            "pokemon",
            "search",
            "list",
            "browse",
            "find",
            "show",
            "all",
            "pokedex",
        ],
        "content": """tool: searchPokemon
api: GET /pokemon
summary: Search for a Pokémon by name or browse all Pokémon paginated.
parameters:
  - name (string, optional): Pokémon name (e.g. "pikachu"). Omit to list all.
  - page (number, optional): Page number for pagination (default 1).
returns: { results: [{ id, name, sprite, types }], totalPage }
example: GET /pokemon?name=pikachu""",
    },
    {
        "id": "fetchPokemonDetails",
        "summary": "Fetch complete Pokémon details: stats, moves, abilities, types, species",
        "tags": [
            "pokemon",
            "details",
            "stats",
            "moves",
            "abilities",
            "types",
            "species",
            "hp",
            "attack",
            "defense",
            "speed",
            "height",
            "weight",
            "flavor",
            "info",
        ],
        "content": """tool: fetchPokemonDetails
api: GET /pokemon/{idOrName}
summary: Get full details for a single Pokémon by ID or name.
parameters:
  - idOrName (number|string, required): Pokémon ID number or lowercase name (e.g. 25 or "pikachu").
returns: { id, name, types, stats{ hp, attack, defense, specialAttack, specialDefense, speed }, abilities, moves, height, weight, flavorText }
example: GET /pokemon/pikachu""",
    },
    {
        "id": "searchMove",
        "summary": "Search moves by name or list all moves",
        "tags": [
            "move",
            "attack",
            "skill",
            "power",
            "accuracy",
            "pp",
            "type",
            "damage",
            "technique",
            "learnset",
        ],
        "content": """tool: searchMove
api: GET /move
summary: Search for a move by name or browse all moves paginated.
parameters:
  - name (string, optional): Move name in lowercase-hyphen format (e.g. "thunderbolt", "fire-blast"). Omit to list all.
  - page (number, optional): Page number.
returns: { results: [{ id, name, type, power, accuracy, pp }], totalPage }
example: GET /move?name=thunderbolt""",
    },
    {
        "id": "searchAbility",
        "summary": "Search Pokémon abilities by name or list all abilities",
        "tags": [
            "ability",
            "passive",
            "trait",
            "effect",
            "hidden",
            "characteristic",
            "skill",
            "special",
        ],
        "content": """tool: searchAbility
api: GET /ability
summary: Search for an ability by name or browse all abilities paginated.
parameters:
  - name (string, optional): Ability name (e.g. "levitate", "lightning-rod"). Omit to list all.
  - page (number, optional): Page number.
returns: { results: [{ id, name, shortEffect }], totalPage }
example: GET /ability?name=levitate""",
    },
    {
        "id": "searchBerry",
        "summary": "Search berries by name or list all berries",
        "tags": ["berry", "item", "held", "natural-gift", "growth", "fruit"],
        "content": """tool: searchBerry
api: GET /berry
summary: Search for a berry by name or browse all berries paginated.
parameters:
  - name (string, optional): Berry name (e.g. "cheri", "sitrus"). Omit to list all.
  - page (number, optional): Page number.
returns: { results: [{ id, name, growth_time, natural_gift_power, natural_gift_type, size, smoothness }] }
example: GET /berry?name=cheri""",
    },
]


def get_matched_pokemon_tools(entities: list[str]) -> list[PokemonTool]:
    """Return all PokéAPI tools ranked by keyword relevance to entities.

    Ported from TS `getMatchedPokemonTools`. Preserves the exact scoring
    rules (base 0.5, +0.08 per tag/firstWord match, +0.2 if firstWord is
    in summary, capped at 1.0) so the resulting `topK` order matches.
    """
    entity_text = " ".join(entities).lower()
    parts = entity_text.split()
    first_word = parts[0] if parts else ""

    scored: list[PokemonTool] = []
    for tool_data in _POKEAPI_TOOL_CATALOG:
        score = 0.5
        for tag in tool_data["tags"]:
            if tag in entity_text or (first_word and first_word in tag):
                score += 0.08
        if first_word and first_word in tool_data["summary"].lower():
            score += 0.2
        scored.append(
            PokemonTool(
                id=tool_data["id"],
                summary=tool_data["summary"],
                tags=list(tool_data["tags"]),
                content=tool_data["content"],
                similarity=min(score, 1.0),
            )
        )

    scored.sort(key=lambda t: t.similarity, reverse=True)
    return scored


def get_all_pokemon_tools_map(entities: list[str]) -> dict[str, PokemonTool]:
    """Convenience wrapper: return tools keyed by id (preserves TS API)."""
    return {tool.id: tool for tool in get_matched_pokemon_tools(entities)}


__all__ = [
    "PokemonTool",
    "get_matched_pokemon_tools",
    "get_all_pokemon_tools_map",
]
