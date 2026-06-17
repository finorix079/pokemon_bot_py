"""Thin re-export layer so the elasticdash CLI can discover workflows.

The ElasticDash Python SDK's `elasticdash run-workflow <name>` command
loads `ed_workflows.py` from the project root (see cli.py:_load_config).
This file exposes a small partial-mocking smoke test that exercises a
real wrapped tool — useful for validating the AH-280 rerun_workflow_mocked
feature against this project without standing up the full chat pipeline.

For the full Pokémon chat pipeline, see `pokemon_bot/chat/handler.py`.
"""

from typing import Any

from pokemon_bot.tools.pokemon_tools import searchPokemon, fetchPokemonDetails


async def pokemonLookupFlow(input: dict[str, Any] | None = None) -> dict[str, Any]:
    """Two-tool workflow: list, then fetch details.

    Both tools are registered via `@ed_tool(name=...)` so they participate
    in mock dispatch under `elasticdash run-workflow ... --mock-config-file`.
    """
    name = (input or {}).get("name", "pikachu")
    listing = await searchPokemon({"searchterm": name, "page": 0})
    detail_id = (input or {}).get("id", 25)
    detail = await fetchPokemonDetails({"id": detail_id})
    return {"listing": listing, "detail": detail}


async def pokemon_chat(input: dict[str, Any] | None = None) -> dict[str, Any]:
    """Full Pokémon chat pipeline workflow.

    Wraps `run_chat_pipeline` so the elasticdash CLI can invoke it via
    `run-workflow pokemon_chat`. The streaming callbacks are routed to
    in-memory buffers; the returned message contains the final synthesised
    answer (same one that would be streamed to the REPL terminal).
    """
    from pokemon_bot.chat.handler import run_chat_pipeline
    from pokemon_bot.schemas.ai import ChatStreamRequest

    payload = input or {}
    request = ChatStreamRequest(
        messages=payload.get("messages", []),
        sessionId=payload.get("sessionId"),
    )

    tokens: list[str] = []
    statuses: list[str] = []

    def _on_status(msg: str) -> None:
        statuses.append(msg)

    def _on_token(tok: str) -> None:
        tokens.append(tok)

    result = await run_chat_pipeline(
        request,
        user_token=payload.get("userToken", ""),
        on_status=_on_status,
        on_token=_on_token,
        on_result=_on_token,
        on_error=_on_token,
    )
    return {
        "message": result.get("message", "".join(tokens)),
        "refinedQuery": result.get("refinedQuery"),
        "sessionId": result.get("sessionId"),
        "messageId": result.get("messageId"),
        "statuses": statuses,
    }
