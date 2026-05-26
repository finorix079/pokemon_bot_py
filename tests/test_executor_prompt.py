"""Prompt-preservation tests for the executor.

The user explicitly required that all current LLM prompts be carried over
byte-for-byte from the TypeScript source. These tests anchor that
guarantee for the two highest-risk prompts in the codebase:

1. The Kimi extraction system instruction (executor.ts:114-147)
2. The OpenAI final-answer synthesis system prompt (executor.ts:183-185)
3. The "IMPORTANT: …" user-message block (executor.ts:237-243) that
   contains the known extra-stats bug — its preservation is intentional.
"""

from __future__ import annotations

import asyncio

import pytest

from pokemon_bot.chat import executor


def test_extraction_system_instruction_contains_critical_rules() -> None:
    s = executor._EXTRACTION_SYSTEM_INSTRUCTION
    # Anchors from the TS source
    assert s.startswith("You are an expert at extracting useful information")
    assert "CRITICAL RULES:" in s
    assert "Keep the output CONCISE - include only what was asked by the user" in s
    assert "FIELD PRESERVATION RULES (CRITICAL):" in s
    assert "FACTUAL REPORTING ONLY:" in s
    assert "If no new useful data is found, return the existing useful data as is." in s


def test_final_answer_system_prompt_byte_identical() -> None:
    expected = (
        "You are a helpful assistant that synthesizes information from API responses to answer user questions.\n"
        "Provide a clear, concise, and well-formatted answer based on the accumulated data.\n"
        "Use the actual data from the API responses to provide specific, accurate information."
    )
    assert executor._FINAL_ANSWER_SYSTEM_PROMPT == expected


def test_generate_final_answer_user_content_preserves_important_block(
    llm_recorder,
) -> None:
    """The `IMPORTANT: …` user-message block must be present verbatim.

    This is the load-bearing text that causes the executor synthesis LLM
    to enumerate extra stats — preserving it is the explicit requirement
    of this port.
    """

    async def go() -> None:
        await executor.generate_final_answer(
            "What is the speed stat of Raichu?",
            [{"step": 1, "response": {"hp": 60, "speed": 110}}],
            "api-key-unused",
        )

    asyncio.run(go())
    openai_calls = [c for c in llm_recorder.calls if c["name"] == "openai"]
    assert len(openai_calls) == 1
    user_msg = openai_calls[0]["kwargs"]["messages"][-1]["content"]
    for anchor in (
        "Original Question: What is the speed stat of Raichu?",
        "API Response Data:",
        "IMPORTANT: The data above includes complete arrays. Pay careful attention to:",
        "- Learning methods for moves (level-up, tutor, machine, egg, etc.)",
        "- Type information for moves",
        "- Power values for moves",
        "- Any other detailed attributes",
        "Only state facts that are explicitly present in the data.",
    ):
        assert anchor in user_msg, f"missing anchor: {anchor!r}"


def test_generate_final_answer_temperature_unchanged(llm_recorder) -> None:
    """The known-buggy temperature of 0.7 must be preserved."""

    async def go() -> None:
        await executor.generate_final_answer("q", [], "")

    asyncio.run(go())
    openai_calls = [c for c in llm_recorder.calls if c["name"] == "openai"]
    assert openai_calls[0]["kwargs"]["temperature"] == 0.7
    assert openai_calls[0]["kwargs"]["max_tokens"] == 2048


@pytest.mark.parametrize(
    "stopped_reason,expected_prefix",
    [
        ("max_iterations", "Sorry, I was unable to gather enough information"),
        ("stuck_state", "It seems that the information you're looking for"),
    ],
)
def test_generate_final_answer_stop_reasons(
    stopped_reason: str, expected_prefix: str
) -> None:
    """`max_iterations` and `stuck_state` short-circuit to canned responses."""

    async def go() -> str:
        return await executor.generate_final_answer(
            "q", [], "", stopped_reason=stopped_reason
        )

    answer = asyncio.run(go())
    assert answer.startswith(expected_prefix)


def test_resolve_poke_api_tool_path_mapping() -> None:
    """`/pokemon/{id}` → fetchPokemonDetails; `/pokemon` → searchPokemon."""
    assert executor.resolve_poke_api_tool("/pokemon/raichu", {}) == {
        "toolName": "fetchPokemonDetails",
        "input": {"id": "raichu"},
    }
    assert executor.resolve_poke_api_tool("/pokemon/25", {}) == {
        "toolName": "fetchPokemonDetails",
        "input": {"id": 25},
    }
    assert executor.resolve_poke_api_tool("/pokemon", {"name": "pikachu"}) == {
        "toolName": "searchPokemon",
        "input": {"searchterm": "pikachu", "page": 0},
    }
    assert executor.resolve_poke_api_tool("/berry/cheri", {}) == {
        "toolName": "searchBerry",
        "input": {"query": "cheri", "page": 0},
    }
    assert executor.resolve_poke_api_tool("/unknown", {}) is None
