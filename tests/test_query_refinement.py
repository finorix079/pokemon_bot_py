"""Tests for `clarify_and_refine_user_input` happy + degraded paths.

The regex parser is preserved verbatim from the TS source; these tests
anchor:

1. Happy path: full response → all six fields parsed correctly.
2. Degraded path: empty / malformed response → documented fallback values
   (refined query = original input, entities = [input], intentType = FETCH,
   etc.) — same behaviour as the TS code.
3. Conversation context detection: `Previous context:\n…\n\nCurrent query: …`
   is split out into HISTORICAL CONTEXT / CURRENT QUERY blocks before
   reaching the LLM.
"""

from __future__ import annotations

import asyncio

import pytest

from pokemon_bot.utils.query_refinement import (
    SYSTEM_PROMPT,
    clarify_and_refine_user_input,
    handle_query_concepts_and_needs,
)


_HAPPY_RESPONSE = """Refined Query: What is the speed stat of Raichu?
Language: EN
Concepts: [pokemon stats, speed stat, Raichu]
API Needs: [pokemon details retrieval, stat information]
Entities: ["raichu", "pokemon base stats", "speed stat value"]
IntentType: FETCH"""


def test_happy_path_parses_all_fields(llm_recorder) -> None:
    llm_recorder.set_response("openai", _HAPPY_RESPONSE)

    result = asyncio.run(clarify_and_refine_user_input("What's the speed of raichu?"))

    assert result["refinedQuery"] == "What is the speed stat of Raichu?"
    assert result["language"] == "EN"
    assert result["concepts"] == ["pokemon stats", "speed stat", "Raichu"]
    assert result["apiNeeds"] == ["pokemon details retrieval", "stat information"]
    assert result["entities"] == [
        "raichu",
        "pokemon base stats",
        "speed stat value",
    ]
    assert result["intentType"] == "FETCH"
    assert result["referenceTask"] is None


def test_degraded_path_falls_back(llm_recorder) -> None:
    """Empty / malformed LLM output → all documented fallbacks fire."""
    llm_recorder.set_response("openai", "garbage with no recognised structure")

    result = asyncio.run(clarify_and_refine_user_input("hello"))

    assert result["refinedQuery"] == "hello"
    assert result["language"] == "EN"
    assert result["concepts"] == []
    assert result["apiNeeds"] == []
    assert result["entities"] == ["hello"]  # fallback = [user_input]
    assert result["intentType"] == "FETCH"


def test_intent_type_modify_recognised(llm_recorder) -> None:
    response = """Refined Query: Add Pikachu to my watchlist
Language: EN
Concepts: [add]
API Needs: [watchlist]
Entities: ["pikachu"]
IntentType: MODIFY"""
    llm_recorder.set_response("openai", response)

    result = asyncio.run(clarify_and_refine_user_input("add pikachu to my watchlist"))
    assert result["intentType"] == "MODIFY"


def test_conversation_context_splits_into_historical_block(llm_recorder) -> None:
    """When the input matches `Previous context:\\n…\\n\\nCurrent query: …`,
    the user message sent to the LLM must contain the HISTORICAL CONTEXT and
    CURRENT QUERY (PRIMARY FOCUS) headers."""
    llm_recorder.set_response("openai", _HAPPY_RESPONSE)
    asyncio.run(
        clarify_and_refine_user_input(
            "Previous context:\nuser: about pikachu\nassistant: details\n\n"
            "Current query: show me its moves"
        )
    )
    sent_user_msg = llm_recorder.calls[-1]["kwargs"]["messages"][1]["content"]
    assert "HISTORICAL CONTEXT (for reference only):" in sent_user_msg
    assert "CURRENT QUERY (PRIMARY FOCUS):" in sent_user_msg
    assert "show me its moves" in sent_user_msg


def test_system_prompt_byte_identical_anchor() -> None:
    """Sanity anchor: ensure the byte-identical SYSTEM_PROMPT we verified
    in Step 4 hasn't drifted."""
    assert SYSTEM_PROMPT.startswith(
        "You are an expert that refines user queries and identifies what needs to be investigated"
    )
    assert "CRITICAL - DATABASE RULES:" in SYSTEM_PROMPT
    assert "CRITICAL - FOLLOW-UP QUERIES WITH \"AMONG THEM\"" in SYSTEM_PROMPT
    assert 'IntentType: ["FETCH"/"MODIFY"]' in SYSTEM_PROMPT


def test_handle_query_concepts_and_needs_skips_redundant_sort() -> None:
    out = handle_query_concepts_and_needs(["ATK"], ["sortByATK", "searchPokemon"])
    assert out == {"requiredApis": ["searchPokemon"], "skippedApis": ["sortByATK"]}

    out2 = handle_query_concepts_and_needs([], ["sortByATK"])
    assert out2 == {"requiredApis": ["sortByATK"], "skippedApis": []}
