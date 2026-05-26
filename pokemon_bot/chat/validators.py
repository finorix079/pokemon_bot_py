"""chat/validators.py — intent classifier + goal-completion validator.

Ported from open-react-template/app/api/chat/validators.ts.

Both system prompts (`detectResolutionVsExecution` and
`validateNeedMoreActions`) are preserved byte-for-byte — they are the
single largest behaviour-anchor in the executor loop.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal, Optional

from ..utils.ai_handler import kimi_chat_completion, openai_chat_completion


# ---------------------------------------------------------------------------
# detect_resolution_vs_execution
# ---------------------------------------------------------------------------


_RESOLUTION_VS_EXECUTION_PROMPT = """You are a query intent classifier.

RESOLUTION queries are those that:
- Check, verify, or confirm the current state
- Ask "has X been done?", "is Y cleared?", "how many Z?"
- Retrieve information to verify a previous action
- Query the database to check current state
- Examples: "Has the watchlist been cleared?", "Did the deletion succeed?", "Show current state", "How many items in my team?"

EXECUTION queries are those that:
- Perform actions or modifications
- Add, delete, update, create data
- Directly call modification APIs
- Examples: "Clear the watchlist", "Delete this item", "Add to team"

Respond with ONLY ONE WORD: either "resolution" or "execution\""""


async def detect_resolution_vs_execution(
    refined_query: str,
    execution_plan: Any,
    api_key: str,
) -> Literal["resolution", "execution"]:
    """Classify a query/plan as a resolution (read-only check) or execution
    (mutation). Mirrors TS `detectResolutionVsExecution`. Defaults to
    `"execution"` on any error — same as the TS catch branch.
    """
    _ = api_key
    try:
        intent = await kimi_chat_completion(
            messages=[
                {"role": "system", "content": _RESOLUTION_VS_EXECUTION_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Query: {refined_query}\n\n"
                        f"Execution Plan: {json.dumps(execution_plan, indent=2, default=str)}\n\n"
                        f"Intent:"
                    ),
                },
            ],
            temperature=0.1,
            max_tokens=10,
        )
        result = (intent or "").strip().lower()
        print(f'🔍 Detected intent: {result} for query: "{refined_query}"')
        if result == "resolution":
            return "resolution"
        return "execution"
    except Exception as error:  # noqa: BLE001
        print(f"Error detecting resolution vs execution: {error}")
        return "execution"


# ---------------------------------------------------------------------------
# validate_need_more_actions
# ---------------------------------------------------------------------------


VALIDATOR_DATA_CHAR_LIMIT = 60_000


def _truncate_for_validator(data: Any) -> str:
    """Serialise + truncate large fields for the validator LLM prompt.

    Mirrors TS `truncateForValidator`. Cap matches the TS source exactly so
    the validator LLM sees the same context window.
    """
    serialised = json.dumps(data, indent=2, default=str)
    if len(serialised) <= VALIDATOR_DATA_CHAR_LIMIT:
        return serialised
    return serialised[:VALIDATOR_DATA_CHAR_LIMIT] + "\n... [truncated for length]"


_VALIDATOR_SYSTEM_PROMPT = """You are the VALIDATOR.

Your ONLY responsibility is to determine whether
the ORIGINAL USER GOAL has been fully satisfied.

You do NOT care whether:
- an API call succeeded
- a step executed without error
- the current execution plan has no remaining steps

You ONLY care about:
→ whether the user's original intent is fulfilled in the current world state.

────────────────────────────────────────
CORE PRINCIPLE (NON-NEGOTIABLE)
────────────────────────────────────────

A successful API call ≠ task completion.

An empty execution plan ≠ task completion.

Only the satisfaction of the ORIGINAL USER GOAL
determines completion.

────────────────────────────────────────
INPUTS YOU WILL RECEIVE
────────────────────────────────────────

You are given:

1. original_user_query (immutable)
2. canonical_user_goal (normalized form, if available)
3. execution_history (all executed API calls + responses)
4. world_state (accumulated facts inferred from execution)
5. last_execution_plan (may be incomplete or incorrect)

You MUST evaluate completion ONLY against (1) or (2).

────────────────────────────────────────
ABSOLUTE RULES
────────────────────────────────────────

1. You MUST NOT infer or invent a new goal.
2. You MUST NOT replace the user goal with a planner step description.
3. You MUST NOT assume the planner plan was complete or correct.
4. You MUST NOT conclude completion solely because:
   - an API returned success
   - data was retrieved
   - no remaining steps exist

If the user goal implies a state change,
you MUST verify that the state change has occurred.

────────────────────────────────────────
GOAL SATISFACTION CHECK (MANDATORY)
────────────────────────────────────────

You MUST answer the following questions IN ORDER:

1. What is the user's original intent?
2. What observable state change or final answer would satisfy it?
3. Does the current world_state conclusively show that state?

If the answer to (3) is NO or UNCERTAIN:
→ the task is NOT complete.

Uncertainty MUST be treated as NOT COMPLETE.

────────────────────────────────────────
COMMON GOAL PATTERNS (GUIDELINES)
────────────────────────────────────────

A) Information retrieval goals
   (e.g. "Which Pokémon has the highest Attack?")
   → Completion requires:
     - a final answer derived from data
     - not just raw data retrieval

B) State-changing goals
   (e.g. "Add Aggron to my watchlist")
   → Completion requires:
     - confirmation that the state changed
     - e.g. POST success AND/OR watchlist contains the ID

C) Multi-step goals
   → Completion requires:
     - ALL required sub-actions completed
     - Partial progress is NOT sufficient

────────────────────────────────────────
CRITICAL: NO RESULTS / NOT FOUND DETECTION
────────────────────────────────────────

If a search/query API call returns:
- Empty array/list (length = 0)
- null result
- "not found" message
- 404 status code
- Error indicating item doesn't exist

AND the user is searching for a specific item by name/identifier:

FIRST, check if there is ANY related data in Accumulated Results:
- If related data exists (e.g., moves for "zygarde" when searching "zygarde-mega")
- If useful information was found with similar identifiers
- If the conversation context referenced a variant that exists

→ DO NOT trigger "item_not_found"
→ USE the related/variant data that was found
→ Conclude: needsMoreActions = false (but with reason explaining the variant was used)

ONLY IF no related data exists at all:
→ The item DOES NOT EXIST in the system
→ DO NOT request more searches with different variations
→ DO NOT say "try a different search endpoint"
→ Conclude: needsMoreActions = false
→ Reason: "The requested item '[name]' was not found in the system after searching"
→ Set "item_not_found": true

────────────────────────────────────────
FORBIDDEN HEURISTICS
────────────────────────────────────────

❌ "The API call succeeded, so we're done"
❌ "There are no remaining steps"
❌ "The planner didn't include more actions"
❌ "The data exists, so the goal must be satisfied"
❌ "Keep searching with different variations" (when item clearly doesn't exist)

────────────────────────────────────────
CRITICAL: COUNT DERIVATION RULE
────────────────────────────────────────

If the goal asks for "count", "how many", "number of", etc.,
and an API endpoint returns a full list/array:

→ Counts MUST be derived by array.length
→ DO NOT request a dedicated count endpoint
→ DO NOT say "we need a count API"

────────────────────────────────────────
OUTPUT FORMAT (JSON ONLY)
────────────────────────────────────────

If the goal IS satisfied:

{
  "needsMoreActions": false,
  "reason": "Clear explanation of how the original user goal has been fully satisfied based on world state"
}

If the goal is NOT satisfied:

{
  "needsMoreActions": true,
  "reason": "What part of the original user goal is still unmet",
  "missing_requirements": [
    "Explicit unmet condition 1",
    "Explicit unmet condition 2"
  ],
  "suggested_next_action": "High-level description of what must happen next (NOT a full plan)"
}

If the requested item/entity DOES NOT EXIST (after search returned empty/null/404):

{
  "needsMoreActions": false,
  "reason": "The requested item '[name]' does not exist in the system. Search returned no results.",
  "item_not_found": true
}

────────────────────────────────────────
FINAL OVERRIDE RULE
────────────────────────────────────────

If you are unsure whether the user goal has been met,
you MUST respond with needsMoreActions = true.

False negatives are acceptable.
False positives are NOT."""


_JSON_FENCE_RE = re.compile(r"```json|```")
_JSON_OBJECT_RE = re.compile(r"\{[\s\S]*\}")


async def validate_need_more_actions(
    original_query: str,
    executed_steps: list[Any],
    accumulated_results: list[Any],
    api_key: str,
    last_execution_plan: Optional[Any] = None,
) -> dict[str, Any]:
    """Decide whether more API actions are needed to fulfil the user goal.

    Mirrors TS `validateNeedMoreActions`. Returns a dict with at least
    `needsMoreActions` and `reason`; optionally `missing_requirements`,
    `suggested_next_action`, `useful_data`, `item_not_found`.

    Both the system prompt and the user prompt assembly are preserved
    byte-for-byte from the TS source.
    """
    _ = api_key
    try:
        if last_execution_plan and isinstance(last_execution_plan, dict):
            plan_payload = last_execution_plan.get("execution_plan", last_execution_plan)
            selected_tools_spec = last_execution_plan.get("selected_tools_spec")
        else:
            plan_payload = last_execution_plan
            selected_tools_spec = None

        plan_str = (
            json.dumps(plan_payload, indent=2, default=str)
            if plan_payload is not None
            else "No plan available"
        )
        tools_block = ""
        if selected_tools_spec is not None:
            tools_block = (
                "\nAvailable Tools (used in plan):\n"
                f"{json.dumps(selected_tools_spec, indent=2, default=str)}\n\n"
                "These tools show what capabilities are available. If a tool returns an array,\n"
                "counts can be derived via array.length. DO NOT request count endpoints.\n"
            )

        user_prompt = (
            f"Original Query: {original_query}\n\n"
            f"Last Execution Plan: {plan_str}\n\n"
            f"{tools_block}\n"
            f"Executed Steps (with responses): {_truncate_for_validator(executed_steps)}\n\n"
            f"Accumulated Results: {_truncate_for_validator(accumulated_results)}\n\n"
            "IMPORTANT:\n"
            "1. Check if the last execution plan had multiple steps (e.g., fetching data for multiple IDs)\n"
            "2. Verify if ALL required IDs/entities have been fetched\n"
            "3. Review the \"Available Tools\" to see what derivations are possible (e.g., counts from array.length)\n"
            "4. Only request more actions if there are genuinely missing IDs or the goal is incomplete\n"
            "5. DO NOT request count/aggregation endpoints if arrays are already available\n\n"
            "Can we answer the original query with the information we have? Or do we need more API calls?"
        )

        content = await openai_chat_completion(
            messages=[
                {"role": "system", "content": _VALIDATOR_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=4096,
        )
        print(f"Validator Response 2: {content}")

        sanitised = _JSON_FENCE_RE.sub("", content).strip()
        m = _JSON_OBJECT_RE.search(sanitised)
        if m:
            result = json.loads(m.group(0))
            print(f"Validator Decision: {result}")
            return result

        return {
            "needsMoreActions": False,
            "reason": "Unable to parse validator response",
        }
    except Exception as error:  # noqa: BLE001
        print(f"Error in validator: {error}")
        return {
            "needsMoreActions": False,
            "reason": "Validator error, proceeding with available data",
        }


__all__ = [
    "VALIDATOR_DATA_CHAR_LIMIT",
    "detect_resolution_vs_execution",
    "validate_need_more_actions",
]
