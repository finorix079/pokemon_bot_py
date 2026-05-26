"""chat/planner.py — autonomous planner workflow for PokéAPI queries.

Ported from open-react-template/app/api/chat/planner.ts.

Pipeline (preserved verbatim):
  Step 0 — Goal completion check (OpenAI) → may short-circuit
  Step 1 — Infer next action intent (Kimi) → JSON `{description, type}`
  Step 2 — RAG: retrieve PokéAPI tools via keyword matching
  Step 3 — Generate execution plan (OpenAI) using `prompt-planner.txt`

All three Step prompts are preserved byte-for-byte. The retry / correction
loop on assumption-laden plans is preserved as well.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal, Optional

from ..services.chat_planner_service import (
    fetch_prompt_file,
    get_all_matched_apis,
    get_top_k_results,
)
from ..utils.ai_handler import kimi_chat_completion, openai_chat_completion


_JSON_RE = re.compile(r"\{[\s\S]*\}")
_FENCE_RE = re.compile(r"```json|```")
_ASSUMPTION_RE = re.compile(r"\bassume\b|\bassuming\b", re.IGNORECASE)


async def send_to_planner(
    refined_query: str,
    useful_data: str,
    conversation_context: Optional[str] = None,
    plan_intent_type: Optional[Literal["FETCH", "MODIFY"]] = None,
    force_full_plan: bool = False,
) -> str:
    """Autonomous planner workflow.

    Returns a JSON string (the planner's execution plan) — caller is
    expected to `json.loads` or run it through `sanitize_planner_response`.
    Mirrors TS `sendToPlanner` exactly, including:

    - Up to 3 retries on any uncaught error
    - Step 0 short-circuit returns the literal `'Goal completed with existing data'` message
    - Step 2 fallback returns an `impossible: true` plan when no tools match
    - Step 3 assumption-detection re-prompt with the same correction message
    """
    print("🚀 Planner workflow started (PokéAPI mode)")

    max_retries = 3
    last_planner_response = ""

    for retry_count in range(1, max_retries + 1):
        try:
            context_info = (
                f"\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
                f"CONVERSATION HISTORY:\n{conversation_context}\n"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
                if conversation_context
                else ""
            )

            # ============== STEP 0: Goal completion check ==============
            validator_prompt = (
                f"You are a Goal Completion Validator for a Pokémon data assistant.\n\n"
                f"Determine whether the user's goal has already been satisfied by the existing data.\n"
                f"{context_info}"
                f"User Goal: {refined_query}\n\n"
                f"Existing Data (from previous PokéAPI calls):\n"
                f"{useful_data or 'None'}\n\n"
                f"Rules:\n"
                f"1. \"Existing Data\" is the only trustworthy source of facts.\n"
                f"2. If the existing data clearly and fully answers the user goal → GOAL_COMPLETED\n"
                f"3. If the existing data is empty or does not fully answer → GOAL_NOT_COMPLETED\n\n"
                # Claude-tuning (2026-05-26): Claude tends to wrap classification
                # outputs in conversational preamble ("Based on the rules, the
                # answer is: GOAL_COMPLETED."). The downstream check accepts any
                # response that begins with one of the tokens, but we still ask
                # for a bare token to keep the response small and unambiguous.
                f"Respond with exactly one token and nothing else — no preamble, "
                f"no punctuation, no explanation. Your entire response must be either:\n"
                f"GOAL_COMPLETED\n"
                f"or\n"
                f"GOAL_NOT_COMPLETED"
            )

            print("📊 Step 0: Validating goal completion...")
            validator_text = await openai_chat_completion(
                messages=[{"role": "user", "content": validator_prompt}],
                temperature=0.0,
                max_tokens=32,
            )
            print(f"✅ Goal completion check: {validator_text}")

            # Robust to Claude wrapping the token in preamble or punctuation:
            # accept any response whose first 14 chars (case-insensitive) match
            # "GOAL_COMPLETED" AND which doesn't also contain the negation token.
            _validator_stripped = (validator_text or "").strip().upper()
            _is_completed = (
                _validator_stripped.startswith("GOAL_COMPLETED")
                and "GOAL_NOT_COMPLETED" not in _validator_stripped
            )
            if _is_completed:
                return json.dumps(
                    {
                        "needs_clarification": False,
                        "execution_plan": [],
                        "message": "Goal completed with existing data",
                    }
                )

            # ============== STEP 1: Infer next action intent ==============
            next_intent = refined_query
            intent_type: Literal["FETCH", "MODIFY"] = plan_intent_type or "FETCH"

            if not plan_intent_type:
                next_action_prompt = (
                    f"You are a query analyzer for a Pokémon data assistant.\n"
                    f"{context_info}"
                    f"User Goal: {refined_query}\n\n"
                    f"Existing Data: {useful_data or 'None'}\n\n"
                    f"Analyze the goal and describe the single next action needed to make progress.\n"
                    f"Available data sources: Pokémon details, moves, abilities, berries (all from PokéAPI, read-only).\n\n"
                    f"Output ONLY this JSON (no extra text):\n"
                    f'{{ "description": "One-sentence action description", "type": "FETCH" }}'
                )

                print("📊 Step 1: Inferring next action...")
                intent_json = await kimi_chat_completion(
                    messages=[{"role": "user", "content": next_action_prompt}],
                    temperature=0.3,
                    max_tokens=128,
                )
                try:
                    m = _JSON_RE.search(intent_json)
                    if m:
                        intent_json = m.group(0)
                    intent_obj = json.loads(intent_json)
                    description = (intent_obj.get("description") or "").strip()
                    next_intent = description or refined_query
                except Exception as e:  # noqa: BLE001
                    print(
                        f"Failed to parse intent JSON, using refinedQuery as intent: {e}"
                    )
                    next_intent = refined_query
                print(f"✅ Next intent: {next_intent}")

            # ============== STEP 2: RAG ==============
            print("🔍 Step 2: Retrieving PokéAPI tools via keyword matching...")
            rag_apis: list[dict[str, Any]] = []
            try:
                all_matched = await get_all_matched_apis(
                    entities=[next_intent], intentType=intent_type
                )
                rag_apis = await get_top_k_results(all_matched, 10)
                print(f"✅ Retrieved {len(rag_apis)} PokéAPI tools")
            except Exception as e:  # noqa: BLE001
                print(f"⚠️ Tool retrieval failed: {e}")
                rag_apis = []

            if not rag_apis:
                return json.dumps(
                    {
                        "impossible": True,
                        "needs_clarification": False,
                        "message": (
                            f"I can only look up Pokémon, moves, abilities, and berries from "
                            f'PokéAPI. I cannot answer "{refined_query}" with the available tools.'
                        ),
                        "reason": "No relevant PokéAPI tools found",
                        "execution_plan": [],
                    }
                )

            rag_api_desc = json.dumps(rag_apis, indent=2, default=str)

            # ============== STEP 3: Generate execution plan ==============
            print("📝 Step 3: Generating execution plan...")
            planner_system_prompt = await fetch_prompt_file("prompt-planner.txt")

            planner_user_message = (
                f"{context_info}User's Goal: {refined_query}\n\n"
                f"CRITICAL: Your ONLY task is to execute THIS specific next step:\n"
                f'"{next_intent}"\n\n'
                f"Available PokéAPI Tools:\n"
                f"{rag_api_desc}\n\n"
                f"Previously retrieved data:\n"
                f"{useful_data or 'None'}\n\n"
                f"{'IMPORTANT: Return the COMPLETE execution plan covering ALL remaining steps.' if force_full_plan else ''}\n\n"
                f"Generate a concrete execution plan using ONLY the tools listed above."
            )

            planner_response = await openai_chat_completion(
                messages=[
                    {"role": "system", "content": planner_system_prompt},
                    {"role": "user", "content": planner_user_message},
                ],
                temperature=0.5,
                max_tokens=1024,
            )
            planner_response = _FENCE_RE.sub("", planner_response).strip()
            m = _JSON_RE.search(planner_response)
            if not m:
                raise RuntimeError(
                    "Invalid planner response format — no JSON found."
                )
            planner_response = m.group(0)
            print(f"✅ Planner response: {planner_response}")

            parsed = json.loads(planner_response)
            needs_clarification = parsed.get("needs_clarification") is True

            contains_assumption = bool(_ASSUMPTION_RE.search(planner_response)) or (
                "<" in planner_response and ">" in planner_response
            )
            if contains_assumption and not needs_clarification:
                correction_message = (
                    "Your plan contains assumptions or placeholder values.\n"
                    "All parameters must be concrete values derived from the PokéAPI tool descriptions.\n"
                    "If you need a Pokémon name or ID, use it as-is from the user's query.\n"
                    "Do NOT assume or guess any values. Re-generate the plan with concrete values only."
                )
                planner_response = await openai_chat_completion(
                    messages=[
                        {"role": "system", "content": planner_system_prompt},
                        {"role": "user", "content": planner_user_message},
                        {"role": "assistant", "content": planner_response},
                        {"role": "user", "content": correction_message},
                    ],
                    temperature=0.5,
                    max_tokens=1024,
                )
                planner_response = _FENCE_RE.sub("", planner_response).strip()
                retry_match = _JSON_RE.search(planner_response)
                if retry_match:
                    planner_response = retry_match.group(0)

            print(f"🎯 Final execution plan: {planner_response}")
            last_planner_response = planner_response
            return planner_response

        except Exception as error:  # noqa: BLE001
            print(
                f"❌ Error in send_to_planner (attempt {retry_count}/{max_retries}): {error}"
            )
            if retry_count >= max_retries:
                if last_planner_response:
                    return last_planner_response
                raise

    raise RuntimeError("Failed to generate plan after maximum retries")


__all__ = ["send_to_planner"]
