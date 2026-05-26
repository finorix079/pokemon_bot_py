"""chat/executor.py — extraction, final-answer synthesis, iterative loop.

Ported from open-react-template/app/api/chat/executor.ts (1117 lines).

Key contracts preserved verbatim:
- `extract_useful_data_from_api_responses` uses Kimi with the same system
  instruction the TS code uses (including the recent fix where the prompt
  was relaxed to "include only what was asked by the user unless
  absolutely necessary otherwise").
- `generate_final_answer` uses OpenAI/Anthropic dispatch with the same
  system prompt and the same `IMPORTANT: …` user-message block —
  **including the known extra-stats bug** in `executor.ts:183-243` that
  was root-caused in the previous task. The user instructed us to
  preserve all prompts verbatim, so it ships as-is.
- `execute_iterative_planner` mirrors the 776-line TS function: FETCH /
  MODIFY split, fan-out support, depends_on_step propagation, validator
  re-plan loops, stuck-state detection, max-iteration / max-planning-cycle
  caps, and the moves-filtering pre-final-answer pass.

Adaptations:
- Langfuse `parentSpan` / `LangfuseObservation` arguments are dropped;
  the Python port uses the named agent tool registry directly without
  observation parenting (the same code path the TS uses when no parent
  span is supplied).
- The `NextResponse.json(...)` early-return for placeholder-resolution
  failure is replaced by a plain dict return — callers were checking the
  payload, not the response object.
- Axios-specific sanitiser fields (`request`, `socket`, `agent`, `res`,
  AxiosHeaders) are kept in the sanitiser even though httpx doesn't emit
  them, so any hand-built error dict that contains them still gets stripped.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Awaitable, Callable, Literal, Optional

from ..services.api_schema_loader import find_api_parameters
from ..services.chat_planner_service import (
    RequestContext,
    UsefulDataEntry,
    get_all_matched_apis,
    get_top_k_results,
    serialize_useful_data_in_order,
)
from ..services.task_service import SavedTask
from ..tools.pokemon_tools import apiService as _apiServiceTool, maybe_await
from ..utils.ai_handler import (
    get_agent_tools,
    kimi_chat_completion,
    openai_chat_completion,
)
from .planner import send_to_planner
from .planner_utils import (
    contains_placeholder_reference,
    resolve_placeholders,
    sanitize_planner_response,
)
from .validators import validate_need_more_actions


# ---------------------------------------------------------------------------
# Tool resolution from URL path
# ---------------------------------------------------------------------------


def resolve_poke_api_tool(
    api_path: str, parameters: dict[str, Any]
) -> Optional[dict[str, Any]]:
    """Map a planner-generated `path` + `parameters` to a named PokéAPI tool.

    Ported from TS `resolvePokeApiTool`. Returns `{toolName, input}` or
    `None` for unrecognised paths.

    Mapping rules (preserved):
      `/pokemon/{name}` → `fetchPokemonDetails`  `{id}`
      `/pokemon`        → `searchPokemon`        `{searchterm, page}`
      `/move/{name}`    → `searchMove`           `{searchterm, page}`
      `/move`           → `searchMove`           `{searchterm, page}`
      `/ability/{name}` → `searchAbility`        `{searchterm, page}`
      `/ability`        → `searchAbility`        `{searchterm, page}`
      `/berry/{name}`   → `searchBerry`          `{query, page}`
      `/berry`          → `searchBerry`          `{query, page}`
    """
    clean_path = (api_path or "").rstrip("/")
    segments = [s for s in clean_path.split("/") if s]
    resource = segments[0] if segments else ""
    name_or_id = segments[1] if len(segments) >= 2 else None
    try:
        page = int(parameters.get("page", 0)) if isinstance(parameters, dict) else 0
    except (TypeError, ValueError):
        page = 0

    if resource == "pokemon" and name_or_id:
        try:
            parsed: Any = int(name_or_id)
        except ValueError:
            parsed = name_or_id
        return {"toolName": "fetchPokemonDetails", "input": {"id": parsed}}
    if resource == "pokemon":
        searchterm = str(
            (parameters or {}).get("searchterm") or (parameters or {}).get("name") or ""
        )
        return {"toolName": "searchPokemon", "input": {"searchterm": searchterm, "page": page}}
    if resource == "move":
        searchterm = name_or_id or str(
            (parameters or {}).get("searchterm") or (parameters or {}).get("name") or ""
        )
        return {"toolName": "searchMove", "input": {"searchterm": searchterm, "page": page}}
    if resource == "ability":
        searchterm = name_or_id or str(
            (parameters or {}).get("searchterm") or (parameters or {}).get("name") or ""
        )
        return {"toolName": "searchAbility", "input": {"searchterm": searchterm, "page": page}}
    if resource == "berry":
        query = name_or_id or str(
            (parameters or {}).get("query") or (parameters or {}).get("name") or ""
        )
        return {"toolName": "searchBerry", "input": {"query": query, "page": page}}
    return None


# ---------------------------------------------------------------------------
# Kimi extraction prompt (preserved verbatim from executor.ts:114-147)
# ---------------------------------------------------------------------------


_EXTRACTION_SYSTEM_INSTRUCTION = """You are an expert at extracting useful information from API responses to help answer user queries.

Given the original user query, the refined query, and the final deliverable generated so far,
extract any useful data points, facts, or details from the API responses that could aid in answering the user's question.

CRITICAL RULES:
1. If the new API response contains UPDATED or MORE ACCURATE information, REPLACE the old data
2. Only keep UNIQUE and NON-REDUNDANT information
3. Remove any duplicate or outdated facts
4. Keep the output CONCISE - include only what was asked by the user unless absolutely necessary otherwise
5. If it contains things like ID, deleted, or other important data, make sure to include those

FIELD PRESERVATION RULES (CRITICAL):
- ALWAYS preserve ALL ID fields (id, pokemon_id, user_id, team_id, etc.) - these are often required for subsequent API calls
- ALWAYS preserve foreign key relationships (e.g., if an item has both "id" and "pokemon_id", keep BOTH)
- ALWAYS preserve status fields (deleted, active, success, etc.)
- ALWAYS preserve timestamps (created_at, updated_at, etc.) if they might be relevant
- When in doubt, KEEP the field rather than removing it
- Check the available APIs context to see if any downstream APIs might need specific fields

FACTUAL REPORTING ONLY:
- Report ONLY what the API response explicitly states (e.g., "3 items were deleted", "ID 123 was created")
- DO NOT infer or state goal completion (e.g., NEVER say "watchlist has been cleared", "task completed", "goal achieved")
- DO NOT interpret the action's success in terms of user goals
- State facts like: "deletedCount: 3", "success: true", "ID: 456", "pokemon_id: 789"
- Let the validator and final answer generator determine if the goal is met

FORMAT:
Structure the extracted data to preserve relationships. For list responses, maintain the structure:
- If response contains an array of objects, preserve key fields from each object
- For single objects, preserve all important fields
- Use clear labels to indicate what each piece of data represents

If no new useful data is found, return the existing useful data as is."""


async def extract_useful_data_from_api_responses(
    refined_query: str,
    final_deliverable: str,
    existing_useful_data: str,
    api_response: str,
    api_schema: Optional[dict[str, Any]] = None,
    available_apis: Optional[list[dict[str, Any]]] = None,
) -> str:
    """Extract + merge useful data from a single API response.

    Ported from TS `extractUsefulDataFromApiResponses`. The system
    instruction and user message format are preserved verbatim, including
    the conditionally-attached `schemaContext` and `availableApisContext`
    blocks.
    """
    try:
        schema_context = ""
        if api_schema:
            schema_context = (
                "\n\nAPI Schema Context (endpoint that was just called):\n"
                f"Path: {api_schema.get('path')}\n"
                f"Method: {api_schema.get('method')}\n"
                f"Request Body: {json.dumps(api_schema.get('requestBody') or {}, indent=2, default=str)}\n"
                f"Parameters: {json.dumps(api_schema.get('parameters') or {}, indent=2, default=str)}"
            )

        available_apis_context = ""
        if available_apis:
            summaries: list[str] = []
            for api in available_apis[:10]:
                try:
                    content = api.get("content")
                    if not isinstance(content, str):
                        content = json.dumps(content, default=str)
                    summaries.append(
                        f"- {api.get('id')}: {api.get('summary') or 'No summary'}\n"
                        f"  {content[:200]}..."
                    )
                except Exception:  # noqa: BLE001
                    summaries.append(
                        f"- {api.get('id')}: {api.get('summary') or 'No summary'}"
                    )
            available_apis_context = (
                "\n\nAvailable APIs (for understanding data dependencies):\n"
                + "\n".join(summaries)
                + "\n\nCRITICAL: Check if any downstream APIs might need fields from the "
                "current response.\nFor example, if a \"delete watchlist\" API requires "
                '"pokemon_id", then pokemon_id must be preserved from the "get watchlist" '
                "response."
            )

        user_message = (
            f"Refined User Query: {refined_query}\n"
            f"Final Deliverable: {final_deliverable}\n"
            f"Existing Useful Data: {existing_useful_data}\n"
            f"API Response: {api_response}{schema_context}{available_apis_context}\n\n"
            f"Extracted Useful Data: "
        )

        extracted = await kimi_chat_completion(
            messages=[
                {"role": "system", "content": _EXTRACTION_SYSTEM_INSTRUCTION},
                {"role": "user", "content": user_message},
            ],
            temperature=0.5,
            max_tokens=4096,
        )
        return extracted
    except Exception as error:  # noqa: BLE001
        print(f"Error extracting useful data: {error}")
        return existing_useful_data


# ---------------------------------------------------------------------------
# Final-answer synthesis (NOTE: contains the known extra-stats bug)
# ---------------------------------------------------------------------------


_FINAL_ANSWER_SYSTEM_PROMPT = (
    "You are a helpful assistant that synthesizes information from API responses to answer user questions.\n"
    "Provide a clear, concise, and well-formatted answer based on the accumulated data.\n"
    "Use the actual data from the API responses to provide specific, accurate information."
)


async def generate_final_answer(
    original_query: str,
    accumulated_results: list[Any],
    api_key: str,
    stopped_reason: Optional[str] = None,
    useful_data: Optional[str] = None,
) -> str:
    """Generate a final natural-language answer from accumulated API results.

    Ported from TS `generateFinalAnswer`. Special stop reasons short-circuit
    to canned responses (`max_iterations`, `stuck_state`, `item_not_found`).

    NOTE: The `IMPORTANT: …` user-message block (executor.ts:237-243) and
    the 0.7 temperature are preserved verbatim, which is the root cause of
    the extra-stats bug investigated in the previous task. The user
    explicitly asked for all prompts to be carried over — so it ships
    unchanged.
    """
    _ = api_key
    try:
        additional_context = ""

        if stopped_reason == "max_iterations":
            return (
                "Sorry, I was unable to gather enough information to provide a "
                "complete answer within the allowed steps."
            )
        if stopped_reason == "stuck_state":
            return (
                "It seems that the information you're looking for may not be "
                "available through the current APIs. If you have more specific "
                "details or another question, feel free to ask!"
            )
        if stopped_reason == "item_not_found":
            searched_item = ""
            try:
                for result in accumulated_results:
                    response = result.get("response") if isinstance(result, dict) else None
                    if not response:
                        continue
                    is_empty_array = isinstance(response, list) and len(response) == 0
                    is_null_result = (
                        isinstance(response, dict) and response.get("result") is None
                    )
                    is_empty_results = (
                        isinstance(response, dict)
                        and isinstance(response.get("results"), list)
                        and len(response["results"]) == 0
                    )
                    msg = (
                        response.get("message", "").lower()
                        if isinstance(response, dict) and isinstance(response.get("message"), str)
                        else ""
                    )
                    if (
                        is_empty_array
                        or is_null_result
                        or is_empty_results
                        or "not found" in msg
                    ):
                        step_field = (
                            result.get("step") if isinstance(result, dict) else None
                        )
                        desc_field = (
                            result.get("description") if isinstance(result, dict) else None
                        )
                        searched_item = str(step_field or desc_field or "")
                        break
            except Exception as e:  # noqa: BLE001
                print(f"Could not extract searched item: {e}")
            suffix = f" ({searched_item})" if searched_item else ""
            return (
                f"I couldn't find the item you're looking for{suffix} in the system. "
                "The search returned no results. Please check the spelling or try a "
                "different search term."
            )

        accumulated_json = json.dumps(accumulated_results, indent=2, default=str)
        # NOTE: This block is the known extra-stats bug — preserved verbatim.
        user_content = (
            f"Original Question: {original_query}\n\n"
            f"API Response Data:\n"
            f"{accumulated_json}{useful_data or ''}\n\n"
            "IMPORTANT: The data above includes complete arrays. Pay careful attention to:\n"
            "- Learning methods for moves (level-up, tutor, machine, egg, etc.)\n"
            "- Type information for moves\n"
            "- Power values for moves\n"
            "- Any other detailed attributes\n\n"
            "Only state facts that are explicitly present in the data. "
            "Do not make assumptions about learning methods or other attributes."
        )

        message = await openai_chat_completion(
            messages=[
                {"role": "system", "content": _FINAL_ANSWER_SYSTEM_PROMPT + additional_context},
                {"role": "user", "content": user_content},
            ],
            temperature=0.7,
            max_tokens=2048,
        )
        return message or "Unable to generate answer."
    except Exception as error:  # noqa: BLE001
        print(f"Error generating final answer: {error}")
        return "Error generating answer from the gathered information."


# ---------------------------------------------------------------------------
# Sanitiser helpers (replace TS WeakSet-based circular-ref strippers)
# ---------------------------------------------------------------------------


_OMIT_KEYS = {"request", "socket", "agent", "res"}


def _sanitize_for_serialization(obj: Any) -> Any:
    """Strip circular refs + axios-specific keys.

    Mirrors `sanitizeForSerialization` in executor.ts. The TS code uses
    `WeakSet` to detect circular refs; here we track visited ids. Axios
    headers / config / request etc. only appear if the upstream code hand-
    rolls error responses with those keys — preserving the strip is
    defensive parity with TS.
    """
    seen: set[int] = set()

    def walk(value: Any, parent_key: str = "") -> Any:
        if isinstance(value, dict):
            if id(value) in seen:
                return "[Circular]"
            seen.add(id(value))
            out: dict[str, Any] = {}
            for k, v in value.items():
                if k in _OMIT_KEYS:
                    out[k] = "[Omitted]"
                    continue
                if k == "config" and isinstance(v, dict):
                    out[k] = {"method": v.get("method"), "url": v.get("url"), "data": v.get("data")}
                    continue
                if k == "headers" and getattr(v, "__class__", type(None)).__name__ == "AxiosHeaders":
                    out[k] = dict(v)
                    continue
                out[k] = walk(v, k)
            return out
        if isinstance(value, list):
            if id(value) in seen:
                return "[Circular]"
            seen.add(id(value))
            return [walk(item, parent_key) for item in value]
        return value

    # Round-trip through JSON to match the TS behaviour exactly
    return json.loads(json.dumps(walk(obj), default=str))


def _get_api_call_key(
    path: str, method: str, params: Any, body: Any
) -> str:
    """Stable cache key for an API call. Mirrors TS `getApiCallKey`."""

    def stable(obj: Any) -> str:
        if obj is None or not isinstance(obj, (dict, list)):
            return str(obj)
        if isinstance(obj, list):
            return "[" + ",".join(stable(x) for x in obj) + "]"
        items = sorted(obj.items())
        return "{" + ",".join(f'"{k}":{stable(v)}' for k, v in items) + "}"

    combined: dict[str, Any] = {}
    if isinstance(params, dict):
        combined.update(params)
    if isinstance(body, dict):
        combined["_body"] = body
    return f"{method.lower()} {path}::{stable(combined)}"


# ---------------------------------------------------------------------------
# Move-list pre-filtering for the final answer
# ---------------------------------------------------------------------------


_TYPE_KEYWORDS: tuple[str, ...] = (
    "steel", "fire", "water", "electric", "grass", "ice", "fighting", "poison",
    "ground", "flying", "psychic", "bug", "rock", "ghost", "dragon", "dark",
    "fairy", "normal",
)


def _prepare_results_for_final_answer(
    refined_query: str, accumulated_results: list[Any]
) -> list[Any]:
    """Filter `moves` arrays by query-mentioned type.

    Mirrors the post-loop `preparedResults` block in executor.ts:1047-1094.
    For any result with `response.result.moves` of length > 10 the moves
    are filtered to the type mentioned in `refined_query`, with
    `movesCount` / `filteredMovesCount` / `movesNote` annotations added.
    """
    query_lower = refined_query.lower()
    mentioned_type = next((t for t in _TYPE_KEYWORDS if t in query_lower), None)

    prepared: list[Any] = []
    for result in accumulated_results:
        if not isinstance(result, dict):
            prepared.append(result)
            continue
        response = result.get("response")
        result_data = response.get("result") if isinstance(response, dict) else None
        moves = result_data.get("moves") if isinstance(result_data, dict) else None
        if not isinstance(moves, list) or len(moves) <= 10:
            prepared.append(result)
            continue
        print(f"Processing {len(moves)} moves for final answer")
        filtered = moves
        if mentioned_type:
            relevant = [
                m
                for m in moves
                if (m.get("type_name") or "").lower() == mentioned_type
                or (m.get("typeName") or "").lower() == mentioned_type
            ]
            print(f"Filtered to {len(relevant)} {mentioned_type}-type moves")
            if relevant:
                filtered = relevant

        new_result = dict(result)
        new_response = dict(response)
        new_response["result"] = {
            **result_data,
            "moves": filtered,
            "movesCount": len(moves),
            "filteredMovesCount": len(filtered),
            "movesNote": (
                f"Filtered to {len(filtered)} {mentioned_type}-type moves out of {len(moves)} total"
                if mentioned_type
                else f"All {len(filtered)} moves included"
            ),
        }
        new_result["response"] = new_response
        prepared.append(new_result)
    return prepared


# ---------------------------------------------------------------------------
# Iterative planner / executor
# ---------------------------------------------------------------------------


_PATH_PARAM_RE = re.compile(r"\{(\w+)\}")


async def _call_api(
    api_path: str,
    parameters: dict[str, Any],
    api_schema: dict[str, Any],
    user_token: str,
) -> Any:
    """Dispatch one API call.

    If `resolve_poke_api_tool` can map the path to a named tool, route via
    the agent_tools registry (mirrors the TS `parentSpan && agentTools[...]`
    branch, minus the Langfuse parent). Otherwise fall back to the generic
    `apiService` tool — same fallback as TS.
    """
    resolved = resolve_poke_api_tool(api_path, parameters)
    if resolved is not None:
        tools = get_agent_tools()
        tool = tools.get(resolved["toolName"])
        if tool is not None:
            print(
                f'📡 Calling agentTool "{resolved["toolName"]}" '
                f"(input={resolved['input']})"
            )
            return await tool.execute(resolved["input"], None)
    base_url = os.environ.get("NEXT_PUBLIC_POKEAPI_BASE_URL", "https://pokeapi.co/api/v2")
    return await maybe_await(
        _apiServiceTool({"baseUrl": base_url, "schema": api_schema, "userToken": user_token})
    )


def _populate_empty_arrays(
    obj: Any, results: list[Any], path: str = ""
) -> None:
    """Mirror TS `populateEmptyArrays` — fills empty arrays with IDs."""
    if not isinstance(obj, dict):
        return
    for key in list(obj.keys()):
        full_path = f"{path}.{key}" if path else key
        v = obj[key]
        if isinstance(v, list) and len(v) == 0:
            num_ids = 1
            key_l = key.lower()
            if "pokemon" in key_l and "id" in key_l:
                num_ids = 3
            if "type" in key_l:
                extracted_ids = [
                    item.get("type_id") or item.get("id")
                    for item in results[:num_ids]
                ]
            else:
                extracted_ids = [
                    item.get("id") or item.get("pokemon_id")
                    for item in results[:num_ids]
                ]
            obj[key] = extracted_ids
            print(f"Populated {full_path} with: {json.dumps(extracted_ids, default=str)}")
        elif isinstance(v, dict):
            _populate_empty_arrays(v, results, full_path)


def _populate_single_ids(
    obj: Any, results: list[Any], path: str = ""
) -> None:
    """Mirror TS `populateSingleIds` — fills `*id` nulls with first result ID."""
    if not isinstance(obj, dict):
        return
    for key in list(obj.keys()):
        full_path = f"{path}.{key}" if path else key
        v = obj[key]
        if v is None and "id" in key.lower():
            first = results[0] if results else {}
            obj[key] = first.get("id") or first.get("pokemon_id")
            print(f"Populated {full_path} with single ID: {obj[key]}")
        elif isinstance(v, dict):
            _populate_single_ids(v, results, full_path)


async def execute_iterative_planner(
    refined_query: str,
    matched_apis: list[dict[str, Any]],
    initial_plan_response: str,
    api_key: str,
    user_token: str,
    final_deliverable: str,
    useful_data_map: dict[str, Any],
    conversation_context: str,
    entities: Optional[list[Any]] = None,
    request_context: Optional[RequestContext] = None,
    max_iterations: int = 50,
    reference_task: Optional[SavedTask] = None,
    on_tool_call: Optional[Callable[[dict[str, Any]], None]] = None,
) -> dict[str, Any]:
    """Iterative plan execution loop.

    Ported from TS `executeIterativePlanner` (executor.ts:261-1117). Preserves:
    - FETCH vs MODIFY validator branches with their own re-plan contexts
    - Fan-out support (per-value re-issue when path-param scalar gets an array)
    - depends_on_step dependency propagation (id/pokemon_id/teamId / type_id)
    - Placeholder reference resolution before each step
    - Stuck-state detection (2 consecutive iterations without progress)
    - Max-iteration / max-planning-cycle caps (50 / 20)
    - Pre-final-answer move filtering by query-mentioned type
    """
    if entities is None:
        entities = []
    if request_context is None:
        request_context = RequestContext()

    current_plan_response = initial_plan_response
    accumulated_results: list[dict[str, Any]] = []
    executed_steps: list[dict[str, Any]] = []
    iteration = 0
    plan_iteration = 0
    first_id = (matched_apis[0] or {}).get("id", "") if matched_apis else ""
    intent_type: Literal["FETCH", "MODIFY"] = (
        "FETCH" if first_id.startswith("semantic") else "MODIFY"
    )
    stuck_count = 0
    stopped_reason = ""

    reference_task_context = ""
    if reference_task is not None:
        reference_task_context = (
            "\n\nReference task (reuse if similar):\n"
            + json.dumps(
                {
                    "id": reference_task.id,
                    "taskName": reference_task.taskName,
                    "taskType": reference_task.taskType,
                    "steps": [vars(s) if hasattr(s, "__dict__") else s for s in (reference_task.steps or [])],
                },
                indent=2,
                default=str,
            )
            + "\nPrefer adapting this task's steps if it aligns with the current goal."
        )
    conversation_context_with_reference = (
        f"{conversation_context}{reference_task_context}"
        if reference_task_context
        else conversation_context
    )

    if reference_task is not None:
        print(
            f"🧭 Iterative executor using reference task {reference_task.id} "
            f"({reference_task.taskName}) for replanning context."
        )
    else:
        print("🧭 Iterative executor running without reference task.")

    print("\n" + "=" * 80)
    print("🔄 STARTING ITERATIVE PLANNER")
    print(f"Max API calls allowed: {max_iterations}")
    print("=" * 80)

    sanitized_plan_response = current_plan_response
    print(f"sanitizedPlanResponse: {sanitized_plan_response}")
    actionable_plan: dict[str, Any] = json.loads(sanitized_plan_response)

    while plan_iteration < 20:
        plan_iteration += 1
        print(
            f"\n--- Planning Cycle {plan_iteration} "
            f"(API calls made: {iteration}/{max_iterations}) ---"
        )

        try:
            print(f"Current Actionable Plan: {json.dumps(actionable_plan, indent=2, default=str)}")

            if actionable_plan.get("needs_clarification"):
                print(f"Planner requires clarification: {actionable_plan.get('reason')}")
                return {
                    "error": "Clarification needed",
                    "clarification_question": actionable_plan.get("clarification_question"),
                    "reason": actionable_plan.get("reason"),
                }

            execution_plan = actionable_plan.get("execution_plan") or []
            if not execution_plan:
                print("No more steps in execution plan")
                break

            progress_before_execution = len(accumulated_results)
            print(
                f"\n📋 Executing complete plan with {len(execution_plan)} steps"
            )
            print(
                f"📌 Execution mode: "
                + (
                    "MODIFY (execute all, validate once at end)"
                    if intent_type == "MODIFY"
                    else "FETCH (execute and validate)"
                )
            )

            has_api_error = False

            while execution_plan:
                step = execution_plan.pop(0)
                step_number_display = step.get("step_number") or len(executed_steps) + 1
                print(
                    f"\nExecuting step {step_number_display}: "
                    f"{json.dumps(step, indent=2, default=str)}"
                )

                # Reference-task MODIFY detection
                if actionable_plan.get("_from_reference_task") and reference_task is not None:
                    is_modify_step = step.get("stepType") == 2
                    step_method_l = ((step.get("api") or {}).get("method") or "").lower()
                    is_modify_by_method = step_method_l in ("post", "put", "patch", "delete")
                    if is_modify_step or is_modify_by_method:
                        print(
                            "\n⏸️  MODIFY step detected in reference task flow. "
                            "Pausing for user approval."
                        )
                        return {
                            "message": (
                                "Reference task execution paused before MODIFY action. "
                                "Review the gathered data and approve to proceed."
                            ),
                            "reason": "modify_step_reached",
                            "executedSteps": executed_steps,
                            "accumulatedResults": accumulated_results,
                            "remainingPlan": {
                                "steps": [step, *execution_plan],
                                "description": "Remaining steps to execute after user approval",
                            },
                            "iterations": iteration,
                        }

                api = step.get("api") or {}
                if not (api.get("path") and api.get("method")):
                    print(
                        f"⚠️  Step {step.get('step_number')} is not a valid API call "
                        f"(path: {api.get('path')}, method: {api.get('method')})"
                    )
                    print(
                        "This appears to be a computation/logic step. The planner "
                        "should only generate API call steps."
                    )
                    print(
                        "Skipping this step and will let validator determine if more API calls are needed."
                    )
                    continue

                # Multi-instance fan-out via depends_on_step + path params
                steps_to_execute: list[dict[str, Any]] = [step]
                depends_on = step.get("depends_on_step") or step.get("dependsOnStep")
                if depends_on and accumulated_results:
                    prev = next(
                        (r for r in accumulated_results if r.get("step") == depends_on),
                        None,
                    )
                    if prev and prev.get("response"):
                        response_data = prev["response"]
                        results = None
                        if isinstance(response_data, dict):
                            inner_result = response_data.get("result")
                            if isinstance(inner_result, dict) and isinstance(
                                inner_result.get("results"), list
                            ):
                                results = inner_result["results"]
                            elif isinstance(response_data.get("results"), list):
                                results = response_data["results"]
                        if isinstance(results, list) and len(results) > 1:
                            path_params = _PATH_PARAM_RE.findall(api["path"])
                            if path_params:
                                print(
                                    f"\n🔄 Step {step.get('step_number')} will be executed "
                                    f"{len(results)} times (once for each result from step {depends_on})"
                                )
                                steps_to_execute = []
                                for index, result_item in enumerate(results):
                                    cloned = json.loads(json.dumps(step, default=str))
                                    cloned["_executionIndex"] = index
                                    cloned["_sourceData"] = result_item
                                    steps_to_execute.append(cloned)

                for step_to_execute in steps_to_execute:
                    if iteration >= max_iterations:
                        print(
                            f"⚠️ Reached max iterations ({max_iterations}) during step execution"
                        )
                        return {
                            "error": "Max iterations reached",
                            "message": (
                                f"Sorry, I was unable to complete the task within the "
                                f"allowed {max_iterations} API calls."
                            ),
                            "executedSteps": executed_steps,
                            "accumulatedResults": accumulated_results,
                            "iterations": iteration,
                        }

                    iteration += 1
                    print(
                        f"\n📌 Executing API call #{iteration}/{max_iterations} "
                        f"(step {step_to_execute.get('step_number')})..."
                    )

                    step_api = step_to_execute.get("api") or {}
                    request_body_to_use = step_api.get("requestBody")
                    parameters_to_use = (
                        step_api.get("parameters") or step_to_execute.get("input") or {}
                    )

                    nested_depends_on = step_to_execute.get("depends_on_step") or step_to_execute.get("dependsOnStep")
                    if nested_depends_on and accumulated_results:
                        prev = next(
                            (
                                r
                                for r in accumulated_results
                                if r.get("step") == nested_depends_on
                            ),
                            None,
                        )
                        if prev and prev.get("response"):
                            print(
                                f"Step {step_to_execute.get('step_number')} depends on "
                                f"step {nested_depends_on} - populating data from previous results"
                            )
                            request_body_to_use = json.loads(
                                json.dumps(step_api.get("requestBody"), default=str)
                            )

                            results = None
                            if step_to_execute.get("_sourceData") is not None:
                                results = [step_to_execute["_sourceData"]]
                            else:
                                resp_data = prev["response"]
                                if isinstance(resp_data, dict):
                                    inner = resp_data.get("result")
                                    if isinstance(inner, dict) and isinstance(
                                        inner.get("results"), list
                                    ):
                                        results = inner["results"]
                                    elif isinstance(resp_data.get("results"), list):
                                        results = resp_data["results"]

                            if isinstance(results, list) and len(results) > 0:
                                _populate_empty_arrays(request_body_to_use, results)
                                _populate_single_ids(request_body_to_use, results)

                                api_path = step_api.get("path", "")
                                path_params = _PATH_PARAM_RE.findall(api_path)
                                if path_params:
                                    parameters_to_use = {
                                        **(step_api.get("parameters") or step_to_execute.get("input") or {})
                                    }
                                    for param_name in path_params:
                                        if not parameters_to_use.get(param_name):
                                            first = results[0]
                                            extracted = (
                                                first.get("id")
                                                or first.get("pokemon_id")
                                                or first.get("teamId")
                                            )
                                            if extracted:
                                                parameters_to_use[param_name] = extracted
                                                print(
                                                    f"✅ Auto-populated path parameter "
                                                    f"{{{param_name}}} with value: {extracted}"
                                                )
                                            else:
                                                print(
                                                    f"⚠️  Could not extract ID for path "
                                                    f"parameter {{{param_name}}} from previous step results"
                                                )

                    # Placeholder resolution
                    print(
                        f"\n🔎 Checking for placeholder references in step "
                        f"{step_to_execute.get('step_number') or len(executed_steps) + 1}..."
                    )
                    step_to_check = {
                        "api": {
                            "path": step_api.get("path"),
                            "method": step_api.get("method"),
                            "parameters": parameters_to_use,
                            "requestBody": request_body_to_use,
                        }
                    }
                    if contains_placeholder_reference(step_to_check):
                        print("⚠️  Step contains placeholder reference(s), attempting resolution...")
                        resolution = await resolve_placeholders(step_to_check, executed_steps, api_key)
                        if not resolution.get("resolved"):
                            reason = resolution.get("reason")
                            print(f"❌ CRITICAL: Failed to resolve placeholder - {reason}")
                            return {
                                "error": "Placeholder resolution failed",
                                "reason": reason,
                                "stepNumber": step_to_execute.get("step_number")
                                or len(executed_steps) + 1,
                                "step": step_to_check,
                                "executedSteps": executed_steps,
                                "accumulatedResults": accumulated_results,
                                "message": (
                                    f"Cannot proceed: {reason}. Please review the plan and "
                                    "ensure all referenced steps have been executed."
                                ),
                                "status": 400,
                            }
                        parameters_to_use = step_to_check["api"]["parameters"]
                        request_body_to_use = step_to_check["api"]["requestBody"]
                        print("✅ Placeholders resolved successfully")
                    else:
                        print("✅ No placeholder references detected")

                    parameters_schema = find_api_parameters(step_api["path"], step_api["method"])
                    api_schema = {
                        **step_api,
                        "requestBody": request_body_to_use,
                        "parameters": parameters_to_use,
                        "parametersSchema": parameters_schema,
                    }

                    api_response: Any
                    try:
                        api_response = await _call_api(
                            step_api["path"], parameters_to_use, api_schema, user_token
                        )
                    except Exception as err:  # noqa: BLE001
                        err_msg = str(err)
                        status_code = getattr(err, "statusCode", None) or getattr(err, "status", None)
                        print(
                            f"⚠️  API call encountered an error (statusCode: {status_code}): {err_msg}"
                        )

                        if intent_type == "MODIFY" and status_code in (400, 403, 404, 409, 422, 500):
                            print(
                                f"❌ MODIFY flow encountered critical error ({status_code}): {err_msg}"
                            )
                            has_api_error = True
                            execution_plan = []
                            api_response = {
                                "success": False,
                                "error": True,
                                "statusCode": status_code,
                                "message": err_msg or "API request failed",
                                "details": getattr(err, "response", None) or getattr(err, "data", None),
                                "_originalError": {
                                    "name": err.__class__.__name__,
                                    "message": err_msg,
                                },
                            }
                            sanitized_response = _sanitize_for_serialization(api_response)
                            executed_steps.append(
                                {
                                    "step": step_to_execute,
                                    "response": sanitized_response,
                                    "stepNumber": step_to_execute.get("step_number")
                                    or len(executed_steps),
                                    "error": True,
                                    "errorMessage": err_msg,
                                }
                            )
                            accumulated_results.append(
                                {
                                    "step": step_to_execute.get("step_number") or len(executed_steps),
                                    "description": step_to_execute.get("description") or "API call",
                                    "response": sanitized_response,
                                    "error": True,
                                    "errorMessage": err_msg,
                                }
                            )
                            print("📤 Error recorded. Exiting execution loop to re-plan.")
                            break

                        if "参数类型不匹配" in err_msg:
                            print(f"参数类型不匹配，打回AI重写: {err_msg}")
                            return {
                                "error": "参数类型不匹配",
                                "reason": err_msg,
                                "executedSteps": executed_steps,
                                "accumulatedResults": accumulated_results,
                                "clarification_question": (
                                    f"参数类型不匹配：{err_msg}。请根据API schema重写参数。"
                                ),
                            }

                        api_response = {
                            "success": False,
                            "error": True,
                            "statusCode": status_code or 500,
                            "message": err_msg or "API request failed",
                            "details": getattr(err, "response", None) or getattr(err, "data", None),
                            "_originalError": {
                                "name": err.__class__.__name__,
                                "message": err_msg,
                            },
                        }
                        print(f"📋 Treating error as response data: {api_response}")

                    # Fan-out support — preserve TS behaviour
                    if (
                        isinstance(api_response, dict)
                        and api_response.get("needsFanOut")
                    ):
                        fan_out_param = api_response.get("fanOutParam")
                        fan_out_values = api_response.get("fanOutValues") or []
                        base_schema = api_response.get("baseSchema") or {}
                        mapped_params = api_response.get("mappedParams") or {}
                        print(
                            f"\n🔄 需要 fan-out: {fan_out_param} = "
                            f"[{', '.join(map(str, fan_out_values))}]"
                        )
                        fan_out_results: list[dict[str, Any]] = []
                        for value in fan_out_values:
                            single_value_schema = {
                                **base_schema,
                                "parameters": {**mapped_params, fan_out_param: value},
                                "parametersSchema": parameters_schema,
                            }
                            print(f"  📤 Fan-out 调用 {fan_out_param}={value}")
                            try:
                                base_url = os.environ.get(
                                    "NEXT_PUBLIC_POKEAPI_BASE_URL",
                                    "https://pokeapi.co/api/v2",
                                )
                                single_result = await maybe_await(
                                    _apiServiceTool(
                                        {
                                            "baseUrl": base_url,
                                            "schema": single_value_schema,
                                            "userToken": user_token,
                                        }
                                    )
                                )
                            except Exception as fan_err:  # noqa: BLE001
                                fan_err_msg = str(fan_err)
                                if "参数类型不匹配" in fan_err_msg:
                                    print(f"参数类型不匹配，打回AI重写: {fan_err_msg}")
                                    return {
                                        "error": "参数类型不匹配",
                                        "reason": fan_err_msg,
                                        "executedSteps": executed_steps,
                                        "accumulatedResults": accumulated_results,
                                        "clarification_question": (
                                            f"参数类型不匹配：{fan_err_msg}。请根据API schema重写参数。"
                                        ),
                                    }
                                print(
                                    f"⚠️  Fan-out call for {fan_out_param}={value} "
                                    f"encountered an error: {fan_err_msg}"
                                )
                                single_result = {
                                    "success": False,
                                    "error": True,
                                    "statusCode": getattr(fan_err, "statusCode", None)
                                    or getattr(fan_err, "status", None)
                                    or 500,
                                    "message": fan_err_msg or "API request failed",
                                }
                            fan_out_results.append({fan_out_param: value, "result": single_result})
                        print(f"✅ Fan-out 完成，共 {len(fan_out_results)} 个结果")
                        merged_response = {
                            "fanOutResults": fan_out_results,
                            "summary": (
                                f"Retrieved data for {len(fan_out_results)} {fan_out_param}(s)"
                            ),
                        }
                        api_response.update(merged_response)

                    print(f"(executor) API Response: {api_response}")

                    flat_useful_data_map = request_context.flatUsefulDataMap
                    useful_data_array = request_context.usefulDataArray

                    api_call_key = _get_api_call_key(
                        api_schema["path"],
                        api_schema["method"],
                        parameters_to_use,
                        request_body_to_use,
                    )
                    prev_useful_data = flat_useful_data_map.get(api_call_key, "")
                    is_new_entry = api_call_key not in flat_useful_data_map

                    sanitized_response = _sanitize_for_serialization(api_response)

                    new_useful_data = await extract_useful_data_from_api_responses(
                        refined_query,
                        final_deliverable,
                        prev_useful_data,
                        json.dumps(sanitized_response, default=str),
                        api_schema,
                        matched_apis,
                    )
                    flat_useful_data_map[api_call_key] = new_useful_data
                    if is_new_entry:
                        useful_data_array.append(
                            UsefulDataEntry(
                                key=api_call_key,
                                data=new_useful_data,
                                timestamp=int(time.time() * 1000),
                            )
                        )
                    else:
                        for entry in useful_data_array:
                            if entry.key == api_call_key:
                                entry.data = new_useful_data
                                entry.timestamp = int(time.time() * 1000)
                                break

                    processed_response = api_response
                    try:
                        if isinstance(api_response, str):
                            processed_response = json.loads(api_response)
                        if isinstance(processed_response, (dict, list)):
                            processed_response = _sanitize_for_serialization(processed_response)
                    except Exception as e:  # noqa: BLE001
                        print(f"Could not process API response: {e}")

                    sanitized_processed = _sanitize_for_serialization(processed_response)
                    step_number_resolved = step_to_execute.get("step_number") or len(executed_steps) + 1
                    executed_steps.append(
                        {
                            "stepNumber": step_number_resolved,
                            "step": step_to_execute,
                            "response": sanitized_processed,
                        }
                    )
                    accumulated_results.append(
                        {
                            "step": step_to_execute.get("step_number") or len(executed_steps),
                            "description": step_to_execute.get("description") or "API call",
                            "response": sanitized_processed,
                            "executionIndex": step_to_execute.get("_executionIndex"),
                        }
                    )
                    if on_tool_call is not None:
                        try:
                            on_tool_call(
                                {
                                    "stepNumber": step_to_execute.get("step_number")
                                    or len(executed_steps),
                                    "description": step_to_execute.get("description") or "API call",
                                    "path": step_api.get("path"),
                                    "method": step_api.get("method"),
                                }
                            )
                        except Exception as e:  # noqa: BLE001
                            print(f"on_tool_call callback raised: {e}")

                    print(
                        f"✅ Step {step_to_execute.get('step_number') or len(executed_steps)} "
                        f"completed. Remaining steps in plan: {len(execution_plan)}"
                    )

                # If inner for-loop broke via the MODIFY-error path, propagate to outer
                if has_api_error and intent_type == "MODIFY":
                    break

            print(
                f"\n✅ Completed all planned steps. "
                f"Total executed: {len(accumulated_results) - progress_before_execution}"
            )

            # ---- MODIFY validation / re-plan ----
            if intent_type == "MODIFY" and not has_api_error:
                print(
                    "📌 MODIFY flow: All steps executed without errors. "
                    "Validating goal completion..."
                )
                all_matched_apis = await get_all_matched_apis(
                    entities=entities, intentType=intent_type, context=request_context
                )
                top_k_results = await get_top_k_results(all_matched_apis, 20)
                useful_data_str = serialize_useful_data_in_order(request_context)

                validation_result = await validate_need_more_actions(
                    refined_query,
                    _sanitize_for_serialization(executed_steps),
                    _sanitize_for_serialization(accumulated_results),
                    api_key,
                    actionable_plan,
                )
                print(f"Post-execution validation result: {validation_result}")

                if not validation_result.get("needsMoreActions"):
                    print("✅ MODIFY validation confirmed: goal is complete")
                    if validation_result.get("item_not_found"):
                        stopped_reason = "item_not_found"
                    break

                print(f"⚠️  MODIFY validation failed: {validation_result.get('reason')}")
                planner_context = (
                    f"\nOriginal Query: {refined_query}\n\n"
                    f"Executed Actions:\n{json.dumps(executed_steps, indent=2, default=str)}\n\n"
                    f"Results from Execution:\n{json.dumps(accumulated_results, indent=2, default=str)}\n\n"
                    f"Goal Completion Status:\n"
                    f"The user's goal has NOT been fully completed. Here's why:\n"
                    f"{validation_result.get('reason')}\n\n"
                    f"Missing Requirements:\n"
                    f"{json.dumps(validation_result.get('missing_requirements'), indent=2, default=str)}\n\n"
                    f"Suggested Next Action:\n"
                    f"{validation_result.get('suggested_next_action') or 'Generate a new plan to complete the goal'}\n\n"
                    f"Available APIs:\n{json.dumps(top_k_results[:10], indent=2, default=str)}\n\n"
                    f"Reference task (reuse if similar): "
                    f"{json.dumps({'id': reference_task.id, 'taskName': reference_task.taskName}) if reference_task is not None else 'N/A'}\n\n"
                    f"Please generate a NEW PLAN to complete the user's goal."
                )
                current_plan_response = await send_to_planner(
                    planner_context,
                    useful_data_str,
                    conversation_context_with_reference,
                    intent_type,
                )
                actionable_plan = json.loads(sanitize_planner_response(current_plan_response))
                print("✅ Generated new plan from validation feedback (MODIFY re-plan)")
                continue

            if intent_type == "MODIFY" and has_api_error:
                print("🔄 MODIFY flow error detected. Re-planning with error feedback...")
                all_matched_apis = await get_all_matched_apis(
                    entities=entities, intentType=intent_type, context=request_context
                )
                top_k_results = await get_top_k_results(all_matched_apis, 20)
                useful_data_str = serialize_useful_data_in_order(request_context)

                last_executed = executed_steps[-1] if executed_steps else None
                last_response = (last_executed or {}).get("response") or {}
                error_details = last_response.get("message") if isinstance(last_response, dict) else None
                last_step_info = (last_executed or {}).get("step") or {}
                last_step_api = last_step_info.get("api") or {}

                planner_context = (
                    f"\nOriginal Query: {refined_query}\n\n"
                    f"Previous Plan Failed:\n"
                    f"The following action encountered an error and was not completed:\n"
                    f"Step {last_step_info.get('step_number')}: {last_step_info.get('description')}\n"
                    f"API: {(last_step_api.get('method') or '').upper()} {last_step_api.get('path')}\n"
                    f"Error: {error_details or 'Unknown error'} (Status: {last_response.get('statusCode') if isinstance(last_response, dict) else None})\n\n"
                    f"Executed Steps Before Error:\n"
                    f"{json.dumps(executed_steps[:-1], indent=2, default=str)}\n\n"
                    f"Available APIs:\n{json.dumps(top_k_results[:10], indent=2, default=str)}\n\n"
                    f"Reference task (reuse if similar): "
                    f"{json.dumps({'id': reference_task.id, 'taskName': reference_task.taskName}) if reference_task is not None else 'N/A'}\n\n"
                    f"Please generate a NEW PLAN that avoids the failed action and finds an alternative approach."
                )
                current_plan_response = await send_to_planner(
                    planner_context,
                    useful_data_str,
                    conversation_context_with_reference,
                    intent_type,
                )
                actionable_plan = json.loads(sanitize_planner_response(current_plan_response))
                print("✅ Generated new plan after error (MODIFY recovery)")
                continue

            # ---- FETCH validation / re-plan ----
            if intent_type == "FETCH":
                progress_made = len(accumulated_results) > progress_before_execution
                if not progress_made:
                    stuck_count += 1
                    print(f"⚠️  No progress made in this iteration (stuck count: {stuck_count})")
                    if reference_task is not None and stuck_count >= 1:
                        print(
                            "⏹️  Short-circuiting iterative loop: reference task "
                            "provided but no progress made."
                        )
                        stopped_reason = "stuck_state"
                        break
                    if stuck_count >= 2:
                        print(
                            "Detected stuck state: no new API calls in 2 consecutive iterations"
                        )
                        break
                else:
                    stuck_count = 0

                print("\n🔍 FETCH flow: Validating if more actions are needed...")
                validation_result = await validate_need_more_actions(
                    refined_query,
                    _sanitize_for_serialization(executed_steps),
                    _sanitize_for_serialization(accumulated_results),
                    api_key,
                    actionable_plan,
                )
                print(f"Validation result: {validation_result}")

                if not validation_result.get("needsMoreActions"):
                    print("✅ Validator confirmed: sufficient information gathered")
                    if validation_result.get("item_not_found"):
                        print("❌ Item not found - will generate answer explaining this")
                        stopped_reason = "item_not_found"
                    break

                print(f"⚠️  Validator says more actions needed: {validation_result.get('reason')}")
                all_matched_apis = await get_all_matched_apis(
                    entities=entities, intentType=intent_type, context=request_context
                )
                top_k_results = await get_top_k_results(all_matched_apis, 20)
                useful_data_str = serialize_useful_data_in_order(request_context)

                planner_context = (
                    f"\nOriginal Query: {refined_query}\n\n"
                    f"Matched APIs Available: {json.dumps(top_k_results[:10], indent=2, default=str)}\n\n"
                    f"Executed Steps So Far: {json.dumps(executed_steps, indent=2, default=str)}\n\n"
                    f"Accumulated Results: {json.dumps(accumulated_results, indent=2, default=str)}\n\n"
                    f"Previous Plan: {json.dumps(actionable_plan, indent=2, default=str)}\n\n"
                    f"The validator says more actions are needed: "
                    f"{validation_result.get('suggested_next_action') or validation_result.get('reason')}\n\n"
                    f"Useful data from execution history that could help: "
                    f"{validation_result.get('useful_data') or 'N/A'}\n\n"
                    f"Reference task (reuse if similar): "
                    f"{json.dumps({'id': reference_task.id, 'taskName': reference_task.taskName}) if reference_task is not None else 'N/A'}\n\n"
                    "IMPORTANT: If the available APIs do not include an endpoint that can provide the required information:\n"
                    "1. Check if any of the accumulated results contain the information in a different format\n"
                    "2. Consider if the data can be derived or inferred from existing results\n"
                    "3. If truly impossible with available APIs, set needs_clarification: true with reason explaining what API is missing\n\n"
                    "Please generate the next step in the plan, or indicate that no more steps are needed."
                )
                current_plan_response = await send_to_planner(
                    planner_context,
                    useful_data_str,
                    conversation_context_with_reference,
                    intent_type,
                )
                actionable_plan = json.loads(sanitize_planner_response(current_plan_response))
                print("\n🔄 Generated new plan from validator feedback (FETCH mode)")

        except Exception as error:  # noqa: BLE001
            print(f"Error during iterative planner execution: {error}")
            return {
                "error": "Failed during iterative execution",
                "details": str(error),
                "executedSteps": executed_steps,
                "accumulatedResults": accumulated_results,
                "usefulData": useful_data_map,
            }

    if iteration >= max_iterations:
        print(f"Reached max API call limit ({max_iterations})")
        stopped_reason = "max_iterations"
    elif plan_iteration >= 20:
        print("Reached max planning cycles (20)")
        stopped_reason = "max_planning_cycles"
    elif stuck_count >= 2:
        print("Stopped due to stuck state (repeated validation reasons)")
        stopped_reason = "stuck_state"

    print("\n📊 Execution Summary:")
    print(f"  - Total API calls made: {iteration}/{max_iterations}")
    print(f"  - Planning cycles: {plan_iteration}")
    print(f"  - Stopped reason: {stopped_reason or 'goal_completed'}")

    print("\n" + "=" * 80)
    print("📝 GENERATING FINAL ANSWER")
    print("=" * 80)

    sanitized_accumulated = _sanitize_for_serialization(accumulated_results)
    prepared_results = _prepare_results_for_final_answer(refined_query, sanitized_accumulated)
    useful_data_str = serialize_useful_data_in_order(request_context)

    final_answer = await generate_final_answer(
        refined_query,
        prepared_results,
        api_key,
        stopped_reason,
        useful_data_str,
    )

    print("\n" + "=" * 80)
    print("✅ ITERATIVE PLANNER COMPLETED")
    print("=" * 80)

    return {
        "message": final_answer,
        "executedSteps": executed_steps,
        "accumulatedResults": accumulated_results,
        "usefulData": request_context.flatUsefulDataMap,
        "iterations": iteration,
    }


__all__ = [
    "resolve_poke_api_tool",
    "extract_useful_data_from_api_responses",
    "generate_final_answer",
    "execute_iterative_planner",
]
