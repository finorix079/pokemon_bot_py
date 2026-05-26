"""chat/planner_utils.py — planner orchestration + planner-response sanitiser.

Ported from open-react-template/app/api/chat/plannerUtils.ts.

The TS code uses `jaison` (a JSON5-style repair library) to fix malformed
planner output. We replace it with `_jaison_lite`: a minimal cleanup pass
that handles the common LLM-generated issues (`'` → `"`, trailing commas,
markdown fences). Strict JSON-parsable output is unchanged.

The TS reference-task fast path is preserved (skipped at runtime in the
CLI port because `reference_task` is always `None`, but kept for parity).
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal, Optional

from ..services.chat_planner_service import RequestContext
from ..services.task_service import SavedTask
from ..utils.ai_handler import openai_chat_completion
from .planner import send_to_planner


# ---------------------------------------------------------------------------
# JSON repair helper (jaison replacement)
# ---------------------------------------------------------------------------


_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")
_FENCE_RE = re.compile(r"```json|```")
_SINGLE_QUOTE_KEY_RE = re.compile(r"(?<=[\{,\s])'([A-Za-z0-9_]+)'\s*:")
_SINGLE_QUOTE_VALUE_RE = re.compile(r":\s*'([^']*)'")
_FIRST_JSON_RE = re.compile(r"\{[\s\S]*\}|\[[\s\S]*\]")


def _jaison_lite(text: str) -> Optional[Any]:
    """Minimal JSON-repair pass — handles the cases jaison was added for.

    Tries strict `json.loads` first, then progressively cleans:
    - strip markdown fences
    - convert single-quoted keys/values to double-quoted
    - strip trailing commas before `}` or `]`

    Returns the parsed value or `None` if everything fails.
    """
    candidates = [text]
    cleaned = _FENCE_RE.sub("", text).strip()
    if cleaned != text:
        candidates.append(cleaned)
    repaired = _SINGLE_QUOTE_KEY_RE.sub(r'"\1":', cleaned)
    repaired = _SINGLE_QUOTE_VALUE_RE.sub(r': "\1"', repaired)
    repaired = _TRAILING_COMMA_RE.sub(r"\1", repaired)
    candidates.append(repaired)
    for cand in candidates:
        try:
            return json.loads(cand)
        except Exception:  # noqa: BLE001
            continue
    return None


# ---------------------------------------------------------------------------
# Planner-response sanitiser
# ---------------------------------------------------------------------------


def sanitize_planner_response(response: str) -> str:
    """Sanitise a raw planner response into a valid JSON string.

    Ported from TS `sanitizePlannerResponse`. Extracts the first balanced
    JSON object or array, runs it through `_jaison_lite`, and returns the
    re-serialised JSON string. Raises on failure.
    """
    try:
        m = _FIRST_JSON_RE.search(response)
        if not m:
            raise RuntimeError("No JSON object or array found in the response.")
        json_fixed = _jaison_lite(m.group(0))
        if json_fixed is not None:
            return json.dumps(json_fixed)
        raise RuntimeError("No valid JSON found in the response.")
    except Exception as error:  # noqa: BLE001
        print(f"Error sanitizing planner response: {error}")
        raise


# ---------------------------------------------------------------------------
# Reference-task fast path
# ---------------------------------------------------------------------------


_CONTENT_RE = re.compile(r"—\s*(\w+)\s+(.+?)$")


def _reference_task_steps_to_plan(reference_task: SavedTask) -> list[dict[str, Any]]:
    plan: list[dict[str, Any]] = []
    for step in reference_task.steps or []:
        step_content = step.stepContent or ""
        m = _CONTENT_RE.search(step_content)
        if m:
            method = m.group(1).lower()
            path = m.group(2).strip()
        else:
            method = "get"
            path = "/pokemon"
        plan.append(
            {
                "step_number": step.stepOrder or 1,
                "description": step_content or f"Step {step.stepOrder}",
                "api": {
                    "path": path,
                    "method": method,
                    "parameters": {},
                    "requestBody": {},
                },
                "stepType": step.stepType,
            }
        )
    return plan


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------


async def run_planner_with_inputs(
    *,
    topKResults: list[Any],
    refinedQuery: str,
    apiKey: str,
    usefulData: str,
    conversationContext: Optional[str] = None,
    finalDeliverable: Optional[str] = None,
    intentType: Literal["FETCH", "MODIFY"] = "FETCH",
    entities: Optional[list[str]] = None,
    requestContext: Optional[RequestContext] = None,
    referenceTask: Optional[SavedTask] = None,
) -> dict[str, Any]:
    """Prepare planner inputs, call `send_to_planner`, parse + adjust.

    Mirrors TS `runPlannerWithInputs`. Returns `{actionablePlan, planResponse}`.
    The reference-task fast path is preserved even though it never fires in
    the CLI port (referenceTask is always None for now).
    """
    _ = topKResults, apiKey, requestContext
    print(f"🔀 Planner routing: intentType={intentType}")

    if (
        referenceTask is not None
        and getattr(referenceTask, "steps", None)
        and isinstance(referenceTask.steps, list)
        and len(referenceTask.steps) > 0
    ):
        print(
            f"✨ FAST PATH: Using reference task plan ({len(referenceTask.steps)} steps)"
        )
        execution_plan = _reference_task_steps_to_plan(referenceTask)
        actionable_plan: dict[str, Any] = {
            "needs_clarification": False,
            "phase": "execution",
            "final_deliverable": finalDeliverable or "",
            "execution_plan": execution_plan,
            "selected_tools_spec": [],
            "_from_reference_task": True,
            "_reference_task_id": referenceTask.id,
        }
        plan_response = json.dumps(actionable_plan)
        print(f"🪄 Using reference plan: {plan_response}")
        return {"actionablePlan": actionable_plan, "planResponse": plan_response}

    plan_response = await send_to_planner(
        refinedQuery, usefulData, conversationContext, intentType
    )

    try:
        sanitised = sanitize_planner_response(plan_response)
        actionable_plan = json.loads(sanitised)
        if actionable_plan and finalDeliverable:
            actionable_plan["final_deliverable"] = finalDeliverable
        if isinstance(actionable_plan, dict):
            actionable_plan.pop("_replanned", None)
            actionable_plan.pop("_replan_reason", None)
        if isinstance(actionable_plan, dict) and actionable_plan.get("impossible"):
            print("🚫 Planner marked task as impossible.")
            return {
                "actionablePlan": actionable_plan,
                "planResponse": plan_response,
            }
    except Exception as error:  # noqa: BLE001
        print(f"Failed to parse planner response as JSON: {error}")
        print(f"Original Planner Response: {plan_response}")
        raise RuntimeError("Failed to parse planner response") from error

    if (
        intentType == "MODIFY"
        and isinstance(actionable_plan, dict)
        and actionable_plan.get("phase") == "resolution"
        and entities
        and len(entities) > 0
    ):
        print(
            "♻️ Plan is resolution-only for MODIFY intent; re-planning with forceFullPlan=true..."
        )
        retry_response = await send_to_planner(
            refinedQuery, usefulData, conversationContext, "MODIFY", True
        )
        sanitised_retry = sanitize_planner_response(retry_response)
        actionable_plan = json.loads(sanitised_retry)
        plan_response = retry_response
        if isinstance(actionable_plan, dict) and finalDeliverable:
            actionable_plan["final_deliverable"] = finalDeliverable

    return {"actionablePlan": actionable_plan, "planResponse": plan_response}


# ---------------------------------------------------------------------------
# Placeholder helpers (duplicated from chat_planner_service for parity)
# ---------------------------------------------------------------------------


_PLACEHOLDER_REF_RE = re.compile(r"resolved_from_step_\d+", re.IGNORECASE)


def contains_placeholder_reference(obj: Any) -> bool:
    """Recursively check for `resolved_from_step_N` strings. Mirrors TS."""

    def check_value(value: Any) -> bool:
        if isinstance(value, str):
            return bool(_PLACEHOLDER_REF_RE.search(value))
        if isinstance(value, list):
            return any(check_value(v) for v in value)
        if isinstance(value, dict):
            return any(check_value(v) for v in value.values())
        return False

    return check_value(obj)


async def resolve_placeholders(
    step_to_execute: dict[str, Any],
    executed_steps: list[dict[str, Any]],
    api_key: str,
) -> dict[str, Any]:
    """Resolve `resolved_from_step_N` placeholders via an LLM extractor.

    This is the *plannerUtils* variant (TS lines 160-270). It uses a slimmer
    extraction prompt than the one in `chat_planner_service.resolve_placeholders`
    — both are kept here for full TS parity, including the duplication.
    """
    _ = api_key
    placeholder_re = re.compile(r"resolved_from_step_(\d+)", re.IGNORECASE)
    found_placeholder = False
    placeholder_step_num: Optional[int] = None

    api = step_to_execute.get("api") or {}
    parameters = api.get("parameters")
    if isinstance(parameters, dict):
        for key, value in parameters.items():
            if isinstance(value, str):
                m = placeholder_re.search(value)
                if m:
                    found_placeholder = True
                    placeholder_step_num = int(m.group(1))
                    print(
                        f"🔍 Detected placeholder in parameters.{key}: "
                        f'"{value}" (references step {placeholder_step_num})'
                    )

    request_body = api.get("requestBody")

    def check_body(obj: Any, body_path: str = "") -> bool:
        nonlocal found_placeholder, placeholder_step_num
        if not isinstance(obj, dict):
            return False
        for key, value in obj.items():
            full_path = f"{body_path}.{key}" if body_path else key
            if isinstance(value, str):
                m = placeholder_re.search(value)
                if m:
                    found_placeholder = True
                    placeholder_step_num = int(m.group(1))
                    print(
                        f"🔍 Detected placeholder in requestBody.{full_path}: "
                        f'"{value}" (references step {placeholder_step_num})'
                    )
                    return True
            elif isinstance(value, dict):
                if check_body(value, full_path):
                    return True
        return False

    if isinstance(request_body, dict):
        check_body(request_body)

    if not found_placeholder or placeholder_step_num is None:
        return {"resolved": True}

    referenced_step: Optional[dict[str, Any]] = None
    for s in executed_steps:
        if (
            s.get("step") == placeholder_step_num
            or s.get("stepNumber") == placeholder_step_num
            or (
                isinstance(s.get("step"), dict)
                and s["step"].get("step_number") == placeholder_step_num
            )
        ):
            referenced_step = s
            break

    if referenced_step is None:
        reason = (
            f"Referenced step {placeholder_step_num} has not been executed yet"
        )
        print(f"❌ {reason}")
        return {"resolved": False, "reason": reason}

    try:
        # Slim extraction prompt — matches TS plannerUtils.ts:222-237.
        extraction_prompt = (
            f"Extract the value to replace "
            f'"resolved_from_step_{placeholder_step_num}" from the previous '
            f"step's response.\n"
            f"Return ONLY the raw value (no JSON, no explanation).\n"
            f'If not found, return "ERROR: not found".\n\n'
            f"Current step API path: {api.get('path')}\n"
            f"Previous step response:\n"
            f"{json.dumps(referenced_step.get('response'), indent=2, default=str)}"
        )
        extracted_value = await openai_chat_completion(
            messages=[{"role": "system", "content": extraction_prompt}],
            temperature=0.2,
            max_tokens=100,
        )
        print(f'✅ Extracted placeholder value: "{extracted_value}"')
        if not extracted_value or extracted_value.startswith("ERROR:"):
            return {
                "resolved": False,
                "reason": f"Failed to extract value: {extracted_value}",
            }

        token = f"resolved_from_step_{placeholder_step_num}"
        if isinstance(parameters, dict):
            for key, value in list(parameters.items()):
                if isinstance(value, str) and token in value:
                    parameters[key] = extracted_value

        if isinstance(request_body, dict):

            def replace_in_body(obj: Any) -> None:
                if not isinstance(obj, dict):
                    return
                for key, value in list(obj.items()):
                    if isinstance(value, str) and token in value:
                        obj[key] = value.replace(token, extracted_value)
                    elif isinstance(value, dict):
                        replace_in_body(value)

            replace_in_body(request_body)

        return {"resolved": True}
    except Exception as error:  # noqa: BLE001
        print(f"❌ Error resolving placeholder: {error}")
        return {"resolved": False, "reason": str(error)}


__all__ = [
    "run_planner_with_inputs",
    "sanitize_planner_response",
    "contains_placeholder_reference",
    "resolve_placeholders",
]
