"""services/chat_planner_service.py — planner support utilities.

Ported from open-react-template/services/chatPlannerService.ts.

Responsibilities (preserved from TS):
- `get_all_matched_apis` — keyword-scored tool retrieval (uses
  `pokemon_rag_service.get_matched_pokemon_tools`). Replaces the legacy
  OpenAI ada-002 embedding lookup.
- `get_top_k_results` — sort + slice the matched tools.
- `fetch_prompt_file` — load a prompt template from disk.
- `detect_entity_name`, `deep_clone`, `replace_in_object`,
  `substitute_api_placeholders` — entity name detection and `{POKEMON_NAME}`
  substitution helpers.
- `serialize_useful_data_in_order` — chronological JSON dump of
  `RequestContext.usefulDataArray`.
- `extract_json` — brace-balanced JSON extractor (matches the TS impl).
- `estimate_tokens` — `len(text)//4` heuristic.
- `contains_placeholder_reference` / `resolve_placeholders` —
  `resolved_from_step_N` placeholder detection and LLM-driven resolution.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Optional

from ..utils.ai_handler import openai_chat_completion
from .pokemon_rag_service import (
    PokemonTool,
    get_matched_pokemon_tools,
)


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


@dataclass
class UsefulDataEntry:
    """One entry in `RequestContext.usefulDataArray`."""

    key: str
    data: str
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))


@dataclass
class RequestContext:
    """Request-scoped context — keeps planner state isolated per turn.

    Mirrors TS `RequestContext`. The CLI port uses one instance per REPL
    turn so concurrent prompts (even in tests) don't bleed into each other.
    """

    ragEntity: Optional[str] = None
    flatUsefulDataMap: dict[str, Any] = field(default_factory=dict)
    usefulDataArray: list[UsefulDataEntry] = field(default_factory=list)


@dataclass
class PlannerMessage:
    """Alias for the chat message shape used by planner helpers.

    Distinct from `schemas.ai.ChatMessage` only in that it allows the
    additional planner-internal fields TS uses (none required today).
    """

    role: Literal["user", "assistant", "system"]
    content: str


# ---------------------------------------------------------------------------
# Tool retrieval
# ---------------------------------------------------------------------------


async def get_all_matched_apis(
    *,
    entities: list[str],
    intentType: Literal["FETCH", "MODIFY"],
    context: Optional[RequestContext] = None,
) -> dict[str, PokemonTool]:
    """Return all PokéAPI tools matched to the given entities.

    Ported from TS `getAllMatchedApis`. PokéAPI is read-only so `intentType`
    is accepted for interface parity but does not alter the result set.
    """
    _ = context  # accepted for TS parity, currently unused
    print(
        f"🔎 PokéAPI RAG: keyword matching for entities=[{', '.join(entities)}], "
        f"intent={intentType}"
    )
    all_matched: dict[str, PokemonTool] = {}
    for tool in get_matched_pokemon_tools(entities):
        all_matched[tool.id] = tool
    print(f"✅ PokéAPI RAG: matched {len(all_matched)} tools")
    return all_matched


async def get_top_k_results(
    all_matched_apis: dict[str, PokemonTool], top_k: int
) -> list[dict[str, Any]]:
    """Return the top-K entries from the matched APIs map, sorted by similarity.

    Output shape mirrors TS: `{id, summary, tags, content}` dicts.
    """
    sorted_items = sorted(
        all_matched_apis.values(), key=lambda t: t.similarity, reverse=True
    )
    sliced = sorted_items[:top_k]
    results: list[dict[str, Any]] = [
        {
            "id": item.id,
            "summary": item.summary,
            "tags": item.tags,
            "content": item.content,
        }
        for item in sliced
    ]
    print(
        f"\n✅ Top {len(results)} tools selected: {[t['id'] for t in results]}"
    )
    return results


# ---------------------------------------------------------------------------
# Prompt-file loading
# ---------------------------------------------------------------------------


def _prompt_candidate_dirs() -> list[Path]:
    """Where to look for `fetch_prompt_file` lookups, in order."""
    candidates: list[Path] = []
    env_override = os.environ.get("PROMPTS_DIR")
    if env_override:
        candidates.append(Path(env_override).expanduser().resolve())
    here = Path(__file__).resolve().parent.parent
    candidates.append(here / "prompts")
    project_root = here.parent
    candidates.append(project_root.parent / "open-react-template" / "src" / "doc")
    return candidates


async def fetch_prompt_file(file_name: str) -> str:
    """Read a prompt file. Mirrors TS `fetchPromptFile`.

    Search order: `PROMPTS_DIR` env var → bundled `pokemon_bot/prompts/` →
    sibling repo's `src/doc/`. Raises `RuntimeError` if none of them
    contain the file (same surfaced error as the TS version).
    """
    for base in _prompt_candidate_dirs():
        candidate = base / file_name
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")
    raise RuntimeError(
        f"Error fetching prompt file: {file_name} not found in any of "
        f"{[str(p) for p in _prompt_candidate_dirs()]}"
    )


# ---------------------------------------------------------------------------
# Entity name detection + placeholder substitution
# ---------------------------------------------------------------------------


_QUOTED_RE = re.compile(r"""['"]([^'"]+)['"]""")
_VERB_NOUN_RE = re.compile(r"\b(?:add|remove|delete|drop|clear)\s+([A-Za-z0-9_-]+)", re.IGNORECASE)


def detect_entity_name(refined_query: str) -> Optional[str]:
    """Extract the primary entity from a refined query.

    Mirrors TS `detectEntityName`: prefer quoted spans, fall back to a
    verb-noun pattern, then the last whitespace-separated token > 1 char.
    """
    text = refined_query or ""
    q = _QUOTED_RE.search(text)
    if q:
        return q.group(1)
    vn = _VERB_NOUN_RE.search(text)
    if vn:
        return vn.group(1)
    parts = text.strip().split()
    if not parts:
        return None
    last_token = parts[-1]
    return last_token if len(last_token) > 1 else None


def deep_clone(obj: Any) -> Any:
    """Deep-clone via JSON round-trip — matches TS `deepClone`'s semantics."""
    if obj is None:
        return {}
    return json.loads(json.dumps(obj))


def replace_in_object(obj: Any, search: str, replacement: str) -> None:
    """In-place case-insensitive string replace inside a nested dict/list.

    Mirrors TS `replaceInObject` (which mutates with
    `obj[key] = obj[key].replace(new RegExp(search, 'gi'), replacement)`).
    """
    if not isinstance(obj, dict):
        return
    pattern = re.compile(re.escape(search), re.IGNORECASE)
    for key in list(obj.keys()):
        v = obj[key]
        if isinstance(v, str):
            obj[key] = pattern.sub(replacement, v)
        elif isinstance(v, dict):
            replace_in_object(v, search, replacement)


def substitute_api_placeholders(
    api: Optional[dict[str, Any]],
    refined_query: str,
    fallback: dict[str, str],
) -> dict[str, Any]:
    """Substitute `{POKEMON_NAME}` placeholders inside an API descriptor.

    Mirrors TS `substituteApiPlaceholders`. Returns a fresh `{path, method,
    parameters, requestBody}` dict; mutations are isolated to the deep clones.
    """
    api = api or {}
    entity_name = detect_entity_name(refined_query)
    method = (api.get("method") or fallback.get("method") or "get").lower()
    api_path = api.get("path") or fallback.get("path") or "/pokemon"
    parameters = deep_clone(api.get("parameters") or {})
    request_body = deep_clone(api.get("requestBody") or {})

    if entity_name:
        replace_in_object(parameters, "{POKEMON_NAME}", entity_name)
        replace_in_object(request_body, "{POKEMON_NAME}", entity_name)
        api_path = api_path.replace("{POKEMON_NAME}", entity_name)

    return {
        "path": api_path,
        "method": method,
        "parameters": parameters,
        "requestBody": request_body,
    }


def serialize_useful_data_in_order(context: RequestContext) -> str:
    """JSON-dump `usefulDataArray` ordered by timestamp.

    Mirrors TS `serializeUsefulDataInOrder`. Returns the literal `"{}"`
    string when the array is empty (preserved for prompt-string parity).
    """
    if not context.usefulDataArray:
        return "{}"
    ordered = sorted(context.usefulDataArray, key=lambda e: e.timestamp)
    ordered_entries = {entry.key: entry.data for entry in ordered}
    return json.dumps(ordered_entries, indent=2)


# ---------------------------------------------------------------------------
# JSON extractor + token estimator
# ---------------------------------------------------------------------------


def extract_json(content: str) -> Optional[dict[str, str]]:
    """Pull out the first balanced JSON object/array from `content`.

    Mirrors TS `extractJSON`. Returns `{json, text}` where `text` is the
    portion preceding the JSON span, or `None` if no balanced span parses.
    """
    try:
        trimmed = content.strip()
        obj_start = trimmed.find("{")
        arr_start = trimmed.find("[")
        if obj_start == -1 and arr_start == -1:
            return None

        json_start = -1
        json_end = -1
        if obj_start != -1 and (arr_start == -1 or obj_start < arr_start):
            json_start = obj_start
            depth = 0
            for i in range(obj_start, len(trimmed)):
                if trimmed[i] == "{":
                    depth += 1
                elif trimmed[i] == "}":
                    depth -= 1
                if depth == 0:
                    json_end = i + 1
                    break
        elif arr_start != -1:
            json_start = arr_start
            depth = 0
            for i in range(arr_start, len(trimmed)):
                if trimmed[i] == "[":
                    depth += 1
                elif trimmed[i] == "]":
                    depth -= 1
                if depth == 0:
                    json_end = i + 1
                    break

        if json_start == -1 or json_end == -1:
            return None
        json_str = trimmed[json_start:json_end]
        text = trimmed[:json_start].strip()
        json.loads(json_str)
        return {"json": json_str, "text": text}
    except Exception:  # noqa: BLE001
        return None


def estimate_tokens(text: str) -> int:
    """Crude 1 token ≈ 4 chars estimator. Mirrors TS `estimateTokens`."""
    return math.ceil(len(text) / 4)


# ---------------------------------------------------------------------------
# Placeholder reference detection + LLM resolution
# ---------------------------------------------------------------------------


_PLACEHOLDER_REF_RE = re.compile(r"resolved_from_step_\d+", re.IGNORECASE)


def contains_placeholder_reference(obj: Any) -> bool:
    """Recursively check if `obj` contains a `resolved_from_step_N` string."""

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

    Ported from TS `resolvePlaceholders`. Mutates `step_to_execute` in place
    to substitute the extracted values. Returns `{resolved, reason?}`.

    The extraction prompt is preserved byte-for-byte from the TS source so
    behaviour parity is maintained.
    """
    _ = api_key  # accepted for TS parity; the underlying client reads from env
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
            or (isinstance(s.get("step"), dict) and s["step"].get("step_number") == placeholder_step_num)
        ):
            referenced_step = s
            break

    if referenced_step is None:
        reason = f"Referenced step {placeholder_step_num} has not been executed yet"
        print(f"❌ {reason}")
        return {"resolved": False, "reason": reason}

    print(f"\n📋 RESOLVING PLACEHOLDER: resolved_from_step_{placeholder_step_num}")

    try:
        extraction_prompt = (
            "You are a data extraction expert. Extract the correct value to "
            'replace a "resolved_from_step_X" placeholder.\n\n'
            "RULES:\n"
            "1. Analyze the current step's API call to understand what value is needed\n"
            "2. Look at the referenced step's response to find the matching data\n"
            "3. Return ONLY the extracted value (no explanation, no JSON wrapping)\n"
            '4. If the data cannot be found, return "ERROR: [reason]"\n\n'
            "Current Step:\n"
            f"- API Path: {api.get('path')}\n"
            f"- Parameters: {json.dumps(api.get('parameters') or {}, default=str)}\n\n"
            f"Previous Step (Step {placeholder_step_num}) Response:\n"
            f"{json.dumps(referenced_step.get('response'), indent=2, default=str)}\n\n"
            "Return ONLY the value to substitute:"
        )
        extracted_value = await openai_chat_completion(
            messages=[{"role": "system", "content": extraction_prompt}],
            temperature=0.2,
            max_tokens=100,
        )
        print(f'✅ LLM extracted value: "{extracted_value}"')
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
    "RequestContext",
    "UsefulDataEntry",
    "PlannerMessage",
    "get_all_matched_apis",
    "get_top_k_results",
    "fetch_prompt_file",
    "detect_entity_name",
    "deep_clone",
    "replace_in_object",
    "substitute_api_placeholders",
    "serialize_useful_data_in_order",
    "extract_json",
    "estimate_tokens",
    "contains_placeholder_reference",
    "resolve_placeholders",
]
