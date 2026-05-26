"""chat/message_utils.py — message summarisation / filtering / context utils.

Ported from open-react-template/app/api/chat/messageUtils.ts.

Notes on prompt preservation: the Kimi summarisation prompt (TS L44-85)
is preserved byte-for-byte, and `filter_plan_messages` keeps every one of
the case-insensitive substring checks from the TS source.
"""

from __future__ import annotations

import asyncio
import json
import math
import re
from typing import Any, Optional

from ..services.chat_planner_service import RequestContext
from ..utils.ai_handler import kimi_chat_completion


# ---------------------------------------------------------------------------
# Useful-data serialisation + token estimator
# ---------------------------------------------------------------------------


def serialize_useful_data_in_order(context: RequestContext) -> str:
    """Serialise `usefulDataArray` entries in chronological order.

    Mirrors TS `serializeUsefulDataInOrder`. Returns `"{}"` (literal string,
    not `"{\n}"`) when the array is empty — preserved for prompt parity.

    Note: a near-duplicate of this lives in `services.chat_planner_service`;
    both are kept to mirror the TS file structure (the TS code has the same
    duplication for the same reason — module isolation).
    """
    if not context.usefulDataArray:
        return "{}"
    ordered = sorted(context.usefulDataArray, key=lambda e: e.timestamp)
    ordered_entries = {entry.key: entry.data for entry in ordered}
    return json.dumps(ordered_entries, indent=2)


def estimate_tokens(text: str) -> int:
    """Rough token estimator: `len // 4`. Mirrors TS `estimateTokens`."""
    return math.ceil(len(text) / 4)


# ---------------------------------------------------------------------------
# Single-message summariser (Kimi)
# ---------------------------------------------------------------------------


_SUMMARIZE_SYSTEM_PROMPT = """You are a message summarizer that extracts ONLY critical information from conversation messages.

CRITICAL RULES - NO DATA LOSS PERMITTED:
1. Preserve ALL numbers, IDs, quantities, counts, statistics (e.g., "125 moves", "ID: 25", "3 Pokémon")
2. Preserve ALL names: Pokémon names, move names, ability names, team names, item names
3. Preserve ALL specific entities and identifiers (e.g., "Pikachu", "Thunderbolt", "Static")
4. Preserve ALL data values, measurements, and attributes (e.g., "Electric-type", "power: 90")
5. Preserve ALL relationships and associations (e.g., "Pikachu's moves", "team members", "in watchlist")
6. Preserve ALL lists and enumerations completely (e.g., if 10 items mentioned, keep all 10)
7. Preserve ALL temporal information (e.g., "added on 2023-01-15", "last updated")
8. Remove conversational fluff, greetings, explanations, and filler text
9. Keep only factual data in a concise structured format

Special Attention To:
- Pokemon data: name, ID, type(s), abilities, stats, moves, evolution
- Move data: name, type, power, accuracy, PP, damage class, learning method
- Team data: team name, team ID, member count, member names and IDs
- Watchlist data: item names, IDs, count
- Ability data: name, effect, hidden/normal
- User actions: what was requested, what was completed

Format: Extract as bullet points or compact sentences.

Examples:

Input: "Great! I found Pikachu for you. Pikachu is an Electric-type Pokémon with ID 25. It has the abilities Static and Lightning Rod. It can learn 125 moves including Thunderbolt, Quick Attack, and Thunder Shock. Would you like to know more about any of these moves?"

Output: "Pikachu (ID: 25), Electric-type, Abilities: Static, Lightning Rod, Can learn 125 moves: Thunderbolt, Quick Attack, Thunder Shock"

Input: "I've checked your watchlist. You currently have 3 Pokémon in your watchlist: Charizard (ID: 6), Mewtwo (ID: 150), and Dragonite (ID: 149). They were added on different dates. Is there anything you'd like to do with these?"

Output: "Watchlist: 3 Pokémon - Charizard (ID: 6), Mewtwo (ID: 150), Dragonite (ID: 149)"

Input: "Your team 'Elite Squad' (ID: 42) has 6 members: Pikachu, Charizard, Blastoise, Venusaur, Gengar, Dragonite. The team was created on 2024-01-15 and last updated on 2024-06-20."

Output: "Team 'Elite Squad' (ID: 42): 6 members - Pikachu, Charizard, Blastoise, Venusaur, Gengar, Dragonite. Created: 2024-01-15, Updated: 2024-06-20"

Input: "add pikachu to my team"
Output: "add pikachu to my team" (keep short messages as-is)

Now summarize this message:"""


async def summarize_message(message: dict[str, Any], api_key: str) -> dict[str, Any]:
    """Summarise a single message while preserving critical data.

    Short messages (< 500 chars) and system messages are returned unchanged.
    Mirrors TS `summarizeMessage`. The system prompt is byte-identical to
    the TS source.
    """
    _ = api_key  # kept for TS signature parity; the underlying client reads from env
    content = message.get("content", "")
    role = message.get("role")
    if len(content) < 500 or role == "system":
        return message

    try:
        summarised = await kimi_chat_completion(
            messages=[
                {"role": "system", "content": _SUMMARIZE_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            temperature=0.1,
            max_tokens=1024,
        )
        if summarised and len(summarised) < len(content):
            pct = round((1 - len(summarised) / len(content)) * 100)
            print(
                f"📝 Summarized message: {len(content)} → {len(summarised)} "
                f"chars ({pct}% reduction)"
            )
            new_msg = dict(message)
            new_msg["content"] = summarised
            return new_msg
    except Exception as error:  # noqa: BLE001
        print(f"Message summarization failed: {error}")
    return message


# ---------------------------------------------------------------------------
# Plan-message filtering
# ---------------------------------------------------------------------------


_USER_APPROVAL_EXACT_RE = re.compile(
    r"^(approve|yes|no|reject|cancel|abort|proceed|ok|confirm|go ahead|deny|decline|disagree)$",
    re.IGNORECASE,
)
_USER_APPROVAL_TRAILING_RE = re.compile(r"^(approve|yes|reject|no|cancel)[\s!.]*$")
_USER_APPROVAL_PREFIX_RE = re.compile(r"^(please )?(approve|reject|cancel|proceed)")

_ASSISTANT_PLAN_SUBSTRINGS = (
    "what you're about to do",
    "execution_plan",
    "needs_clarification",
    '"phase":',
    "would you like me to",
    "here's the plan",
    "i'll now",
    "approval needed",
    "do you approve",
)

_ASSISTANT_CLARIFY_SUBSTRINGS = (
    "could you clarify",
    "what do you mean",
    "i need clarification",
    "please clarify",
    "could you please provide",
    "i'm not sure what you mean",
)


def filter_plan_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Drop plan proposals, approval responses, and clarification requests.

    Mirrors TS `filterPlanMessages` (substring lists + three regexes).
    """
    filtered: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        raw = message.get("content", "")
        content_lower = raw.lower()
        content_stripped = raw.strip()

        if role == "assistant" and any(
            sub in content_lower for sub in _ASSISTANT_PLAN_SUBSTRINGS
        ):
            print("⏭️ Filtering out plan proposal from assistant")
            continue
        if role == "user" and (
            _USER_APPROVAL_EXACT_RE.match(content_stripped)
            or _USER_APPROVAL_TRAILING_RE.match(content_stripped)
            or _USER_APPROVAL_PREFIX_RE.match(content_stripped)
        ):
            print("⏭️ Filtering out approval/rejection message from user")
            continue
        if role == "assistant" and any(
            sub in content_lower for sub in _ASSISTANT_CLARIFY_SUBSTRINGS
        ):
            print("⏭️ Filtering out clarification request from assistant")
            continue
        filtered.append(message)
    return filtered


# ---------------------------------------------------------------------------
# Multi-message summariser
# ---------------------------------------------------------------------------


async def summarize_messages(
    messages: list[dict[str, Any]], api_key: str
) -> list[dict[str, Any]]:
    """Summarise older messages, keeping the last 5 intact.

    Mirrors TS `summarizeMessages`. If there are 10 or fewer messages, the
    list is returned unchanged.
    """
    if len(messages) <= 10:
        return messages

    recent = messages[-5:]
    old = messages[:-5]
    print(
        f"📊 Summarizing {len(old)} old messages, keeping {len(recent)} "
        f"recent messages intact"
    )
    try:
        summarised_old = await asyncio.gather(
            *(summarize_message(m, api_key) for m in old)
        )
        return [*summarised_old, *recent]
    except Exception as error:  # noqa: BLE001
        print(f"Error summarizing messages: {error}")
    return recent


# ---------------------------------------------------------------------------
# Entity helpers (duplicated from chat_planner_service for parity with TS)
# ---------------------------------------------------------------------------


_QUOTED_RE = re.compile(r"""['"]([^'"]+)['"]""")
_VERB_NOUN_RE = re.compile(
    r"\b(?:add|remove|delete|drop|clear)\s+([A-Za-z0-9_-]+)", re.IGNORECASE
)


def detect_entity_name(refined_query: str) -> Optional[str]:
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
    last = parts[-1]
    return last if len(last) > 1 else None


def deep_clone(obj: Any) -> Any:
    if obj is None:
        return {}
    return json.loads(json.dumps(obj))


def replace_in_object(obj: Any, search: str, replacement: str) -> None:
    if not isinstance(obj, dict):
        return
    pattern = re.compile(re.escape(search), re.IGNORECASE)
    for key in list(obj.keys()):
        v = obj[key]
        if isinstance(v, str):
            obj[key] = pattern.sub(replacement, v)
        elif isinstance(v, dict):
            replace_in_object(v, search, replacement)


__all__ = [
    "serialize_useful_data_in_order",
    "estimate_tokens",
    "summarize_message",
    "summarize_messages",
    "filter_plan_messages",
    "detect_entity_name",
    "deep_clone",
    "replace_in_object",
]
