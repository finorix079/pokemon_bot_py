"""chat/session.py — in-memory pending-plan store + session ID generator.

Ported from open-react-template/app/api/chat/session.ts.

The CLI port keeps the `pending_plans` map for completeness even though the
REPL skips the write-op approval gate. The 5-minute cleanup interval from
the TS source is omitted (the REPL process is short-lived and single-user);
callers that need long-running cleanup can call `cleanup_expired_plans()`
manually.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Literal, Optional


@dataclass
class PendingPlan:
    """One entry in `pending_plans`. Mirrors TS pendingPlans value shape."""

    plan: dict[str, Any]
    planResponse: str
    refinedQuery: str
    topKResults: list[Any] = field(default_factory=list)
    conversationContext: str = ""
    finalDeliverable: str = ""
    entities: list[Any] = field(default_factory=list)
    intentType: Literal["FETCH", "MODIFY"] = "FETCH"
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))
    referenceTask: Optional[Any] = None


# Module-level store. Keyed by sessionId.
pending_plans: dict[str, PendingPlan] = {}


def cleanup_expired_plans(max_age_ms: int = 3_600_000) -> int:
    """Remove pending plans older than `max_age_ms`. Returns the count removed.

    Mirrors the `setInterval` cleanup in TS session.ts:23-30. Called on
    demand instead of periodically because the CLI is short-lived.
    """
    now_ms = int(time.time() * 1000)
    expired = [sid for sid, data in pending_plans.items() if now_ms - data.timestamp > max_age_ms]
    for sid in expired:
        pending_plans.pop(sid, None)
    return len(expired)


def generate_session_id(messages: list[Any]) -> str:
    """Generate a stable session ID from conversation messages.

    Ported from TS `generateSessionId`. Uses all messages except the last
    so the ID survives an approval-style follow-up. The hash is the same
    DJB2-like bitwise hash as the TS version, kept byte-for-byte for parity
    (truncated to int32 via `& 0xFFFFFFFF` then mapped to signed int32).
    """
    messages_for_hash = messages[:-1] if messages else []
    if not messages_for_hash:
        return f"session_new_{int(time.time() * 1000)}"

    keep_first = messages_for_hash[: min(3, len(messages_for_hash))]
    key_messages = list(keep_first)
    last = messages_for_hash[-1]
    if last is not None and last not in keep_first:
        key_messages.append(last)
    key_messages = [m for m in key_messages if m is not None]

    def _role(m: Any) -> str:
        return getattr(m, "role", None) or (m.get("role") if isinstance(m, dict) else "")

    def _content(m: Any) -> str:
        return getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else "")

    content = "|".join(f"{_role(m)}:{_content(m)}" for m in key_messages)

    hash_value = 0
    for ch in content:
        char_code = ord(ch)
        # Mirror JS `((hash << 5) - hash) + char` then `hash & hash` (no-op cast to int32).
        hash_value = ((hash_value << 5) - hash_value) + char_code
        # Force into signed 32-bit range to match JS behaviour.
        hash_value &= 0xFFFFFFFF
        if hash_value >= 0x80000000:
            hash_value -= 0x100000000

    return f"session_{abs(hash_value)}"


__all__ = [
    "PendingPlan",
    "pending_plans",
    "cleanup_expired_plans",
    "generate_session_id",
]
