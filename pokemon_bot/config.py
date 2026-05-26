"""Configuration / env loading for pokemon_bot.

Mirrors the env-var lookups scattered across:
- app/api/chat-stream/route.ts (ANTHROPIC_API_KEY)
- utils/aiHandler.ts            (KIMI_API_KEY, KIMI_BASE_URL via OpenAI-compatible client)

Loads `.env` from the project root if python-dotenv is available; otherwise
falls back to plain `os.environ` reads.

Migration note (2026-05-26): the chat-completion path was switched from
OpenAI to Anthropic Claude. `OPENAI_API_KEY` is no longer required and
the `AI_PROVIDER` env var has been removed (Claude is the sole provider
for non-Kimi calls). The field is retained on `Config` as an optional
empty string for backward compatibility with anything that might still
read it; the runtime no longer consults it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # pragma: no cover — dotenv is in pyproject, fallback only for trimmed envs
    pass


@dataclass(frozen=True)
class Config:
    """Resolved runtime configuration.

    Required keys raise ``RuntimeError`` if missing; optional keys default
    to the same values the TypeScript project used (modulo the OpenAI →
    Claude migration).
    """

    anthropic_api_key: str
    kimi_api_key: Optional[str]
    kimi_base_url: str
    anthropic_model: str
    kimi_model: str
    # Kept for backward-compat with callers that still read it; runtime
    # ignores its value because OpenAI is no longer used.
    openai_api_key: Optional[str] = None


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(
            f"Missing required environment variable: {name}. "
            f"Copy .env.example to .env and fill it in, or export it in your shell."
        )
    return value


def _optional(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip() or default


def load() -> Config:
    """Resolve the runtime configuration. Raises on missing required keys."""
    return Config(
        anthropic_api_key=_require("ANTHROPIC_API_KEY"),
        kimi_api_key=os.environ.get("KIMI_API_KEY") or None,
        kimi_base_url=_optional("KIMI_BASE_URL", "https://api.moonshot.cn/v1"),
        anthropic_model=_optional("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929"),
        kimi_model=_optional("KIMI_MODEL", "moonshot-v1-32k"),
        openai_api_key=os.environ.get("OPENAI_API_KEY") or None,
    )
