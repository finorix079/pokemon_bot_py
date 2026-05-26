"""pokemon_bot.__main__ — interactive REPL entry point.

Run with:
    python -m pokemon_bot

Reads user messages from stdin one line at a time, keeps an in-memory
conversation history, and pipes each turn through `chat.handler.run_chat_pipeline`.
Pipeline progress is shown as status lines; the final answer is streamed
token-by-token to stdout.

Equivalent to the TS `/api/chat-stream` HTTP route but with no SSE
framing — the REPL is the only consumer.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from typing import Any

from . import config
from .schemas.ai import ChatMessage, ChatStreamRequest
from .chat.handler import run_chat_pipeline
from .utils.cli_stream import (
    write_answer_footer,
    write_answer_header,
    write_error,
    write_result,
    write_status,
    write_token,
)


BANNER = """\
\x1b[36mpokemon-bot\x1b[0m — interactive PokéAPI assistant
Type a question and press Enter. Use \x1b[33m/exit\x1b[0m or Ctrl+D to quit.
"""


def _read_line(prompt: str) -> str | None:
    """Read one line from stdin. Returns `None` on EOF / Ctrl+D."""
    try:
        return input(prompt)
    except EOFError:
        return None


async def _run_repl(args: argparse.Namespace) -> int:
    try:
        config.load()
    except RuntimeError as exc:
        write_error(str(exc))
        return 2

    history: list[ChatMessage] = []
    session_id: str | None = None

    sys.stdout.write(BANNER + "\n")
    sys.stdout.flush()

    while True:
        user_text = _read_line("\x1b[32myou\x1b[0m ❯ ")
        if user_text is None:
            sys.stdout.write("\n")
            break
        user_text = user_text.strip()
        if not user_text:
            continue
        if user_text in ("/exit", "/quit", "/bye"):
            break
        if user_text == "/clear":
            history.clear()
            session_id = None
            write_status("Conversation history cleared.")
            continue

        history.append(ChatMessage(role="user", content=user_text))
        request = ChatStreamRequest(messages=list(history), sessionId=session_id)

        write_answer_header()
        try:
            result = await run_chat_pipeline(
                request,
                user_token=args.user_token or "",
                on_status=write_status,
                on_token=write_token,
                on_result=write_result,
                on_error=write_error,
            )
        except Exception as exc:  # noqa: BLE001
            write_error(f"Pipeline failed: {exc}")
            history.pop()
            continue
        write_answer_footer()

        assistant_text = (result or {}).get("message", "") if isinstance(result, dict) else ""
        if assistant_text:
            history.append(ChatMessage(role="assistant", content=assistant_text))
        if isinstance(result, dict) and result.get("sessionId"):
            session_id = result["sessionId"]

    return 0


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="pokemon-bot",
        description="Interactive PokéAPI chatbot (Python port of /api/chat-stream).",
    )
    parser.add_argument(
        "--user-token",
        default="",
        help="Optional user token forwarded to query refinement (unused for read-only PokéAPI flows).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        return asyncio.run(_run_repl(args))
    except KeyboardInterrupt:
        sys.stdout.write("\n")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
