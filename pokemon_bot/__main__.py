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
import os
import sys
from typing import Any, Optional

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


# ---------------------------------------------------------------------------
# ElasticDash observability — start a session at REPL boot, wrap every turn
# in start_trace / end_trace so tools + LLM calls share a trace_id.
# ---------------------------------------------------------------------------


async def _maybe_init_observability() -> Any:
    """Start ElasticDash observability if ELASTICDASH_SERVER_URL is set.

    Returns an `ObservabilityHandle` on success, or `None` if either the
    SDK is unavailable or the env var is unset. Failures are downgraded
    to a warning — the REPL must remain usable without telemetry.
    """
    server_url = os.environ.get("ELASTICDASH_SERVER_URL", "").strip()
    if not server_url:
        return None
    try:
        from elasticdash_sdk.observability import (
            init_observability,
            ObservabilityOptions,
        )
    except ImportError:
        write_status("[observability] elasticdash-sdk not installed — tracing disabled")
        return None
    try:
        handle = await init_observability(
            ObservabilityOptions(
                server_url=server_url,
                api_key=os.environ.get("ELASTICDASH_API_KEY") or None,
            )
        )
        write_status(f"[observability] session={handle.session_id}")
        return handle
    except Exception as exc:  # noqa: BLE001
        write_status(f"[observability] init failed: {exc} — continuing without tracing")
        return None


def _start_trace_safe(name: str) -> None:
    try:
        from elasticdash_sdk.observability import start_trace
    except ImportError:
        return
    start_trace(name)


def _end_trace_safe() -> None:
    try:
        from elasticdash_sdk.observability import end_trace
    except ImportError:
        return
    end_trace()


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

    obs_handle = await _maybe_init_observability()

    try:
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
            _start_trace_safe("pokemon_chat")
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
                _end_trace_safe()
                continue
            finally:
                # Close the trace even on the happy path so the next turn
                # gets a fresh trace_id rather than re-using the previous.
                # `end_trace` is idempotent — calling it after the except
                # branch above is harmless.
                _end_trace_safe()
            write_answer_footer()

            assistant_text = (result or {}).get("message", "") if isinstance(result, dict) else ""
            if assistant_text:
                history.append(ChatMessage(role="assistant", content=assistant_text))
            if isinstance(result, dict) and result.get("sessionId"):
                session_id = result["sessionId"]
    finally:
        if obs_handle is not None:
            try:
                await obs_handle.shutdown()
            except Exception as exc:  # noqa: BLE001
                write_status(f"[observability] shutdown failed: {exc}")

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
