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
import contextlib
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, Optional

from . import config
from .schemas.ai import ChatMessage, ChatStreamRequest
from .chat.handler import run_chat_pipeline
from .utils.cli_stream import (
    real_stdout,
    write_answer_footer,
    write_answer_header,
    write_error,
    write_result,
    write_status,
    write_token,
)


# `cli_stream` captured the real terminal stdout at import time. Use it
# for every terminal-bound write so the log-file redirect below doesn't
# swallow banner/prompt/echo bytes.
_TERMINAL_STDOUT = real_stdout()


@contextlib.contextmanager
def _redirect_pipeline_output_to_log() -> Iterator[Path]:
    """Redirect `sys.stdout`/`sys.stderr` to `.temp/logs/pokemon_bot-<ts>.log`.

    The pipeline emits a lot of diagnostic `print()` lines (planner steps,
    executor state, query refinement, etc.). On the REPL these collide
    with the prompt — a bare `\\n` from cooked mode lands while the
    terminal is in cbreak/raw for `_read_line_raw`, producing a staircase
    that overlays the input line. Sending those prints to a file removes
    the collision and keeps the noise reviewable.

    User-facing output (banner, prompt, status, streamed answer) uses
    `_TERMINAL_STDOUT` so it bypasses the redirect.
    """
    log_dir = Path(".temp/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = log_dir / f"pokemon_bot-{timestamp}.log"
    log_file = log_path.open("w", buffering=1, encoding="utf-8")
    original_stdout, original_stderr = sys.stdout, sys.stderr
    sys.stdout = log_file
    sys.stderr = log_file
    try:
        yield log_path
    finally:
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_file.close()


BANNER = """\
\x1b[36mpokemon-bot\x1b[0m — interactive PokéAPI assistant
Type a question and press Enter. Use \x1b[33m/exit\x1b[0m, \x1b[33mEsc\x1b[0m, or Ctrl+D to quit.
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
        write_status("[sdk] ELASTICDASH_SERVER_URL not set — tracing disabled")
        return None
    try:
        from elasticdash_sdk.observability import (
            init_observability,
            ObservabilityOptions,
        )
    except ImportError:
        write_status("[sdk] elasticdash-sdk not installed — tracing disabled")
        return None
    write_status(f"[sdk] init_observability → {server_url}")
    try:
        handle = await init_observability(
            ObservabilityOptions(
                server_url=server_url,
                api_key=os.environ.get("ELASTICDASH_API_KEY") or None,
            )
        )
        write_status(f"[sdk] observability session={handle.session_id}")
        return handle
    except Exception as exc:  # noqa: BLE001
        write_status(f"[sdk] init_observability failed: {exc} — continuing without tracing")
        return None


def _start_trace_safe(name: str) -> None:
    try:
        from elasticdash_sdk.observability import start_trace
    except ImportError:
        return
    write_status(f"[sdk] start_trace({name})")
    start_trace(name)


def _end_trace_safe() -> None:
    try:
        from elasticdash_sdk.observability import end_trace
    except ImportError:
        return
    write_status("[sdk] end_trace")
    end_trace()


_ESC = "\x1b"


def _read_line(prompt: str) -> str | None:
    """Read one line from stdin. Returns `None` on EOF / Ctrl+D / Esc.

    When stdin is a TTY, puts the terminal in raw mode so a bare Esc
    keypress exits immediately. Falls back to plain `input()` for piped
    input (tests, scripts) where raw mode isn't available.
    """
    if not sys.stdin.isatty():
        try:
            return input(prompt)
        except EOFError:
            return None
    return _read_line_raw(prompt)


def _read_line_raw(prompt: str) -> str | None:
    """Raw-mode line reader. Returns `None` on Esc, Ctrl+D (at empty line), or EOF."""
    import termios
    import tty

    out = _TERMINAL_STDOUT
    out.write(prompt)
    out.flush()

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    buf: list[str] = []
    try:
        tty.setraw(fd)
        while True:
            ch = sys.stdin.read(1)
            if not ch:
                return None
            if ch == _ESC:
                out.write("\r\n")
                out.flush()
                return None
            if ch in ("\r", "\n"):
                out.write("\r\n")
                out.flush()
                return "".join(buf)
            if ch == "\x03":  # Ctrl+C — let the outer KeyboardInterrupt path handle it
                raise KeyboardInterrupt
            if ch == "\x04":  # Ctrl+D
                if buf:
                    continue
                out.write("\r\n")
                out.flush()
                return None
            if ch in ("\x7f", "\x08"):  # Backspace / DEL
                if buf:
                    buf.pop()
                    out.write("\b \b")
                    out.flush()
                continue
            if ch < " ":  # other control chars — ignore
                continue
            buf.append(ch)
            out.write(ch)
            out.flush()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


async def _run_repl(args: argparse.Namespace) -> int:
    try:
        config.load()
    except RuntimeError as exc:
        write_error(str(exc))
        return 2

    history: list[ChatMessage] = []
    session_id: str | None = None

    _TERMINAL_STDOUT.write(BANNER + "\n")
    _TERMINAL_STDOUT.flush()

    obs_handle = await _maybe_init_observability()

    try:
        while True:
            # Run the blocking read in a worker thread so the asyncio loop
            # stays alive between turns. Without this, the SDK's heartbeat,
            # periodic flush, and retry chains all stall while we wait for
            # the user — which is what was leaving turn N's buffer mixed
            # into turn N+1's batch and letting engineio abort the socket
            # during idle.
            user_text = await asyncio.to_thread(
                _read_line, "\x1b[32myou\x1b[0m ❯ "
            )
            if user_text is None:
                _TERMINAL_STDOUT.write("\n")
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
    parser.add_argument(
        "--sdk-debug",
        action="store_true",
        help="Enable verbose elasticdash-sdk internal logs (sets ELASTICDASH_DEBUG=1).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.sdk_debug:
        os.environ["ELASTICDASH_DEBUG"] = "1"
    try:
        with _redirect_pipeline_output_to_log() as log_path:
            write_status(f"Pipeline logs → {log_path}")
            return asyncio.run(_run_repl(args))
    except KeyboardInterrupt:
        _TERMINAL_STDOUT.write("\n")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
