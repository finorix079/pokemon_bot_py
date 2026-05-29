"""utils/cli_stream.py — terminal status + token writers for the REPL.

Replaces the Vercel AI SDK data-stream wire protocol used by
`app/api/chat-stream/route.ts`. Each helper writes directly to stdout
so the REPL gets:

- `[status] …` lines for each pipeline stage (mirrors the TS
  `2:[{"type":"status",…}]` frames)
- Streamed final-answer tokens (mirrors the TS `0:"<token>"` frames)
- Plan / clarification / error messages on their own lines

No SSE / "v1" wire framing — those only matter for the HTTP client.
"""

from __future__ import annotations

import sys
from typing import TextIO


# Capture the real terminal stdout at import time. `pokemon_bot.__main__`
# redirects `sys.stdout` to a log file for the duration of the REPL so
# stray pipeline `print()`s land in `.temp/logs/` instead of overlaying
# the prompt; the writers below bypass that redirect so banner, status,
# and streamed tokens still reach the user.
_REAL_STDOUT: TextIO = sys.stdout


def real_stdout() -> TextIO:
    """Return the terminal-bound stdout captured at import time."""
    return _REAL_STDOUT


def _write(line: str, *, end: str = "\n", flush: bool = True) -> None:
    _REAL_STDOUT.write(line + end)
    if flush:
        _REAL_STDOUT.flush()


def write_status(message: str) -> None:
    """Emit a one-line pipeline-stage update."""
    _write(f"\x1b[90m[status]\x1b[0m {message}")


def write_token(token: str) -> None:
    """Stream a single token of the final answer to stdout."""
    _REAL_STDOUT.write(token)
    _REAL_STDOUT.flush()


def write_answer_header() -> None:
    """Newline + faint divider before the streamed answer begins."""
    _write("\x1b[90m─" * 60 + "\x1b[0m")


def write_answer_footer() -> None:
    """Newline after the streamed answer completes."""
    _write("")
    _write("\x1b[90m─" * 60 + "\x1b[0m")


def write_result(message: str) -> None:
    """Emit a complete non-streaming pipeline message (e.g. clarification)."""
    _write(f"\x1b[36m[result]\x1b[0m {message}")


def write_plan(message: str) -> None:
    """Emit a plan-proposal message (only seen if the approval gate ever
    fires; preserved for parity even though the CLI skips it)."""
    _write(f"\x1b[33m[plan]\x1b[0m {message}")


def write_error(message: str) -> None:
    _write(f"\x1b[31m[error]\x1b[0m {message}", flush=True)


__all__ = [
    "real_stdout",
    "write_status",
    "write_token",
    "write_answer_header",
    "write_answer_footer",
    "write_result",
    "write_plan",
    "write_error",
]
