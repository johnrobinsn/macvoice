"""A live status line for ``macvoice run``.

One line at the bottom of the terminal, redrawn ~15 times a second: what
the agent is doing and how loud the microphone is.

    ● hearing you  ▕██████████▍           ▏
    ◐ thinking 0.4s
    ▶ speaking     (talk to interrupt)

Turn summaries print above it (:meth:`StatusLine.print`). It only draws on
a terminal; piped output gets the summaries alone.
"""

from __future__ import annotations

import os
import shutil
import sys
import threading
import time
from typing import Callable, TextIO

FLOOR_DB, CEIL_DB = -70.0, -10.0  # meter range, dBFS: room tone to loud speech
METER_CELLS = 24
_PARTIAL = " ▏▎▍▌▋▊▉"

# state -> (icon, label, ANSI color)
STATES = {
    "listening": ("○", "listening", "2"),
    "hearing": ("●", "hearing you", "32"),
    "thinking": ("◐", "thinking", "33"),
    "speaking": ("▶", "speaking", "36"),
}


def meter(level_db: float, cells: int = METER_CELLS) -> str:
    """A horizontal bar for a level in dBFS, in eighths of a cell."""
    frac = min(1.0, max(0.0, (level_db - FLOOR_DB) / (CEIL_DB - FLOOR_DB)))
    eighths = round(frac * cells * 8)
    full, part = divmod(eighths, 8)
    bar = "█" * full + (_PARTIAL[part] if part else "")
    return "▕" + bar.ljust(cells) + "▏"


def render(state: str, level_db: float, *, elapsed_s: float = 0.0, barge_in: bool = False,
           color: bool = False, width: int = 80) -> str:
    """The status line's text, at most ``width - 1`` visible characters."""
    icon, label, ansi = STATES[state]
    if state == "thinking":
        label = f"{label} {elapsed_s:.1f}s"
    if state == "speaking":
        tail = "(talk to interrupt)" if barge_in else "(mic muted while I talk)"
    elif state == "thinking":
        tail = ""
    else:
        tail = meter(level_db)
    text = f"{icon} {label:<12} {tail}".rstrip()[: max(0, width - 1)]
    if not color:
        return text
    head = len(icon) + 1 + max(len(label), 12)
    return f"\033[{ansi}m{text[:head]}\033[0m{text[head:]}"


class StatusLine:
    """Draws :func:`render` for a live agent on a background thread.

    ``source`` returns ``(state, level_db)``; the meter rises at once and
    falls at ``decay_db_s`` so speech reads as a smooth envelope rather
    than a flicker. Does nothing unless ``stream`` is a terminal."""

    def __init__(self, source: Callable[[], tuple[str, float]], *, barge_in: bool = False,
                 stream: TextIO = sys.stdout, fps: float = 15, decay_db_s: float = 40,
                 enabled: bool | None = None) -> None:
        self.source, self.barge_in, self.stream = source, barge_in, stream
        self.enabled = stream.isatty() if enabled is None else enabled
        self.color = self.enabled and "NO_COLOR" not in os.environ
        self._period = 1 / fps
        self._decay = decay_db_s
        self._level = FLOOR_DB
        self._state = "listening"
        self._since = time.monotonic()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> "StatusLine":
        if self.enabled:
            self._thread = threading.Thread(target=self._loop, name="macvoice-status", daemon=True)
            self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1)
            with self._lock:
                self._write("\r\033[K")

    def print(self, text: str) -> None:
        """Print ``text`` above the status line."""
        with self._lock:
            if self.enabled:
                self._write("\r\033[K")
            self._write(text + "\n")
            if self.enabled and not self._stop.is_set():
                self._draw()

    def line(self, now: float | None = None) -> str:
        """Advance the meter and state to ``now`` and return the line."""
        now = time.monotonic() if now is None else now
        state, level = self.source()
        if state != self._state:
            self._state, self._since = state, now
        self._level = max(level, self._level - self._decay * self._period)
        width = shutil.get_terminal_size((80, 24)).columns
        return render(state, self._level, elapsed_s=now - self._since, barge_in=self.barge_in,
                      color=self.color, width=width)

    def _draw(self) -> None:
        self._write("\r\033[K" + self.line())

    def _write(self, s: str) -> None:
        self.stream.write(s)
        self.stream.flush()

    def _loop(self) -> None:
        while not self._stop.wait(self._period):
            with self._lock:
                self._draw()
