"""Answering an agent's permission request by voice or keyboard.

:func:`parse_answer` turns what the user said into allow / always / reject,
or ``None`` when it can't tell. Anything negative wins ("no, don't do it"
is a no), so a misheard answer can only decline.

:class:`KeyListener` reads single keys (y / n / a) from the terminal while
a question is open, without Enter. It discards keys typed before the
question, so nothing pressed earlier can answer it, and always puts the
terminal back as it found it.
"""

from __future__ import annotations

import os
import re
import select
import sys
import threading
from typing import Callable, TextIO

from macvoice.acp import ALLOW, ALWAYS, REJECT

_NO = {"no", "nope", "nah", "deny", "don't", "dont", "stop", "skip", "cancel", "reject",
       "not", "never", "negative", "wait"}
_ALWAYS = {"always"}
_YES = {"yes", "yeah", "yep", "yup", "sure", "allow", "ok", "okay", "approve", "approved",
        "fine", "affirmative", "proceed", "absolutely", "definitely"}
_YES_PHRASES = ("go ahead", "do it", "go for it", "sounds good")

KEYS = {"y": ALLOW, "a": ALWAYS, "n": REJECT}


def parse_answer(text: str) -> str | None:
    """``allow``, ``always``, ``reject``, or ``None`` if unclear."""
    t = text.lower().replace("’", "'")
    words = set(re.findall(r"[a-z']+", t))
    if words & _NO:
        return REJECT
    if words & _ALWAYS:
        return ALWAYS
    if words & _YES or any(p in t for p in _YES_PHRASES):
        return ALLOW
    return None


class KeyListener:
    """Single keystrokes from a terminal, on a background thread, between
    :meth:`start` and :meth:`stop`. Does nothing unless ``stream`` is a
    terminal."""

    def __init__(self, stream: TextIO = sys.stdin) -> None:
        try:
            self.fd = stream.fileno()
            self.enabled = os.isatty(self.fd)
        except (AttributeError, OSError, ValueError):
            self.fd, self.enabled = -1, False
        self._saved = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self, on_key: Callable[[str], None]) -> None:
        if not self.enabled or self._thread is not None:
            return
        import termios
        import tty

        self._saved = termios.tcgetattr(self.fd)
        termios.tcflush(self.fd, termios.TCIFLUSH)  # keys typed before the question don't count
        # One key at a time, no echo; Ctrl-C still interrupts. TCSANOW: the
        # default waits for pending output to drain, which can hang.
        tty.setcbreak(self.fd, termios.TCSANOW)
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, args=(on_key,), name="macvoice-keys",
                                        daemon=True)
        self._thread.start()

    def _loop(self, on_key: Callable[[str], None]) -> None:
        while not self._stop.is_set():
            if select.select([self.fd], [], [], 0.1)[0]:
                key = os.read(self.fd, 1).decode("utf-8", "replace").lower()
                if key:
                    on_key(key)

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=1)
        self._thread = None
        if self._saved is not None:
            import termios

            termios.tcsetattr(self.fd, termios.TCSANOW, self._saved)
            self._saved = None
