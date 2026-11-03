"""The interactive console for ``macvoice run``: type or talk, one conversation.

An input box at the bottom (prompt_toolkit) with the live status line under
it; the conversation scrolls above, replies rendered as markdown (rich).
Typed prompts and spoken ones go to the same brain, in order.

Keys, as in Claude Code:

- Enter sends. Shift+Enter, Option+Enter, Ctrl+J, or ``\\`` then Enter
  starts a new line. (Shift+Enter needs a terminal that reports it, such as
  Ghostty, kitty, WezTerm or iTerm2: see :data:`KITTY_ON`.)
- Up and Down walk through earlier prompts (saved across runs); Ctrl+R
  searches them.
- Ctrl+C clears what you've typed, or stops the reply, or (pressed twice)
  quits. Esc also stops the reply. Ctrl+D on an empty line quits.
- y / n / a answer a permission question when the input is empty.
"""

from __future__ import annotations

import shutil
import sys
import time
from pathlib import Path

from prompt_toolkit import PromptSession
from prompt_toolkit.application import get_app
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.filters import Condition
from prompt_toolkit.formatted_text import ANSI, HTML
from prompt_toolkit.history import FileHistory, InMemoryHistory
from prompt_toolkit.input.ansi_escape_sequences import ANSI_SEQUENCES
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.keys import Keys
from prompt_toolkit.patch_stdout import patch_stdout
from prompt_toolkit.styles import Style

HISTORY = Path.home() / ".cache" / "macvoice" / "history"

COMMANDS = {
    "/help": "these commands and keys",
    "/mute": "replies as text only (again to speak)",
    "/modes": "the agent's modes",
    "/mode": "switch mode: /mode acceptEdits",
    "/model": "the agent's models, or switch: /model sonnet",
    "/effort": "how hard it thinks, or switch: /effort low",
    "/quit": "leave",
}

HELP = """\
  Enter send                Shift/Option+Enter, Ctrl+J or \\ Enter  new line
  Up/Down earlier prompts   Ctrl+R search history
  Ctrl+C clear / stop / quit (twice)    Esc stop the reply    Ctrl+D quit
  y / n / a  answer a permission question
""" + "".join(f"  {name:<8} {what}\n" for name, what in COMMANDS.items())

# The kitty keyboard protocol, "disambiguate" level: terminals that support
# it (Ghostty, kitty, WezTerm, iTerm2, foot, ...) then report Shift+Enter
# differently from Enter. Others ignore the request. It also changes how
# Ctrl+letter and Esc arrive, so those are taught to prompt_toolkit below.
KITTY_ON, KITTY_OFF = "\x1b[>1u", "\x1b[<u"
_NEWLINE = (Keys.Escape, Keys.ControlM)  # what Option+Enter sends; bound to "new line"


def _teach_kitty_keys() -> None:
    for mod in (2, 3, 5):                         # Shift, Alt, Ctrl + Enter
        ANSI_SEQUENCES[f"\x1b[13;{mod}u"] = _NEWLINE
    ANSI_SEQUENCES["\x1b[27u"] = Keys.Escape
    ANSI_SEQUENCES["\x1b[127;3u"] = (Keys.Escape, Keys.ControlH)   # Option+Backspace
    for c in "abcdefghijklmnopqrstuvwxyz":
        ANSI_SEQUENCES[f"\x1b[{ord(c)};5u"] = getattr(Keys, f"Control{c.upper()}")


_teach_kitty_keys()


class AnswerKeys:
    """Lets :meth:`VoiceAgent.ask` take y / n / a from the console."""

    def __init__(self) -> None:
        self.on_key = None

    def start(self, on_key) -> None:
        self.on_key = on_key

    def stop(self) -> None:
        self.on_key = None

    @property
    def active(self) -> bool:
        return self.on_key is not None


class _SlashCommands(Completer):
    def get_completions(self, document, complete_event):
        text = document.text_before_cursor
        if not text.startswith("/") or " " in text:
            return
        for name, what in COMMANDS.items():
            if name.startswith(text):
                yield Completion(name, start_position=-len(text), display_meta=what)


class Console:
    def __init__(self, agent, *, brain=None, status=None, history_path: Path | None = HISTORY,
                 input=None, output=None, kitty: bool = True) -> None:
        self.agent, self.brain, self.status = agent, brain, status
        self.keys = AnswerKeys()
        self.kitty = kitty
        self._last_ctrl_c = 0.0
        if history_path is not None:
            history_path.parent.mkdir(parents=True, exist_ok=True)
            history = FileHistory(str(history_path))
        else:
            history = InMemoryHistory()
        self.session = PromptSession(
            message=HTML("<ansicyan><b>›</b></ansicyan> "),
            prompt_continuation="  ",
            multiline=True,
            key_bindings=self._bindings(),
            history=history,
            completer=_SlashCommands(),
            complete_while_typing=True,
            placeholder=HTML("<style fg='#7a7a7a'>Type or talk. Enter sends, Shift+Enter for a new "
                             "line, Ctrl+C stops. /help</style>"),
            bottom_toolbar=self._toolbar if status is not None else None,
            refresh_interval=1 / 15 if status is not None else 0,
            style=Style.from_dict({"bottom-toolbar": "noreverse"}),
            input=input, output=output,
        )

    # ── keys ──────────────────────────────────────────────────────────

    def _bindings(self) -> KeyBindings:
        kb = KeyBindings()
        answering = Condition(lambda: self.keys.active and not get_app().current_buffer.text)

        for key in "yna":
            @kb.add(key, filter=answering)
            def _(event, key=key):
                if self.keys.on_key is not None:
                    self.keys.on_key(key)

        @kb.add("enter")
        def _(event):
            buf = event.current_buffer
            if buf.document.text_before_cursor.endswith("\\"):   # "\" then Enter: a new line
                buf.delete_before_cursor(1)
                buf.insert_text("\n")
            elif buf.text.strip():
                buf.validate_and_handle()

        @kb.add(*_NEWLINE)
        @kb.add("c-j")
        def _(event):
            event.current_buffer.insert_text("\n")

        @kb.add("c-c")
        def _(event):
            buf = event.current_buffer
            if buf.text:
                buf.reset()
            elif self.agent.interrupt():
                self._last_ctrl_c = 0.0
                print("  (stopped)")
            elif time.monotonic() - self._last_ctrl_c < 2.0:
                event.app.exit(exception=EOFError())
            else:
                self._last_ctrl_c = time.monotonic()
                print("  (press Ctrl-C again to quit)")

        @kb.add("escape", filter=Condition(lambda: self.agent.busy))
        def _(event):
            if self.agent.interrupt():
                print("  (stopped)")

        @kb.add("c-d", filter=Condition(lambda: not get_app().current_buffer.text))
        def _(event):
            event.app.exit(exception=EOFError())

        return kb

    def _toolbar(self):
        return ANSI(self.status.line())

    # ── input ─────────────────────────────────────────────────────────

    def handle(self, text: str) -> bool:
        """Act on one entry. False to quit."""
        text = text.strip()
        if not text:
            return True
        if text.startswith("/"):
            return self._command(text)
        busy = self.agent.busy
        self.agent.submit_text(text)
        if busy:
            print("  (queued: sends when the current reply finishes)")
        return True

    def _command(self, text: str) -> bool:
        name, _, arg = text.partition(" ")
        arg = arg.strip()
        responder = self.agent.responder
        if name == "/quit":
            return False
        if name == "/help":
            print(HELP, end="")
        elif name == "/mute":
            responder.speak = not responder.speak
            if not responder.speak:
                self.agent.out.stop()
            print("  speech on" if responder.speak else "  speech off: replies are text only")
        elif name in ("/mode", "/modes"):
            modes = getattr(self.brain, "modes", None)
            if not modes:
                print("  this brain has no modes")
            elif name == "/modes" or not arg:
                for m in modes:
                    mark = "*" if m.get("id") == self.brain.mode else " "
                    print(f"  {mark} {m.get('id')}  {m.get('description', '')}")
            elif self.agent.busy:
                print("  wait for the reply to finish, then switch modes")
            else:
                try:
                    print(f"  mode: {self.brain.set_mode(arg)}")
                except Exception as e:  # noqa: BLE001 - ACPError names the choices
                    print(f"  {e}")
        elif name in ("/model", "/effort"):
            self._setting(name[1:], arg)
        else:
            print(f"  unknown command {name} (/help)")
        return True

    def _setting(self, name: str, arg: str) -> None:
        option = self.brain.setting(name) if hasattr(self.brain, "setting") else None
        if option is None:
            print(f"  this brain has no {name} setting")
        elif not arg:
            for c in option.get("options") or []:
                mark = "*" if c.get("value") == option.get("currentValue") else " "
                label = c.get("name") if c.get("name") != c.get("value") else ""
                print(f"  {mark} {c.get('value')}  {label}".rstrip())
        elif self.agent.busy:
            print(f"  wait for the reply to finish, then switch {name}")
        else:
            try:
                print(f"  {name}: {self.brain.set_setting(name, arg)}")
            except Exception as e:  # noqa: BLE001 - ACPError names the choices
                print(f"  {e}")

    # ── output ────────────────────────────────────────────────────────

    def print_turn(self, m) -> None:
        from rich.console import Console as RichConsole
        from rich.markdown import Markdown
        from rich.padding import Padding

        from macvoice.status import timing

        if not m.typed:
            print(f"\033[2m  you : {m.transcript}\033[0m")
        text = m.text or m.reply
        if text:
            # Created per call: under patch_stdout, sys.stdout is a proxy.
            rich = RichConsole(file=sys.stdout, force_terminal=True, highlight=False,
                               width=shutil.get_terminal_size((80, 24)).columns)
            rich.print(Padding(Markdown(text), (0, 0, 0, 2)))
        if m.cancelled:
            print("\033[2m  (stopped)\033[0m")
        elif m.interrupted:
            print("\033[2m  (talked over: the rest wasn't spoken)\033[0m")
        print(f"\033[2m  {timing(m)}\033[0m")

    def activity(self, line: str) -> None:
        print(f"\033[2m{line}\033[0m")

    # ── loop ──────────────────────────────────────────────────────────

    def run(self) -> None:
        """Until /quit, Ctrl+D or Ctrl+C twice."""
        term = sys.__stdout__
        kitty = self.kitty and term is not None and term.isatty()
        if kitty:
            term.write(KITTY_ON)
            term.flush()
        try:
            with patch_stdout(raw=True):
                while True:
                    try:
                        text = self.session.prompt()
                    except EOFError:
                        break
                    except KeyboardInterrupt:
                        continue
                    if not self.handle(text):
                        break
        finally:
            if kitty:
                term.write(KITTY_OFF)
                term.flush()
