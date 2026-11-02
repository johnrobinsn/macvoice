"""The type-or-talk console: keys, commands and output, driven through a
pipe (no terminal)."""

import pytest
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from macvoice.agent import TurnMetrics
from macvoice.console import Console


class FakeOut:
    def __init__(self):
        self.stopped = 0

    def stop(self):
        self.stopped += 1


class FakeResponder:
    speak = True


class FakeAgent:
    def __init__(self, busy=False):
        self.busy, self.submitted, self.interrupts = busy, [], 0
        self.responder, self.out = FakeResponder(), FakeOut()

    def submit_text(self, text):
        self.submitted.append(text)

    def interrupt(self):
        self.interrupts += 1
        return self.busy


class FakeBrain:
    modes = [{"id": "default", "description": "ask first"}, {"id": "plan", "description": "plan only"}]
    mode = "default"

    def set_mode(self, name):
        if name not in ("default", "plan"):
            raise RuntimeError(f"no mode {name!r}. Available: default, plan")
        self.mode = name
        return name


@pytest.fixture
def pipe():
    with create_pipe_input() as inp:
        yield inp


def console(pipe, agent=None, **kw):
    return Console(agent or FakeAgent(), history_path=None, input=pipe, output=DummyOutput(),
                   kitty=False, **kw)


def typed(pipe, keys, agent=None):
    c = console(pipe, agent)
    pipe.send_text(keys)
    return c.session.prompt()


@pytest.mark.parametrize("keys, text", [
    ("hello\r", "hello"),
    ("\rhello\r", "hello"),                       # Enter on an empty line does nothing
    ("one\\\rtwo\r", "one\ntwo"),                 # backslash, Enter
    ("one\x1b\rtwo\r", "one\ntwo"),               # Option+Enter
    ("one\ntwo\r", "one\ntwo"),                   # Ctrl+J
    ("one\x1b[13;2utwo\r", "one\ntwo"),           # Shift+Enter (kitty keyboard protocol)
    ("junk\x03kept\r", "kept"),                   # Ctrl+C clears the line
    ("junk\x1b[99;5ukept\r", "kept"),             # ... and as the kitty protocol sends it
    ("yes\r", "yes"),                             # y is just a letter unless a question is open
])
def test_editing_keys(pipe, keys, text):
    assert typed(pipe, keys) == text


def test_up_arrow_recalls_earlier_prompts(pipe):
    c = console(pipe)
    pipe.send_text("first\r")
    assert c.session.prompt() == "first"
    pipe.send_text("\x1b[A\r")
    assert c.session.prompt() == "first"


def test_ctrl_c_stops_a_reply(pipe, capsys):
    agent = FakeAgent(busy=True)
    assert typed(pipe, "\x03next\r", agent) == "next"
    assert agent.interrupts == 1 and "(stopped)" in capsys.readouterr().out


def test_ctrl_c_twice_quits_when_idle(pipe, capsys):
    with pytest.raises(EOFError):
        typed(pipe, "\x03\x03")
    assert "press Ctrl-C again to quit" in capsys.readouterr().out


def test_ctrl_d_on_an_empty_line_quits(pipe):
    with pytest.raises(EOFError):
        typed(pipe, "\x04")


def test_keys_answer_an_open_question(pipe):
    c = console(pipe)
    answers = []
    c.keys.start(answers.append)
    pipe.send_text("y\x04")
    with pytest.raises(EOFError):                 # "y" answered; the line stayed empty
        c.session.prompt()
    assert answers == ["y"]


def test_typed_prompts_go_to_the_agent(pipe, capsys):
    agent = FakeAgent()
    c = console(pipe, agent)
    assert c.handle("  make a site  ") and agent.submitted == ["make a site"]
    agent.busy = True
    c.handle("then add a footer")
    assert "queued" in capsys.readouterr().out
    assert c.handle("   ") and agent.submitted == ["make a site", "then add a footer"]


def test_commands(pipe, capsys):
    agent, brain = FakeAgent(), FakeBrain()
    c = console(pipe, agent, brain=brain)
    c.handle("/mute")
    assert agent.responder.speak is False and agent.out.stopped == 1
    c.handle("/mute")
    assert agent.responder.speak is True
    c.handle("/modes")
    c.handle("/mode plan")
    c.handle("/mode yolo")
    out = capsys.readouterr().out
    assert "* default  ask first" in out and "mode: plan" in out and "no mode 'yolo'" in out
    agent.busy = True
    c.handle("/mode default")
    assert "wait for the reply" in capsys.readouterr().out and brain.mode == "plan"
    c.handle("/help")
    assert "Shift/Option+Enter" in capsys.readouterr().out
    c.handle("/nope")
    assert "unknown command" in capsys.readouterr().out
    assert c.handle("/quit") is False


def test_no_modes_without_an_acp_brain(pipe, capsys):
    console(pipe).handle("/modes")
    assert "no modes" in capsys.readouterr().out


def test_slash_commands_complete(pipe):
    from prompt_toolkit.document import Document
    c = console(pipe)
    names = [x.text for x in c.session.completer.get_completions(Document("/mo"), None)]
    assert names == ["/modes", "/mode"]
    assert list(c.session.completer.get_completions(Document("hello /mo"), None)) == []


def test_turns_render_markdown(pipe, capsys):
    c = console(pipe)
    m = TurnMetrics(transcript="what changed", text="I fixed **two** bugs:\n\n- one\n- two",
                    reply="I fixed two bugs: one two", llm_first_sentence_ms=900, tts_first_audio_ms=5)
    c.print_turn(m)
    out = capsys.readouterr().out
    assert "you : what changed" in out
    assert "two" in out and "**" not in out and "•" in out      # rendered, not raw
    assert "llm 900 ms" in out


def test_typed_turns_skip_the_echo_and_voice_timings(pipe, capsys):
    c = console(pipe)
    c.print_turn(TurnMetrics(transcript="hi", text="Hello.", typed=True, cancelled=True,
                             llm_first_sentence_ms=800, tts_first_audio_ms=5))
    out = capsys.readouterr().out
    assert "you :" not in out and "endpoint" not in out and "(stopped)" in out
    assert "after you pressed Enter" in out


def test_a_reply_you_talked_over_says_so(pipe, capsys):
    console(pipe).print_turn(TurnMetrics(transcript="hi", text="Hello. More.", interrupted=True))
    out = capsys.readouterr().out
    assert "More." in out and "talked over" in out and "(stopped)" not in out
