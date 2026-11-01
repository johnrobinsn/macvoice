import json
import shutil
import sys
from pathlib import Path

import pytest

from macvoice.acp import CODING_BRIEF, ACPAgent, ACPError
from macvoice.llm import SYSTEM_PROMPT, sentences

FAKE = str(Path(__file__).parent / "fake_acp_agent.py")


@pytest.fixture
def agent_log(tmp_path):
    return tmp_path / "acp.jsonl"


def make(agent_log, tmp_path, **kw):
    return ACPAgent([sys.executable, FAKE, str(agent_log)], cwd=str(tmp_path), timeout=10, **kw)


def logged(agent_log):
    return [json.loads(line) for line in agent_log.read_text().splitlines()]


def user(text):
    return [{"role": "user", "content": text}]


def test_handshake_and_streamed_reply(agent_log, tmp_path):
    agent = make(agent_log, tmp_path)
    assert agent.session_id == "s1" and agent.agent_info == {"name": "fake"}
    assert list(sentences(agent.reply(user("capital of France?")))) == \
        ["Paris is the capital.", "Anything else?"]          # thought chunks are not spoken
    init, new, prompt = logged(agent_log)
    assert init["method"] == "initialize" and init["params"]["protocolVersion"] == 1
    assert new["params"] == {"cwd": str(tmp_path), "mcpServers": []}
    assert prompt["params"]["sessionId"] == "s1"
    agent.close()


def prompts(agent_log):
    return [m["params"]["prompt"][0]["text"] for m in logged(agent_log) if m.get("method") == "session/prompt"]


def test_brief_goes_with_the_first_turn_only(agent_log, tmp_path):
    agent = make(agent_log, tmp_path, coding=False)
    list(agent.reply(user("one")))
    list(agent.reply([*user("one"), {"role": "assistant", "content": "x"}, *user("two")]))
    assert prompts(agent_log) == [f"{SYSTEM_PROMPT}\n\none", "two"]   # the agent keeps its own history
    agent.close()


def test_a_project_gets_the_coding_brief(agent_log, tmp_path):
    agent = make(agent_log, tmp_path)                      # cwd given: working in a project
    list(agent.reply(user("make a website")))
    assert prompts(agent_log) == [f"{CODING_BRIEF}\n\nmake a website"]
    agent.close()


def test_tool_permission_declined_by_default(agent_log, tmp_path):
    agent = make(agent_log, tmp_path)
    assert "".join(agent.reply(user("use a tool"))).startswith("Permission no.")
    agent.close()


def test_tool_permission_allowed_on_request(agent_log, tmp_path):
    agent = make(agent_log, tmp_path, allow_tools=True)
    assert "".join(agent.reply(user("use a tool"))).startswith("Permission yes.")
    agent.close()


def test_closing_the_stream_early_cancels_the_turn(agent_log, tmp_path):
    agent = make(agent_log, tmp_path)
    stream = agent.reply(user("be slow"))
    assert next(stream).startswith("word")
    stream.close()                                         # barge-in
    assert any(m.get("method") == "session/cancel" for m in logged(agent_log))
    # the session is usable afterwards
    assert "Paris" in "".join(agent.reply(user("again")))
    agent.close()


def test_agent_exit_raises(agent_log, tmp_path):
    agent = make(agent_log, tmp_path)
    with pytest.raises(ACPError, match="exited"):
        list(agent.reply(user("die")))


@pytest.mark.models
@pytest.mark.skipif(shutil.which("claude-agent-acp") is None, reason="claude-agent-acp not installed")
def test_claude_code_over_acp(tmp_path):
    """The real thing: Claude Code answers a voice-style question."""
    agent = ACPAgent(cwd=str(tmp_path), timeout=120)
    reply = " ".join(sentences(agent.reply(user("What is the capital of France? One short sentence."))))
    agent.close()
    assert "paris" in reply.lower()


# ── permissions ───────────────────────────────────────────────────────

@pytest.mark.parametrize("answer, said", [("allow", "yes"), ("always", "yes"), ("reject", "no"),
                                          (None, "no"), ("mumble", "no")])
def test_permission_asks_and_follows_the_answer(agent_log, tmp_path, answer, said):
    asked = []
    agent = make(agent_log, tmp_path, permissions="ask", ask=lambda req: asked.append(req) or answer)
    assert "".join(agent.reply(user("use a tool"))).startswith(f"Permission {said}.")
    assert asked[0].agent == "The agent" and asked[0].spoken() == "The agent wants to run npm install. Allow?"
    assert "npm install" in asked[0].printed()
    agent.close()


def test_a_failing_ask_declines(agent_log, tmp_path):
    def broken(req):
        raise RuntimeError("no audio")
    agent = make(agent_log, tmp_path, permissions="ask", ask=broken)
    assert "".join(agent.reply(user("use a tool"))).startswith("Permission no.")
    agent.close()


def test_ask_without_anyone_to_ask_declines(agent_log, tmp_path):
    agent = make(agent_log, tmp_path, permissions="ask")
    assert "".join(agent.reply(user("use a tool"))).startswith("Permission no.")
    agent.close()


def test_unknown_permissions_policy(agent_log, tmp_path):
    with pytest.raises(ValueError):
        make(agent_log, tmp_path, permissions="maybe")


@pytest.mark.parametrize("tool_call, spoken", [
    ({"title": "`npm install`", "kind": "execute"}, "Claude wants to run npm install. Allow?"),
    ({"title": "Bash: ls -la", "kind": "execute"}, "Claude wants to run ls -la. Allow?"),
    ({"title": "Edit", "kind": "edit", "locations": [{"path": "/x/site/index.html"}]},
     "Claude wants to edit index.html. Allow?"),
    ({"title": "x" * 80, "kind": "execute"}, "Claude wants to run a command. Allow?"),
    ({"title": "mkdir site && echo hi > site/index.html", "kind": "execute"},
     "Claude wants to run a command. Allow?"),
    ({"title": "mkdir site && echo hi > site/index.html", "kind": "execute",
      "rawInput": {"command": "mkdir site && echo hi > site/index.html",
                   "description": "Create site folder with an index page"}},
     "Claude wants to create site folder with an index page. Allow?"),
    ({"title": "mcp__thing"}, "Claude wants to use mcp__thing. Allow?"),
    ({}, "Claude wants to use a tool. Allow?"),
])
def test_permission_requests_are_short_to_say(tool_call, spoken):
    from macvoice.acp import PermissionRequest
    assert PermissionRequest("Claude", tool_call).spoken() == spoken


@pytest.mark.parametrize("info, command, name", [
    ({"title": "Claude Code"}, ["x"], "Claude"),
    ({"title": "Gemini CLI"}, ["x"], "Gemini"),
    ({"name": "@agentclientprotocol/claude-agent-acp"}, ["x"], "Claude"),
    ({}, ["/opt/bin/codex-acp"], "Codex"),
    ({"name": "fake"}, ["python", "fake.py"], "The agent"),
])
def test_spoken_name(info, command, name):
    from macvoice.acp import spoken_name
    assert spoken_name(info, command) == name


# ── modes ─────────────────────────────────────────────────────────────

def test_modes_are_read_from_the_session(agent_log, tmp_path):
    agent = make(agent_log, tmp_path)
    assert [m["id"] for m in agent.modes] == ["default", "acceptEdits", "plan"]
    assert agent.mode == "default"
    assert not any(m.get("method") == "session/set_mode" for m in logged(agent_log))
    agent.close()


@pytest.mark.parametrize("wanted", ["acceptEdits", "acceptedits", "Accept Edits"])
def test_mode_by_id_or_name(agent_log, tmp_path, wanted):
    agent = make(agent_log, tmp_path, mode=wanted)
    assert agent.mode == "acceptEdits"
    set_mode = [m for m in logged(agent_log) if m.get("method") == "session/set_mode"]
    assert set_mode[0]["params"] == {"sessionId": "s1", "modeId": "acceptEdits"}
    agent.close()


def test_unknown_mode_lists_the_choices(agent_log, tmp_path):
    with pytest.raises(ACPError, match="no mode 'yolo'. Available: default, acceptEdits, plan"):
        make(agent_log, tmp_path, mode="yolo")


def test_mode_on_an_agent_without_modes(agent_log, tmp_path):
    with pytest.raises(ACPError, match="doesn't advertise any modes"):
        ACPAgent([sys.executable, FAKE, str(agent_log), "nomodes"], cwd=str(tmp_path), timeout=10,
                 mode="plan")


# ── llm.load ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("kw, policy", [({}, "deny"), ({"cwd": "proj"}, "ask"),
                                        ({"cwd": "proj", "allow_tools": True}, "allow"),
                                        ({"permissions": "allow"}, "allow")])
def test_load_picks_the_permission_policy(agent_log, tmp_path, monkeypatch, kw, policy):
    from macvoice import llm
    monkeypatch.setenv("HOME", str(tmp_path))
    if "cwd" in kw:
        kw["cwd"] = str(tmp_path / "new" / "proj")
    agent, about = llm.load("acp", acp_command=f"{sys.executable} {FAKE} {agent_log}", **kw)
    assert agent.permissions == policy and f"tool permissions: {policy}" in about
    if "cwd" in kw:
        assert (tmp_path / "new" / "proj").is_dir() and agent.coding    # created
        assert f"working in {tmp_path / 'new' / 'proj'}" in about
    agent.close()


def test_modes_command(agent_log, tmp_path, capsys):
    from macvoice import cli
    cli.main(["modes", "--acp-command", f"{sys.executable} {FAKE} {agent_log}", "--cwd", str(tmp_path)])
    out = capsys.readouterr().out
    assert "* default" in out and "acceptEdits  (Accept Edits)  edits go through" in out


# ── console support: cancel from another thread, tool activity ───────

def test_cancel_stops_a_turn_waiting_on_a_silent_tool(agent_log, tmp_path):
    import threading
    agent = make(agent_log, tmp_path)
    result = []
    t = threading.Thread(target=lambda: result.append("".join(agent.reply(user("freeze")))))
    t.start()
    deadline = __import__("time").monotonic() + 2
    while agent._prompt_req is None and __import__("time").monotonic() < deadline:
        pass
    agent.cancel()
    t.join(2)
    assert result == [""] and not t.is_alive()
    assert any(m.get("method") == "session/cancel" for m in logged(agent_log))
    agent.cancel()                                         # nothing in flight: a no-op
    assert "Paris" in "".join(agent.reply(user("again")))
    agent.close()


def test_tool_activity_lines(agent_log, tmp_path):
    agent = make(agent_log, tmp_path)
    lines = []
    agent.on_activity = lines.append
    list(agent.reply(user("edit the page")))
    assert lines == ["  ▸ Edit index.html", "  ✓ Edit index.html", "  ▸ npm test", "  ✗ npm test (failed)"]
    agent.close()
