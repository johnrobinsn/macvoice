import json
import shutil
import sys
from pathlib import Path

import pytest

from macvoice.acp import ACPAgent, ACPError
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


def test_brief_goes_with_the_first_turn_only(agent_log, tmp_path):
    agent = make(agent_log, tmp_path)
    list(agent.reply(user("one")))
    list(agent.reply([*user("one"), {"role": "assistant", "content": "x"}, *user("two")]))
    prompts = [m["params"]["prompt"][0]["text"] for m in logged(agent_log) if m.get("method") == "session/prompt"]
    assert prompts == [f"{SYSTEM_PROMPT}\n\none", "two"]   # the agent keeps its own history
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
