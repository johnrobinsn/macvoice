"""Use a coding agent as the brain, over ACP (Agent Client Protocol).

ACP is JSON-RPC over the agent's stdin/stdout, the same shape for every
agent that speaks it: Claude Code (via ``claude-agent-acp``), Gemini CLI,
Codex (via ``codex-acp``), OpenCode and others. So the voice loop can talk
to any of them with one small client:

    initialize -> session/new -> session/prompt (per turn)
                                  <- session/update agent_message_chunk ...
                                  <- result {stopReason}

The agent keeps the conversation itself, so each turn sends only the new
user message.

When the agent asks permission to run a tool, ``permissions`` decides:
"deny" (the default: a voice brain shouldn't edit files or run commands
unattended), "allow" (approve each request once), or "ask" (hand the
request to the ``ask`` callback, which asks the user out loud; see
:meth:`macvoice.agent.VoiceAgent.ask`). Modes ("default", "plan", ...) are
whatever the agent advertises in ``session/new``. Standard library only.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterator

from macvoice.llm import SYSTEM_PROMPT

log = logging.getLogger("macvoice.acp")

PROTOCOL_VERSION = 1


# The brief when the agent works in a project (``cwd`` given): it will run
# tools, so it should talk about what it did rather than read code aloud.
CODING_BRIEF = (
    "You are being driven by voice. Your replies are spoken aloud, so keep them to one to "
    "three short sentences of plain text: no markdown, lists, code or emoji. When you change "
    "files or run commands, say briefly what you did instead of showing the code."
)

PERMISSIONS = ("deny", "allow", "ask")

# The answers an ``ask`` callback returns; anything else declines.
ALLOW, ALWAYS, REJECT = "allow", "always", "reject"


class ACPError(RuntimeError):
    pass


@dataclass
class PermissionRequest:
    """A ``session/request_permission`` from the agent, in the words the
    user hears and sees."""

    agent: str                      # "Claude", "Gemini", ...
    tool_call: dict = field(default_factory=dict)

    @property
    def title(self) -> str:
        return str(self.tool_call.get("title") or "a tool").strip()

    @property
    def details(self) -> str:
        """The exact command or input, for the screen."""
        raw = self.tool_call.get("rawInput")
        if isinstance(raw, dict):
            raw = raw.get("command") or raw.get("file_path") or raw.get("path") or json.dumps(raw)
        return str(raw or "").strip()

    def spoken(self, max_chars: int = 60) -> str:
        """"Claude wants to run npm install. Allow?" The agent's own plain
        description is preferred ("create the site folder"); commands full
        of shell symbols, and long ones, are left to the screen."""
        kind = self.tool_call.get("kind")
        raw = self.tool_call.get("rawInput")
        description = str(raw.get("description") or "").strip().rstrip(".") if isinstance(raw, dict) else ""
        if description and len(description) <= 2 * max_chars:
            return f"{self.agent} wants to {description[0].lower()}{description[1:]}. Allow?"
        what = self.title.replace("`", "")
        for prefix in ("Bash:", "Run:", "Terminal:"):
            if what.startswith(prefix):
                what = what[len(prefix):].strip()
        locations = self.tool_call.get("locations") or []
        if kind in ("edit", "delete", "read", "move") and locations and locations[0].get("path"):
            what = Path(locations[0]["path"]).name
        verb = {"execute": "run", "edit": "edit", "delete": "delete", "read": "read",
                "move": "move", "fetch": "fetch"}.get(kind, "use")
        if len(what) > max_chars or (verb == "run" and any(c in what for c in "&|;><$")):
            what = {"run": "a command", "fetch": "a web page"}.get(verb, "a tool")
        return f"{self.agent} wants to {verb} {what}. Allow?"

    def printed(self) -> str:
        verb = "run" if self.tool_call.get("kind") == "execute" else "use"
        lines = [f"? {self.agent} wants to {verb}: {self.title}"]
        if self.details and self.details != self.title.replace("`", ""):
            lines.append("    " + self.details[:400])
        return "\n".join(lines)


def spoken_name(agent_info: dict, command: list[str]) -> str:
    """What to call the agent out loud: its ``title`` if it gives one,
    else a name recognised in its package or command, else "The agent"."""
    title = str(agent_info.get("title") or "").strip()
    for suffix in (" Code", " CLI", " Agent"):  # "Claude Code wants to..." is a mouthful
        title = title.removesuffix(suffix)
    if title:
        return title
    hay = f"{agent_info.get('name', '')} {' '.join(command)}".lower()
    for key, name in (("claude", "Claude"), ("gemini", "Gemini"), ("codex", "Codex"),
                      ("opencode", "OpenCode"), ("goose", "Goose"), ("qwen", "Qwen")):
        if key in hay:
            return name
    return "The agent"


class ACPAgent:
    def __init__(self, command: list[str] | None = None, cwd: str | None = None,
                 allow_tools: bool = False, timeout: float = 300.0,
                 env: dict[str, str] | None = None, permissions: str | None = None,
                 mode: str | None = None,
                 ask: Callable[[PermissionRequest], str | None] | None = None,
                 coding: bool | None = None) -> None:
        self.command = command or ["claude-agent-acp"]
        self.name = f"acp:{Path(self.command[0]).name}"
        self.permissions = permissions or ("allow" if allow_tools else "deny")
        if self.permissions not in PERMISSIONS:
            raise ValueError(f"permissions must be one of {', '.join(PERMISSIONS)}")
        self.ask, self.timeout = ask, timeout
        # Tool activity as one-line summaries ("▸ Edit index.html"), for the console.
        self.on_activity: Callable[[str], None] | None = None
        self._tool_titles: dict[str, str] = {}
        self._prompt_req: int | None = None
        # Without a project, a neutral working directory, so the agent
        # doesn't pick up whatever you launched from (CLAUDE.md, git, ...).
        self.coding = cwd is not None if coding is None else coding
        self.cwd = str(Path(cwd).expanduser().resolve()) if cwd else str(Path.home() / ".cache/macvoice/acp")
        Path(self.cwd).mkdir(parents=True, exist_ok=True)
        self._proc = subprocess.Popen(self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                      stderr=subprocess.DEVNULL, cwd=self.cwd,
                                      env={**os.environ, **(env or {})}, text=True, bufsize=1)
        self._inbox: queue.Queue[dict | None] = queue.Queue()
        self._next_id = 0
        self._write_lock = threading.Lock()
        threading.Thread(target=self._read, name="acp-reader", daemon=True).start()
        init = self._call("initialize", {
            "protocolVersion": PROTOCOL_VERSION,
            "clientCapabilities": {"fs": {"readTextFile": False, "writeTextFile": False},
                                   "terminal": False},
        })
        self.agent_info = init.get("agentInfo") or {}
        self.spoken_name = spoken_name(self.agent_info, self.command)
        session = self._call("session/new", {"cwd": self.cwd, "mcpServers": []})
        self.session_id = session["sessionId"]
        modes = session.get("modes") or {}
        self.modes: list[dict] = list(modes.get("availableModes") or [])
        self.mode: str | None = modes.get("currentModeId")
        if mode is not None:
            self.set_mode(mode)
        self._first_turn = True

    def set_mode(self, wanted: str) -> str:
        """Switch to one of the agent's advertised modes, by id or name
        (case-insensitive). Raises :class:`ACPError` naming the choices."""
        if not self.modes:
            raise ACPError(f"{self.name} doesn't advertise any modes")
        key = wanted.strip().lower()
        match = next((m for m in self.modes if str(m.get("id", "")).lower() == key
                      or str(m.get("name", "")).lower() == key), None)
        if match is None:
            raise ACPError(f"{self.name} has no mode {wanted!r}. Available: "
                           + ", ".join(str(m.get("id")) for m in self.modes))
        self._call("session/set_mode", {"sessionId": self.session_id, "modeId": match["id"]})
        self.mode = match["id"]
        return self.mode

    # ── transport ─────────────────────────────────────────────────────

    def _read(self) -> None:
        for line in self._proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                self._inbox.put(json.loads(line))
            except ValueError:
                log.debug("non-JSON line from agent: %s", line[:200])
        self._inbox.put(None)  # the agent exited

    def _send(self, msg: dict) -> None:
        with self._write_lock:
            self._proc.stdin.write(json.dumps({"jsonrpc": "2.0", **msg}) + "\n")
            self._proc.stdin.flush()

    def _request(self, method: str, params: dict) -> int:
        self._next_id += 1
        self._send({"id": self._next_id, "method": method, "params": params})
        return self._next_id

    def _messages(self, until_id: int) -> Iterator[dict]:
        """Incoming notifications until the response to ``until_id``, which
        is yielded last. Requests from the agent are answered here."""
        while True:
            try:
                msg = self._inbox.get(timeout=self.timeout)
            except queue.Empty:
                raise ACPError(f"no reply from {self.name} in {self.timeout:.0f} s") from None
            if msg is None:
                raise ACPError(f"{self.name} exited")
            if "method" in msg and "id" in msg:
                self._answer(msg)
            elif "method" in msg:
                yield msg
            elif msg.get("id") == until_id:
                if "error" in msg:
                    raise ACPError(f"{self.name}: {msg['error'].get('message', msg['error'])}")
                yield msg
                return

    def _call(self, method: str, params: dict) -> dict:
        req = self._request(method, params)
        for msg in self._messages(req):
            if msg.get("id") == req:
                return msg.get("result") or {}
        return {}

    def _decide(self, request: PermissionRequest) -> str:
        if self.permissions == "allow":
            return ALLOW
        if self.permissions == "ask" and self.ask is not None:
            try:
                return self.ask(request) or REJECT
            except Exception:  # noqa: BLE001 - a broken prompt must not approve anything
                log.exception("asking about %s failed; declining", request.title)
        return REJECT

    def _answer(self, msg: dict) -> None:
        if msg["method"] == "session/request_permission":
            params = msg["params"]
            request = PermissionRequest(self.spoken_name, params.get("toolCall") or {})
            decision = self._decide(request)
            want = {ALLOW: ("allow_once", "allow_always"),
                    ALWAYS: ("allow_always", "allow_once")}.get(decision, ("reject_once", "reject_always"))
            options = params.get("options", [])
            choice = next((o["optionId"] for k in want for o in options if o.get("kind") == k), None)
            outcome = {"outcome": "selected", "optionId": choice} if choice else {"outcome": "cancelled"}
            log.info("%s %s", "declined" if decision not in (ALLOW, ALWAYS) else "allowed", request.title)
            self._send({"id": msg["id"], "result": {"outcome": outcome}})
        else:
            self._send({"id": msg["id"], "error": {"code": -32601, "message": f"{msg['method']} not supported"}})

    # ── LLM protocol ──────────────────────────────────────────────────

    def reply(self, history: list[dict]) -> Iterator[str]:
        text = history[-1]["content"]
        if self._first_turn:
            # ACP has no system prompt; the first message carries the brief.
            text = f"{CODING_BRIEF if self.coding else SYSTEM_PROMPT}\n\n{text}"
            self._first_turn = False
        req = self._request("session/prompt", {"sessionId": self.session_id,
                                               "prompt": [{"type": "text", "text": text}]})
        self._prompt_req = req
        done = False
        try:
            for msg in self._messages(req):
                if msg.get("id") == req:
                    done = True
                    return
                update = (msg.get("params") or {}).get("update") or {}
                kind = update.get("sessionUpdate")
                if kind == "agent_message_chunk":
                    content = update.get("content") or {}
                    if content.get("type") == "text" and content.get("text"):
                        yield content["text"]
                elif kind in ("tool_call", "tool_call_update"):
                    self._report_tool(update)
        finally:
            self._prompt_req = None
            if not done and self._proc.poll() is None:  # barge-in: stop the agent's turn
                try:
                    self._send({"method": "session/cancel", "params": {"sessionId": self.session_id}})
                except OSError:
                    return
                for _ in self._drain(req):
                    pass

    def cancel(self) -> None:
        """Ask the agent to stop the current turn, from any thread. The
        turn then ends with ``stopReason: cancelled``, which also unblocks a
        reply that's waiting on a silent tool."""
        if self._prompt_req is not None and self._proc.poll() is None:
            try:
                self._send({"method": "session/cancel", "params": {"sessionId": self.session_id}})
            except OSError:
                pass

    def _report_tool(self, update: dict) -> None:
        """One line when a tool starts and one when it finishes."""
        tool_id = str(update.get("toolCallId", ""))
        if update.get("title"):
            self._tool_titles[tool_id] = str(update["title"]).replace("`", "")
        title = self._tool_titles.get(tool_id, "tool")
        status = update.get("status")
        line = None
        if update.get("sessionUpdate") == "tool_call":
            line = f"  ▸ {title}"
        elif status == "completed":
            line = f"  ✓ {title}"
        elif status == "failed":
            line = f"  ✗ {title} (failed)"
        if line and self.on_activity is not None:
            self.on_activity(line)

    def _drain(self, req: int) -> Iterator[dict]:
        try:
            yield from self._messages(req)
        except ACPError:
            return

    def close(self) -> None:
        try:
            self._proc.stdin.close()
            self._proc.wait(timeout=5)
        except Exception:
            self._proc.kill()
