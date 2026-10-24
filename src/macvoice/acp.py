"""Use a coding agent as the brain, over ACP (Agent Client Protocol).

ACP is JSON-RPC over the agent's stdin/stdout, the same shape for every
agent that speaks it: Claude Code (via ``claude-agent-acp``), Gemini CLI,
Codex (via ``codex-acp``), OpenCode and others. So the voice loop can talk
to any of them with one small client:

    initialize -> session/new -> session/prompt (per turn)
                                  <- session/update agent_message_chunk ...
                                  <- result {stopReason}

The agent keeps the conversation itself, so each turn sends only the new
user message. It may ask permission to run tools; by default this client
declines (a voice brain shouldn't edit files or run commands unattended),
``allow_tools=True`` approves once per request. Standard library only.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import threading
from pathlib import Path
from typing import Iterator

from macvoice.llm import SYSTEM_PROMPT

log = logging.getLogger("macvoice.acp")

PROTOCOL_VERSION = 1


class ACPError(RuntimeError):
    pass


class ACPAgent:
    def __init__(self, command: list[str] | None = None, cwd: str | None = None,
                 allow_tools: bool = False, timeout: float = 300.0,
                 env: dict[str, str] | None = None) -> None:
        self.command = command or ["claude-agent-acp"]
        self.name = f"acp:{Path(self.command[0]).name}"
        self.allow_tools, self.timeout = allow_tools, timeout
        # A neutral working directory, so the agent doesn't pick up the
        # project you happen to launch from (CLAUDE.md, git state, ...).
        self.cwd = cwd or str(Path.home() / ".cache/macvoice/acp")
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
        self.session_id = self._call("session/new", {"cwd": self.cwd, "mcpServers": []})["sessionId"]
        self._first_turn = True

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

    def _answer(self, msg: dict) -> None:
        if msg["method"] == "session/request_permission":
            options = msg["params"].get("options", [])
            want = ("allow_once", "allow_always") if self.allow_tools else ("reject_once", "reject_always")
            choice = next((o["optionId"] for k in want for o in options if o.get("kind") == k), None)
            outcome = {"outcome": "selected", "optionId": choice} if choice else {"outcome": "cancelled"}
            title = (msg["params"].get("toolCall") or {}).get("title", "a tool")
            log.info("%s %s", "allowed" if self.allow_tools else "declined", title)
            self._send({"id": msg["id"], "result": {"outcome": outcome}})
        else:
            self._send({"id": msg["id"], "error": {"code": -32601, "message": f"{msg['method']} not supported"}})

    # ── LLM protocol ──────────────────────────────────────────────────

    def reply(self, history: list[dict]) -> Iterator[str]:
        text = history[-1]["content"]
        if self._first_turn:
            # ACP has no system prompt; the first message carries the brief.
            text = f"{SYSTEM_PROMPT}\n\n{text}"
            self._first_turn = False
        req = self._request("session/prompt", {"sessionId": self.session_id,
                                               "prompt": [{"type": "text", "text": text}]})
        done = False
        try:
            for msg in self._messages(req):
                if msg.get("id") == req:
                    done = True
                    return
                update = (msg.get("params") or {}).get("update") or {}
                if update.get("sessionUpdate") == "agent_message_chunk":
                    content = update.get("content") or {}
                    if content.get("type") == "text" and content.get("text"):
                        yield content["text"]
        finally:
            if not done and self._proc.poll() is None:  # barge-in: stop the agent's turn
                try:
                    self._send({"method": "session/cancel", "params": {"sessionId": self.session_id}})
                except OSError:
                    return
                for _ in self._drain(req):
                    pass

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
