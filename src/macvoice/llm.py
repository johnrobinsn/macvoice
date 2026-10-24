"""The agent's brain: any OpenAI-compatible chat endpoint, streamed, or a
coding agent over ACP (:mod:`macvoice.acp`).

Works with local servers (``mlx_lm.server``, Ollama, LM Studio, llama.cpp)
and hosted APIs that speak the same protocol. Standard library only.

Replies are cut into sentences as tokens arrive, so speech starts after
the first sentence instead of the whole reply.
"""

from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from typing import Iterable, Iterator, Protocol

DEFAULT_BASE_URL = "http://localhost:8080/v1"
START_LOCAL_SERVER = ("uvx --from mlx-lm mlx_lm.server "
                      "--model mlx-community/Qwen3-4B-Instruct-2507-4bit --prompt-cache-size 1")

SYSTEM_PROMPT = (
    "You are a helpful voice assistant. Your replies are spoken aloud, so answer in "
    "one to three short sentences of plain text: no markdown, lists, code or emoji."
)


class LLM(Protocol):
    name: str

    def reply(self, history: list[dict]) -> Iterator[str]:
        """Stream reply text for ``history`` (OpenAI chat messages)."""
        ...


class Echo:
    """Repeats the user back. For trying the audio loop without a model
    server, and for measuring the loop without LLM time."""

    name = "echo"

    def reply(self, history: list[dict]) -> Iterator[str]:
        yield f"You said: {history[-1]['content']}"


class OpenAICompatible:
    def __init__(self, base_url: str = DEFAULT_BASE_URL, model: str | None = None,
                 api_key: str | None = None, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model, self.api_key, self.timeout = model, api_key, timeout
        self.name = f"server:{model or base_url}"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def models(self, timeout: float = 2.0) -> list[str]:
        """The server's model ids (``GET /models``); raises ``OSError`` if
        nothing answers."""
        req = urllib.request.Request(f"{self.base_url}/models", headers=self._headers())
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        return [m.get("id", "?") for m in data.get("data") or []]

    def reply(self, history: list[dict]) -> Iterator[str]:
        payload = {"stream": True, "messages": [{"role": "system", "content": SYSTEM_PROMPT}, *history]}
        if self.model:  # local servers use the model they loaded when it's omitted
            payload["model"] = self.model
        req = urllib.request.Request(f"{self.base_url}/chat/completions",
                                     data=json.dumps(payload).encode(), headers=self._headers(),
                                     method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            yield from parse_sse(resp)


def parse_sse(lines: Iterable[bytes]) -> Iterator[str]:
    """Content deltas from an OpenAI streaming response."""
    for raw in lines:
        line = raw.decode("utf-8", "replace").strip()
        if not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            return
        try:
            choices = json.loads(data).get("choices") or []
        except ValueError:
            continue
        for choice in choices:
            delta = (choice.get("delta") or {}).get("content")
            if delta:
                yield delta


_SENTENCE_END = re.compile(r"(?<=[.!?…])[\"')\]]*\s+")
_ABBREVIATIONS = ("mr.", "mrs.", "ms.", "dr.", "st.", "vs.", "e.g.", "i.e.", "etc.", "no.")


def sentences(tokens: Iterable[str], min_chars: int = 12) -> Iterator[str]:
    """Regroup streamed text into sentences for TTS. Fragments shorter than
    ``min_chars`` ("Hi.") are joined with what follows so the voice doesn't
    stop and start on every word; common abbreviations don't split."""
    buf = ""
    for tok in tokens:
        buf += tok
        while True:
            cut = None
            for m in _SENTENCE_END.finditer(buf):
                head = buf[: m.end()].strip()
                if len(head) >= min_chars and not head.lower().endswith(_ABBREVIATIONS):
                    cut = m.end()
                    break
            if cut is None:
                break
            yield buf[:cut].strip()
            buf = buf[cut:]
    if buf.strip():
        yield buf.strip()


def load(name: str | None = None, base_url: str | None = None, model: str | None = None,
         api_key: str | None = None, acp_command: str | None = None,
         allow_tools: bool = False) -> tuple[LLM, str]:
    """The brain, and a line saying what it is.

    ``name`` is "server" (any OpenAI-compatible endpoint, by default a local
    ``mlx_lm.server`` on port 8080), "echo" or "acp". When it isn't given,
    the local server is used if one is running, and echo otherwise.
    """
    if name == "echo":
        return Echo(), "echo (repeats what you say)"
    if name == "acp":
        import shlex

        from macvoice.acp import ACPAgent
        agent = ACPAgent(shlex.split(acp_command) if acp_command else None, allow_tools=allow_tools)
        return agent, f"ACP agent: {' '.join(agent.command)}"
    if name not in (None, "server", "openai"):  # "openai" is an older name for "server"
        raise ValueError(f"unknown LLM {name!r} (server | echo | acp)")

    server = OpenAICompatible(base_url=base_url or DEFAULT_BASE_URL, model=model, api_key=api_key)
    try:
        ids = server.models()
    except (OSError, ValueError) as e:
        if name is not None or base_url is not None:  # asked for explicitly: don't paper over it
            raise RuntimeError(f"no LLM server at {server.base_url} ({e})") from e
        return Echo(), (f"echo (repeats what you say): no LLM server at {server.base_url}.\n"
                        f"       For a local LLM, start one in another terminal:\n"
                        f"       {START_LOCAL_SERVER}")
    shown = model or (ids[0] if len(ids) == 1 else ", ".join(ids) or "the server's model")
    return server, f"{shown} at {server.base_url}"
