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
import urllib.request
from typing import Iterable, Iterator, Protocol

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
    def __init__(self, base_url: str = "http://localhost:8080/v1", model: str | None = None,
                 api_key: str | None = None, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.model, self.api_key, self.timeout = model, api_key, timeout
        self.name = f"openai:{model or base_url}"

    def reply(self, history: list[dict]) -> Iterator[str]:
        payload = {"stream": True, "messages": [{"role": "system", "content": SYSTEM_PROMPT}, *history]}
        if self.model:  # local servers use the model they loaded when it's omitted
            payload["model"] = self.model
        body = json.dumps(payload)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(f"{self.base_url}/chat/completions", data=body.encode(),
                                     headers=headers, method="POST")
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


def load(name: str = "echo", base_url: str | None = None, model: str | None = None,
         api_key: str | None = None, acp_command: str | None = None,
         allow_tools: bool = False) -> LLM:
    if name == "echo":
        return Echo()
    if name == "acp":
        import shlex

        from macvoice.acp import ACPAgent
        return ACPAgent(shlex.split(acp_command) if acp_command else None, allow_tools=allow_tools)
    if name == "openai":
        return OpenAICompatible(base_url=base_url or "http://localhost:8080/v1",
                                model=model, api_key=api_key)
    raise ValueError(f"unknown LLM {name!r} (echo | openai | acp)")
