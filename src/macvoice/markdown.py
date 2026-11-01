"""Markdown and the voice.

Coding agents answer in markdown. On screen it's rendered (the console uses
rich); out loud, code blocks are left out and the markers stripped, so the
voice doesn't read code or say "asterisk asterisk".
"""

from __future__ import annotations

import re

_LINK = re.compile(r"\[([^\]\n]+)\]\(([^)\s]+)\)")


def without_code_blocks(tokens):
    """Stream ``tokens`` with fenced code blocks removed, as they arrive.
    Only the first characters of each line are held back, to tell a fence
    (```) from ordinary text, so speech isn't delayed."""
    in_code = False
    head = ""          # the start of the current line, while it might be a fence
    at_line_start = True
    fence_line = False
    for tok in tokens:
        out = []
        for ch in tok:
            if at_line_start:
                if ch == "\n":
                    if not in_code:
                        out.append(head + ch)
                    head = ""
                    continue
                head += ch
                stripped = head.lstrip()
                if stripped.startswith("```"):
                    in_code = not in_code
                    at_line_start = False
                    fence_line = True
                    head = ""
                    continue
                if stripped and not "```".startswith(stripped):
                    at_line_start = False
                    fence_line = False
                    if not in_code:
                        out.append(head)
                    head = ""
                continue
            if ch == "\n":
                at_line_start = True
                if not in_code and not fence_line:
                    out.append(ch)
                continue
            if not in_code and not fence_line:
                out.append(ch)
        if out:
            yield "".join(out)
    if head and not in_code:
        yield head


def plain(sentence: str) -> str:
    """A sentence as it should be spoken: no markdown markers, link text
    without the URL, no heading or bullet marks."""
    s = _LINK.sub(lambda m: m.group(1), sentence)
    s = re.sub(r"^\s*(#{1,6}|[-*+]|>)\s+", "", s, flags=re.M)
    # Markers at word edges only: snake_case keeps its underscore.
    return re.sub(r"(?<!\w)[*`_]{1,3}(?=\S)|(?<=\S)[*`_]{1,3}(?!\w)", "", s).strip()
