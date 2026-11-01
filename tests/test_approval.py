"""Spoken answers and single-key answers to permission questions."""

import io
import os
import pty
import termios
import time

import pytest

from macvoice.approval import KeyListener, parse_answer


def attrs(fd):
    """Terminal settings, minus PENDIN (the kernel's "input pending" mark)."""
    a = termios.tcgetattr(fd)
    a[3] &= ~termios.PENDIN
    return a


@pytest.mark.parametrize("text, answer", [
    ("Yes.", "allow"), ("yeah go ahead", "allow"), ("Sure, do it.", "allow"), ("Okay", "allow"),
    ("Always.", "always"), ("yes, always allow that", "always"),
    ("No.", "reject"), ("Don't.", "reject"), ("Don’t do that", "reject"), ("Stop", "reject"),
    ("no, yes", "reject"),            # anything negative wins
    ("yes, wait", "reject"),
    ("I know", None),                 # "know" is not "no"
    ("Banana.", None), ("", None), (".", None),
])
def test_parse_answer(text, answer):
    assert parse_answer(text) == answer


def test_key_listener_reads_single_keys_and_restores_the_terminal():
    master, slave = pty.openpty()
    stream = os.fdopen(slave, "r")
    before = attrs(slave)
    keys = []
    listener = KeyListener(stream)
    assert listener.enabled
    os.write(master, b"y")                       # typed before the question
    time.sleep(0.05)
    listener.start(keys.append)
    assert attrs(slave) != before    # single-key mode
    os.write(master, b"N")
    deadline = time.monotonic() + 2
    while not keys and time.monotonic() < deadline:
        time.sleep(0.01)
    listener.stop()
    assert keys == ["n"]                         # the early "y" was discarded
    assert attrs(slave) == before
    stream.close()
    os.close(master)


def test_key_listener_does_nothing_without_a_terminal():
    listener = KeyListener(io.StringIO())
    assert not listener.enabled
    listener.start(lambda key: None)
    listener.stop()
