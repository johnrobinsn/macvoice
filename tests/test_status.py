"""The live status line (no terminal needed)."""

import io
import re

import pytest

from macvoice.status import CEIL_DB, FLOOR_DB, METER_CELLS, StatusLine, meter, render

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def test_meter_spans_floor_to_ceiling():
    assert meter(FLOOR_DB - 20) == "▕" + " " * METER_CELLS + "▏"
    assert meter(CEIL_DB + 5) == "▕" + "█" * METER_CELLS + "▏"
    half = meter((FLOOR_DB + CEIL_DB) / 2)
    assert half.count("█") == METER_CELLS // 2
    assert len(half) == METER_CELLS + 2


def test_meter_draws_partial_cells():
    one_and_a_half = FLOOR_DB + (CEIL_DB - FLOOR_DB) * 1.5 / METER_CELLS
    assert meter(one_and_a_half)[1:3] == "█▌"


@pytest.mark.parametrize("state,expect", [
    ("listening", "○ listening"),
    ("hearing", "● hearing you"),
    ("thinking", "◐ thinking 0.4s"),
    ("speaking", "▶ speaking"),
])
def test_render_names_the_state(state, expect):
    assert render(state, -40, elapsed_s=0.42).startswith(expect)


def test_render_shows_the_meter_only_while_listening_or_hearing():
    assert "▕" in render("listening", -40)
    assert "▕" in render("hearing", -40)
    assert "▕" not in render("thinking", -40)
    assert "▕" not in render("speaking", -40)


def test_render_says_whether_you_can_interrupt():
    assert "talk to interrupt" in render("speaking", -40, barge_in=True)
    assert "mic muted" in render("speaking", -40, barge_in=False)


def test_render_fits_the_terminal():
    assert len(render("hearing", -20, width=20)) == 19


def test_render_colors_only_the_state():
    plain = render("hearing", -30)
    colored = render("hearing", -30, color=True)
    assert colored != plain and ANSI.sub("", colored) == plain
    assert colored.startswith("\x1b[32m● hearing you")


def make_status(source, **kw):
    out = io.StringIO()
    return StatusLine(source, stream=out, enabled=True, fps=10, **kw), out


def test_meter_rises_at_once_and_falls_slowly():
    level = [-20.0]
    status, _ = make_status(lambda: ("listening", level[0]), decay_db_s=40)
    status.line(now=0.0)
    assert status._level == -20
    level[0] = -90
    status.line(now=0.1)
    assert status._level == pytest.approx(-24)  # 40 dB/s for one 0.1 s frame


def test_thinking_time_counts_from_the_state_change():
    state = ["hearing"]
    status, _ = make_status(lambda: (state[0], -50))
    status.line(now=10.0)
    state[0] = "thinking"
    status.line(now=11.0)
    assert "thinking 1.5s" in status.line(now=12.5)


def test_print_goes_above_the_line_and_redraws_it():
    status, out = make_status(lambda: ("listening", -50))
    status.print("  you : hello")
    text = ANSI.sub("", out.getvalue().replace("\r\x1b[K", "\r<clear>"))
    assert text.startswith("\r<clear>  you : hello\n\r<clear>○ listening")


def test_start_draws_and_stop_clears():
    status, out = make_status(lambda: ("listening", -50))
    status.start()
    import time
    deadline = time.time() + 2
    while "listening" not in out.getvalue() and time.time() < deadline:
        time.sleep(0.02)
    status.stop()
    assert "○ listening" in ANSI.sub("", out.getvalue())
    assert out.getvalue().endswith("\r\x1b[K")


def test_off_a_terminal_it_only_prints():
    out = io.StringIO()  # isatty() is False
    status = StatusLine(lambda: ("listening", -50), stream=out).start()
    status.print("  you : hello")
    status.stop()
    assert out.getvalue() == "  you : hello\n"
