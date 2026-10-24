"""Endpointer on synthetic VAD probabilities: every rule, deterministically.

Each chunk is 32 ms. Chunks carry their index as audio so tests can see
exactly which chunks a turn contains."""

import numpy as np
import pytest

from macvoice.endpointing import CHUNK_MS, EndpointConfig, Endpointer
from macvoice.vad import CHUNK


def chunk(i: int) -> np.ndarray:
    return np.full(CHUNK, i, dtype=np.float32)


class Detector:
    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls = []

    def __call__(self, audio):
        self.calls.append(audio)
        return self.answers.pop(0) if len(self.answers) > 1 else self.answers[0]


def run(ep: Endpointer, probs):
    events = []
    for i, p in enumerate(probs):
        e = ep.push(chunk(i), p)
        if e is not None:
            events.append((i, e))
    return events


S, Q = 0.9, 0.0  # speech, quiet


def n(ms):
    return int(round(ms / CHUNK_MS))


def test_silence_policy_ends_after_max_silence():
    ep = Endpointer(None, EndpointConfig(max_silence_ms=1000))
    events = run(ep, [Q] * 5 + [S] * 20 + [Q] * 40)
    kinds = [(i, e.kind) for i, e in events]
    assert kinds[0] == (7, "start")                     # 3rd speech chunk (96 ms)
    end_i, end = events[1]
    assert end.kind == "end" and end.reason == "silence"
    assert end_i == 24 + n(1000)                        # last speech chunk + 1000 ms
    assert end.speech_end_s == pytest.approx(25 * CHUNK_MS / 1000)
    assert end.time_s - end.speech_end_s == pytest.approx(n(1000) * CHUNK_MS / 1000)


def test_turn_audio_has_pre_roll_and_tail():
    cfg = EndpointConfig(max_silence_ms=1000, pre_roll_ms=320, tail_ms=200)
    ep = Endpointer(None, cfg)
    (_, _), (_, end) = run(ep, [Q] * 20 + [S] * 20 + [Q] * 40)
    ids = end.audio.reshape(-1, CHUNK)[:, 0].astype(int).tolist()
    assert ids[0] == 20 + 3 - n(320)                    # pre-roll ends at the opening chunk
    assert ids[-1] == 39 + n(200)                       # last speech chunk + tail
    assert ids == list(range(ids[0], ids[-1] + 1))      # contiguous


def test_short_blip_is_discarded():
    ep = Endpointer(None, EndpointConfig(max_silence_ms=500, min_voiced_ms=250))
    events = run(ep, [S] * 4 + [Q] * 30)
    assert [e.kind for _, e in events] == ["start", "discard"]


def test_two_chunks_of_speech_never_open_a_turn():
    ep = Endpointer(None, EndpointConfig())
    assert run(ep, [S, S, Q, S, S, Q] * 5) == []


def test_hysteresis_keeps_state_between_thresholds():
    ep = Endpointer(None, EndpointConfig(max_silence_ms=500))
    # 0.4 is between silence_prob (0.35) and speech_prob (0.5): it continues
    # whatever came before, so speech -> 0.4 stays speech.
    events = run(ep, [S] * 10 + [0.4] * 30 + [Q] * 30)
    _, end = events[1]
    assert end.speech_end_s == pytest.approx(40 * CHUNK_MS / 1000)


def test_detector_complete_ends_turn_at_check():
    det = Detector(0.9)
    ep = Endpointer(det, EndpointConfig(check_silence_ms=300, max_silence_ms=3000))
    events = run(ep, [S] * 20 + [Q] * 100)
    end_i, end = events[1]
    assert end.reason == "detector" and end.p_complete == 0.9
    assert end_i == 19 + n(300)
    assert len(det.calls) == 1


def test_detector_sees_audio_trimmed_to_speech_end():
    det = Detector(0.9)
    ep = Endpointer(det, EndpointConfig(check_silence_ms=300))
    run(ep, [Q] * 10 + [S] * 20 + [Q] * 30)
    audio = det.calls[0]
    # every chunk value up to the last speech chunk (29), none of the silence after
    ids = set(np.unique(audio).astype(int).tolist())
    assert max(ids) == 29


def test_detector_incomplete_waits_for_max_silence():
    det = Detector(0.1)
    ep = Endpointer(det, EndpointConfig(check_silence_ms=300, max_silence_ms=2000))
    events = run(ep, [S] * 20 + [Q] * 100)
    end_i, end = events[1]
    assert end.reason == "silence" and end.p_complete == 0.1
    assert end_i == 19 + n(2000)
    assert len(det.calls) == 1                          # asked once per pause, not every chunk


def test_resumed_speech_continues_the_turn_and_asks_again():
    det = Detector(0.1, 0.95)                           # "um..." then a real ending
    ep = Endpointer(det, EndpointConfig(check_silence_ms=300, max_silence_ms=3000))
    events = run(ep, [S] * 20 + [Q] * 40 + [S] * 15 + [Q] * 30)
    assert [e.kind for _, e in events] == ["start", "end"]
    _, end = events[1]
    assert end.reason == "detector" and end.detector_calls == 2
    ids = end.audio.reshape(-1, CHUNK)[:, 0].astype(int)
    assert ids.min() <= 2 and ids.max() >= 74           # one turn spanning both parts


def test_threshold_is_inclusive():
    ep = Endpointer(Detector(0.5), EndpointConfig(threshold=0.5))
    assert run(ep, [S] * 20 + [Q] * 20)[1][1].reason == "detector"


def test_max_turn_length():
    ep = Endpointer(None, EndpointConfig(max_turn_s=2.0))
    events = run(ep, [S] * 200)
    assert events[1][1].reason == "max_turn"
    assert events[1][0] < 70


def test_consecutive_turns_and_reset():
    ep = Endpointer(None, EndpointConfig(max_silence_ms=500))
    events = run(ep, ([S] * 15 + [Q] * 30) * 3)
    assert [e.kind for _, e in events] == ["start", "end"] * 3
    ep.push(chunk(0), S)
    ep.push(chunk(0), S)
    ep.push(chunk(0), S)
    assert ep.in_turn
    ep.reset()
    assert not ep.in_turn
