import numpy as np
import pytest

from macvoice.turn import SmartTurn, trim_trailing_silence
from tests.conftest import tone


class FakeSession:
    def __init__(self, p=0.7):
        self.p, self.feeds = p, []

    def run(self, _, feeds):
        self.feeds.append(feeds)
        return [np.array([[self.p]], dtype=np.float32)]


def test_trim_cuts_trailing_silence():
    audio = np.concatenate([tone(300, 1.0), np.zeros(16000, dtype=np.float32)])
    out = trim_trailing_silence(audio)
    assert 16000 <= len(out) <= 16000 + 800 + 320  # 50 ms tail, one frame of slack


def test_trim_is_relative_to_peak():
    quiet = np.concatenate([tone(300, 1.0, amp=0.01), np.zeros(8000, dtype=np.float32)])
    assert len(trim_trailing_silence(quiet)) < 17500


def test_trim_keeps_soft_ending_within_range():
    # a decaying tail 20 dB down is still speech, not silence
    audio = np.concatenate([tone(300, 1.0), tone(300, 0.5, amp=0.05), np.zeros(8000, dtype=np.float32)])
    out = trim_trailing_silence(audio)
    assert len(out) >= 24000


def test_trim_edge_cases():
    assert len(trim_trailing_silence(np.zeros(100, dtype=np.float32))) == 100
    assert len(trim_trailing_silence(np.zeros(16000, dtype=np.float32))) > 0


def test_smart_turn_window_front_pads_and_keeps_last_8s():
    session = FakeSession()
    st = SmartTurn(session=session)
    assert st(tone(300, 1.0)) == pytest.approx(0.7)
    st(tone(300, 12.0))
    for feeds in session.feeds:
        assert feeds["input_features"].shape == (1, 80, 800)
    # 1 s of audio, front-padded: the first 7 s of frames are padding (identical columns)
    feats = session.feeds[0]["input_features"][0]
    assert np.allclose(feats[:, 10], feats[:, 600])
    assert not np.allclose(feats[:, 10], feats[:, 780])


def test_smart_turn_empty_audio():
    assert SmartTurn(session=FakeSession())(np.zeros(0)) == 0.0


@pytest.mark.models
def test_complete_requests_score_high(smart_turn, prompts):
    scores = [smart_turn(a) for (_, _, kind), a in prompts.items() if kind == "complete"]
    assert np.mean(np.array(scores) > 0.5) == 1.0


@pytest.mark.models
def test_filled_pauses_score_low(smart_turn, prompts):
    """A turn that stops on "um," / "uh," / "hmm," is not finished."""
    from macvoice.bench import FILLERS, TURN_ITEMS
    scores = [smart_turn(trim_trailing_silence(a)) for (_, item, kind), a in prompts.items()
              if kind == "head" and FILLERS.search(TURN_ITEMS[item][1][0])]
    assert len(scores) == 16
    assert max(scores) < 0.5


@pytest.mark.models
def test_trailing_silence_pushes_toward_complete(smart_turn, prompts):
    """Why the endpointer trims before asking: silence reads as 'done'."""
    from macvoice.bench import FILLERS, TURN_ITEMS
    heads = [trim_trailing_silence(a) for (_, item, kind), a in prompts.items()
             if kind == "head" and FILLERS.search(TURN_ITEMS[item][1][0])]
    trimmed = np.mean([smart_turn(h) for h in heads])
    padded = np.mean([smart_turn(np.concatenate([h, np.zeros(16000, dtype=np.float32)])) for h in heads])
    assert padded > trimmed
