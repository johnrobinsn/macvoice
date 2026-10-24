import numpy as np
import pytest

from macvoice.audio import frames, read_wav
from macvoice.vad import CHUNK, SileroVAD
from tests.conftest import FIXTURES


class FakeSession:
    """Records inputs; returns the call count as the probability and
    increments the state, so state threading is visible."""

    def __init__(self):
        self.calls = []

    def run(self, _, feeds):
        self.calls.append({k: np.array(v, copy=True) for k, v in feeds.items()})
        return np.array([[len(self.calls) / 10]], dtype=np.float32), feeds["state"] + 1


def test_context_and_state_carry_over():
    session = FakeSession()
    vad = SileroVAD(session=session)
    a = np.full(CHUNK, 0.1, dtype=np.float32)
    b = np.full(CHUNK, 0.2, dtype=np.float32)
    assert vad(a) == pytest.approx(0.1)
    assert vad(b) == pytest.approx(0.2)
    first, second = session.calls
    assert first["input"].shape == (1, 64 + CHUNK)
    assert np.all(first["input"][0, :64] == 0)               # no context yet
    assert np.allclose(second["input"][0, :64], 0.1)         # last 64 samples of a
    assert np.all(second["state"] == 1)                      # state from call 1
    assert second["sr"] == 16000


def test_reset_clears_state():
    session = FakeSession()
    vad = SileroVAD(session=session)
    vad(np.ones(CHUNK, dtype=np.float32))
    vad.reset()
    vad(np.ones(CHUNK, dtype=np.float32))
    assert np.all(session.calls[1]["state"] == 0)
    assert np.all(session.calls[1]["input"][0, :64] == 0)


def test_rejects_wrong_chunk_size():
    with pytest.raises(ValueError, match="512"):
        SileroVAD(session=FakeSession())(np.zeros(480, dtype=np.float32))


@pytest.mark.models
def test_matches_pytorch_reference(vad):
    """Same probabilities as silero-vad's PyTorch JIT model (v6.2.3)."""
    ref = np.load(FIXTURES / "reference_1272-128104-0001.npz")["silero_probs"]
    audio, _ = read_wav(FIXTURES / "librispeech/1272-128104-0001.wav")
    vad.reset()
    ours = np.array([vad(audio[i:i + CHUNK]) for i in range(0, len(audio) - CHUNK + 1, CHUNK)])
    assert len(ours) == len(ref)
    assert np.max(np.abs(ours - ref)) < 1e-4


@pytest.mark.models
def test_speech_vs_silence_and_noise(vad, librispeech):
    vad.reset()
    speech = [vad(c) for c in frames(librispeech[2]["audio"], CHUNK)]
    vad.reset()
    silence = [vad(c) for c in frames(np.zeros(16000, dtype=np.float32), CHUNK)]
    vad.reset()
    rng = np.random.default_rng(0)
    noise = [vad(c) for c in frames(rng.normal(0, 0.01, 16000).astype(np.float32), CHUNK)]
    assert np.mean(np.array(speech) > 0.5) > 0.6
    assert max(silence) < 0.1
    assert max(noise) < 0.5
