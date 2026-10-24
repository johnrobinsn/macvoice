"""The NumPy Whisper log-mel against transformers' WhisperFeatureExtractor.

The reference (tests/fixtures/reference_*.npz) was computed with
transformers 4.57 on a LibriSpeech clip padded to Smart Turn's 8 s window,
so this test needs neither transformers nor PyTorch."""

import numpy as np

from macvoice.audio import read_wav
from macvoice.features import whisper_log_mel
from tests.conftest import FIXTURES, tone

N = 8 * 16000


def test_matches_transformers_reference():
    ref = np.load(FIXTURES / "reference_1272-128104-0001.npz")["whisper_log_mel"]
    audio, _ = read_wav(FIXTURES / "librispeech/1272-128104-0001.wav")
    ours = whisper_log_mel(np.pad(audio, (N - len(audio), 0)), N)
    assert ours.shape == ref.shape == (1, 80, 800)
    assert np.max(np.abs(ours - ref)) < 1e-3


def test_pads_and_truncates_to_window():
    short = whisper_log_mel(tone(300, 1.0), N)
    long = whisper_log_mel(tone(300, 10.0), N)
    assert short.shape == long.shape == (1, 80, 800)
    assert short.dtype == np.float32


def test_dynamic_range_is_clamped_to_8_decades():
    feats = whisper_log_mel(tone(300, 2.0), N)
    # (log10 + 4) / 4 scaling: an 8-decade window is 2.0 wide
    assert feats.max() - feats.min() <= 2.0 + 1e-6


def test_silence_is_finite():
    assert np.isfinite(whisper_log_mel(np.zeros(N, dtype=np.float32), N)).all()
