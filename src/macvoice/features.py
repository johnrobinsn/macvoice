"""Whisper log-mel features in NumPy.

Smart Turn's encoder is Whisper-tiny, and the reference pipeline computes
its input with transformers' ``WhisperFeatureExtractor``, but importing
transformers pulls in PyTorch (~170 MB of runtime before any model loads).
This is a port of exactly the configuration Smart Turn uses, pinned to the
transformers output by ``tests/test_features.py``.
"""

from __future__ import annotations

import numpy as np


def _slaney_mel_filters(sr: int, n_fft: int, n_mels: int) -> np.ndarray:
    f_sp = 200.0 / 3
    min_log_hz, min_log_mel = 1000.0, 1000.0 / f_sp
    logstep = np.log(6.4) / 27.0

    def hz_to_mel(f):
        f = np.asarray(f, dtype=np.float64)
        return np.where(f >= min_log_hz, min_log_mel + np.log(np.maximum(f, 1e-10) / min_log_hz) / logstep,
                        f / f_sp)

    def mel_to_hz(m):
        m = np.asarray(m, dtype=np.float64)
        return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), f_sp * m)

    fft_freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    mel_pts = mel_to_hz(np.linspace(hz_to_mel(0.0), hz_to_mel(sr / 2), n_mels + 2))
    fdiff = np.diff(mel_pts)
    ramps = mel_pts[:, None] - fft_freqs[None, :]
    lower = -ramps[:-2] / fdiff[:-1, None]
    upper = ramps[2:] / fdiff[1:, None]
    filters = np.maximum(0.0, np.minimum(lower, upper))
    filters *= (2.0 / (mel_pts[2: n_mels + 2] - mel_pts[:n_mels]))[:, None]  # Slaney area norm
    return filters  # (n_mels, n_freqs)


_FILTERS: dict[tuple[int, int, int], np.ndarray] = {}


def whisper_log_mel(audio: np.ndarray, n_samples: int, sr: int = 16000, n_fft: int = 400,
                    hop: int = 160, n_mels: int = 80, do_normalize: bool = True) -> np.ndarray:
    """``WhisperFeatureExtractor(chunk_length=n_samples / sr)`` with
    ``padding="max_length"``, ``truncation=True`` and ``do_normalize``.
    Returns ``(1, n_mels, n_samples // hop)`` float32."""
    x = np.asarray(audio, dtype=np.float32).reshape(-1)
    if do_normalize:
        x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
    x = x[:n_samples] if len(x) >= n_samples else np.pad(x, (0, n_samples - len(x)))
    x = x.astype(np.float64)
    key = (sr, n_fft, n_mels)
    if key not in _FILTERS:
        _FILTERS[key] = _slaney_mel_filters(sr, n_fft, n_mels)
    padded = np.pad(x, (n_fft // 2, n_fft // 2), mode="reflect")
    n_frames = 1 + (len(padded) - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    window = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(n_fft) / n_fft)  # periodic Hann
    power = np.abs(np.fft.rfft(padded[idx] * window, axis=1)) ** 2
    mel = _FILTERS[key] @ power.T
    log_spec = np.log10(np.maximum(mel, 1e-10))[:, :-1]
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec[None].astype(np.float32)
