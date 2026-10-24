"""WAV I/O and resampling, NumPy only."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np


def read_wav(path: str | Path, sample_rate: int | None = None) -> tuple[np.ndarray, int]:
    """16-bit PCM WAV -> (float32 mono in [-1, 1], rate). Resampled to
    ``sample_rate`` when given."""
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: only 16-bit PCM WAV is supported")
        rate, channels = w.getframerate(), w.getnchannels()
        pcm = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    audio = pcm.astype(np.float32).reshape(-1, channels).mean(axis=1) / 32768.0
    if sample_rate is not None and sample_rate != rate:
        audio, rate = resample(audio, rate, sample_rate), sample_rate
    return audio.astype(np.float32), rate


def write_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    pcm = (np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0) * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm.tobytes())


def resample(audio: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    """Band-limited resampling in the frequency domain, for whole clips
    (files, synthesized replies). Live audio uses :class:`StreamResampler`."""
    x = np.asarray(audio, dtype=np.float64).reshape(-1)
    if src_rate == dst_rate or len(x) == 0:
        return x.astype(np.float32)
    n_out = int(round(len(x) * dst_rate / src_rate))
    spec = np.fft.rfft(x)
    n_bins = n_out // 2 + 1
    out_spec = np.zeros(n_bins, dtype=np.complex128)
    k = min(n_bins, len(spec))
    out_spec[:k] = spec[:k]
    if n_out % 2 == 0 and k == n_bins and n_out < len(x):
        out_spec[-1] = out_spec[-1].real  # Nyquist bin of a real signal is real
    y = np.fft.irfft(out_spec, n=n_out) * (n_out / len(x))
    return y.astype(np.float32)


class StreamResampler:
    """Resample a live stream block by block with a windowed-sinc filter,
    carrying state across blocks so the output is continuous. Used for the
    voice-processing microphone, which only delivers the hardware rate."""

    def __init__(self, src_rate: int, dst_rate: int, taps: int = 32) -> None:
        self.step = src_rate / dst_rate
        self.cutoff = min(1.0, dst_rate / src_rate) * 0.9
        self.half = taps // 2
        self._buf = np.zeros(self.half, dtype=np.float64)  # history; starts as silence
        self._pos = float(self.half)                        # next output position in _buf

    def __call__(self, block: np.ndarray) -> np.ndarray:
        self._buf = np.concatenate([self._buf, np.asarray(block, dtype=np.float64).reshape(-1)])
        last = len(self._buf) - self.half  # outputs need ``half`` samples of lookahead
        if self._pos >= last:
            return np.zeros(0, dtype=np.float32)
        pos = np.arange(self._pos, last, self.step)
        base = np.floor(pos).astype(np.int64)
        offsets = np.arange(-self.half + 1, self.half + 1)
        idx = base[:, None] + offsets[None, :]
        d = pos[:, None] - idx
        w = 0.5 + 0.5 * np.cos(np.pi * d / self.half)  # Hann over +-half samples
        h = self.cutoff * np.sinc(self.cutoff * d) * w
        out = (self._buf[idx] * h).sum(axis=1)
        nxt = pos[-1] + self.step
        drop = int(np.floor(nxt)) - self.half  # keep ``half`` samples of history
        self._buf = self._buf[drop:]
        self._pos = nxt - drop
        return out.astype(np.float32)


class Chunker:
    """Regroup a stream of arbitrary-length blocks into fixed-size chunks."""

    def __init__(self, size: int) -> None:
        self.size = size
        self._buf = np.zeros(0, dtype=np.float32)

    def __call__(self, block: np.ndarray) -> list[np.ndarray]:
        self._buf = np.concatenate([self._buf, np.asarray(block, dtype=np.float32).reshape(-1)])
        n = len(self._buf) // self.size
        out = list(self._buf[: n * self.size].reshape(n, self.size)) if n else []
        self._buf = self._buf[n * self.size:]
        return out


def frames(audio: np.ndarray, size: int) -> list[np.ndarray]:
    """Split into consecutive ``size``-sample chunks; the last partial chunk
    is zero-padded."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    n = -(-len(audio) // size)
    padded = np.pad(audio, (0, n * size - len(audio)))
    return list(padded.reshape(n, size)) if n else []
