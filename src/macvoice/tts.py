"""Text to speech, streamed.

Default: Phonon 31M English (gradium-ai's ``ptts``). Apache-2.0 weights and
voices, an MIT/Apache-2.0 Rust runtime on Metal, no PyTorch and no GPL
phonemizer. It streams decoder frames as they are produced, so the first
audio arrives in milliseconds rather than after a whole sentence.

Baseline (``--tts kokoro``, needs the ``baseline`` extra): Kokoro-82M on
PyTorch-MPS, which synthesizes a sentence at a time.
"""

from __future__ import annotations

import platform
import sys
from typing import Iterator, Protocol

import numpy as np

TTS_RATE = 24000


class TTS(Protocol):
    name: str
    sample_rate: int

    def stream(self, text: str) -> Iterator[np.ndarray]:
        """Float32 mono chunks at ``sample_rate`` as soon as each is ready."""
        ...


class Phonon:
    name = "phonon"

    def __init__(self, voice: str | None = None, device: str | None = None) -> None:
        import ptts

        apple = sys.platform == "darwin" and platform.machine() == "arm64"
        device = device or ("metal" if apple else "cpu")
        # Quantized weights are CPU-only in ptts; Metal runs f32.
        quant = "f32" if device == "metal" else "q8_0"
        if device == "cpu":
            ptts.set_num_threads(4)
        self._tts = ptts.TTS(lang="en", device=device, quant=quant)
        self.sample_rate = int(self._tts.sample_rate)
        self.voice = voice

    def stream(self, text: str) -> Iterator[np.ndarray]:
        for chunk in self._tts.stream(text, voice=self.voice):
            yield np.asarray(chunk, dtype=np.float32).reshape(-1)


class Kokoro:
    """Baseline: Kokoro-82M on PyTorch (MPS when available)."""

    name = "kokoro"
    sample_rate = TTS_RATE

    def __init__(self, voice: str = "af_heart") -> None:
        import torch
        from kokoro import KPipeline

        device = "mps" if torch.backends.mps.is_available() else "cpu"
        self._pipeline = KPipeline(lang_code="a", device=device, repo_id="hexgrad/Kokoro-82M")
        self.voice = voice

    def stream(self, text: str) -> Iterator[np.ndarray]:
        for _, _, audio in self._pipeline(text, voice=self.voice):
            if audio is not None:
                yield np.asarray(audio, dtype=np.float32).reshape(-1)


def load(name: str = "phonon", voice: str | None = None) -> TTS:
    if name == "phonon":
        return Phonon(voice=voice)
    if name == "kokoro":
        return Kokoro(voice=voice or "af_heart")
    raise ValueError(f"unknown TTS {name!r} (phonon | kokoro)")
