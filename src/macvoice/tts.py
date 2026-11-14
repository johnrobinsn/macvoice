"""Text to speech, streamed.

Default: Phonon 31M English (gradium-ai's ``ptts``). Apache-2.0 weights and
voices, an MIT/Apache-2.0 Rust runtime on Metal, no PyTorch and no GPL
phonemizer. It streams decoder frames as they are produced, so the first
audio arrives in milliseconds rather than after a whole sentence.

Voice cloning (``--tts chatterbox``, needs the ``chatterbox`` extra):
Chatterbox Turbo (``mlx-community/chatterbox-turbo-fp16``) on MLX, which
clones the voice from a reference clip — a ``.wav`` in a voices directory
(``--voice`` picks the name, ``--voices`` the directory) — and streams the
decoded frames as they are produced.

Baseline (``--tts kokoro``, needs the ``baseline`` extra): Kokoro-82M on
PyTorch-MPS, which synthesizes a sentence at a time.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path
from typing import Iterator, Protocol

import contextlib
import io
import os

import numpy as np

TTS_RATE = 24000
# What ptts 0.4 loads by default, pinned here so macvoice.models can fetch
# the same files up front.
PHONON_REPO = "gradium/phonon-31m-en"
PHONON_REVISION = "8c64b55ec301728e2d1695bfea87a08a267d0c2d"
# Chatterbox Turbo on MLX (fp16). Clones the voice from a reference clip.
CHATTERBOX_REPO = "mlx-community/chatterbox-turbo-fp16"


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
        self._tts = ptts.TTS(f"hf://{PHONON_REPO}@{PHONON_REVISION}", lang="en", device=device, quant=quant)
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


class Chatterbox:
    """Chatterbox Turbo on MLX, cloning the voice from a reference clip.

    ``--voice`` names a ``.wav`` (stem) in the voices directory; the clip is
    the reference for zero-shot cloning. The turbo build ignores ``exaggeration``.
    """

    name = "chatterbox"
    sample_rate = TTS_RATE

    def __init__(self, voice: str | None = None, voices: str | Path | None = None,
                 stream_interval: float = 0.3) -> None:
        from mlx_audio.tts.utils import load_model

        self._model = load_model(CHATTERBOX_REPO)
        self.sample_rate = int(self._model.sample_rate)
        self.stream_interval = stream_interval
        self.voices = Path(voices) if voices else None
        self.voice = voice

    def _reference(self) -> str | None:
        """Resolve ``--voice`` to a reference clip path."""
        if not self.voice:
            return None
        if self.voices is None:
            raise FileNotFoundError(
                f"Chatterbox needs a voices directory for voice {self.voice!r} (--voices)")
        if self.voice.endswith(".wav"):
            path = self.voices / self.voice
        else:
            path = self.voices / f"{self.voice}.wav"
        if not path.is_file():
            known = sorted(p.stem for p in self.voices.glob("*.wav"))
            raise FileNotFoundError(f"voice {self.voice!r} not in {self.voices} (have: {', '.join(known)})")
        return str(path.resolve())

    def stream(self, text: str) -> Iterator[np.ndarray]:
        ref = self._reference()
        # mlx_audio hardcodes print() and tqdm() progress bars into the model
        # code; send both streams to /dev/null for the duration of the loop so
        # they don't wreck a live status line (macvoice run) or clutter say.
        with open(os.devnull, "w") as devnull:
            with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
                for frame in self._model.generate(text, ref_audio=ref, stream=True,
                                                  streaming_interval=self.stream_interval):
                    audio = np.asarray(frame.audio, dtype=np.float32).reshape(-1)
                    if len(audio):
                        yield audio


def load(name: str = "phonon", voice: str | None = None, voices: str | None = None) -> TTS:
    if name == "phonon":
        return Phonon(voice=voice)
    if name == "kokoro":
        return Kokoro(voice=voice or "af_heart")
    if name == "chatterbox":
        return Chatterbox(voice=voice, voices=voices)
    raise ValueError(f"unknown TTS {name!r} (phonon | kokoro | chatterbox)")
