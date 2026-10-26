"""The model files macvoice needs, fetched up front.

``macvoice download`` gets them all with progress bars, so ``macvoice run``
can start listening the moment it's typed. Every command that loads models
calls :func:`ensure` first, which downloads only what's missing, the same
way, so skipping ``download`` still works.

Everything is pinned to a commit (Silero to a tag plus its SHA-256), so the
sizes below are exact. Hugging Face files go in the usual Hugging Face
cache (``~/.cache/huggingface/hub``), where the libraries that load them
look; Silero goes in ``~/.cache/macvoice``.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import TextIO

from macvoice import asr, tts, turn, vad


@dataclass(frozen=True)
class Asset:
    label: str                  # what it's for, as the user sees it
    filename: str
    size: int                   # bytes
    repo: str = ""              # Hugging Face repo; empty for Silero's URL
    revision: str | None = None

    def cached(self) -> bool:
        if not self.repo:
            return vad.cache_path().exists()
        from huggingface_hub import try_to_load_from_cache

        return isinstance(try_to_load_from_cache(self.repo, self.filename, revision=self.revision), str)

    def fetch(self) -> None:
        if not self.repo:
            vad.model_path()
            return
        from huggingface_hub import hf_hub_download

        hf_hub_download(self.repo, self.filename, revision=self.revision)


def assets(asr_name: str = "parakeet", precision: str = "q8", tts_name: str = "phonon",
           turn_detector: bool = True, need_tts: bool = True) -> list[Asset]:
    """What a command with these options loads. The PyTorch baselines
    (Whisper, Kokoro) download through their own libraries and aren't
    listed."""
    out = silero() + (smart_turn() if turn_detector else [])
    if asr_name == "parakeet":
        out += parakeet(precision)
    if need_tts and tts_name == "phonon":
        out += phonon()
    return out


def silero() -> list[Asset]:
    return [Asset("voice activity (Silero VAD)", "silero_vad.onnx", vad.SIZE)]


def smart_turn() -> list[Asset]:
    return [Asset("end of turn (Smart Turn)", turn.ONNX_FILENAME, 8679182, turn.HF_REPO, turn.HF_REVISION)]


def parakeet(precision: str = "q8") -> list[Asset]:
    if precision == "q8":
        label, repo, rev, weights = "speech to text (Parakeet, 8-bit)", asr.PARAKEET_Q8, asr.PARAKEET_Q8_REVISION, 736486649
    else:  # converted from the fp32 checkpoint at load
        label, repo, rev, weights = "speech to text (Parakeet)", asr.PARAKEET_MODEL, asr.PARAKEET_REVISION, 2471559904
    return [Asset(label, "config.json", 36176, repo, rev), Asset(label, "model.safetensors", weights, repo, rev)]


def phonon() -> list[Asset]:
    """The files ptts loads for the default voice. Other voices are a few
    hundred kB each, fetched by ptts when first used."""
    return [Asset("text to speech (Phonon)", f, n, tts.PHONON_REPO, tts.PHONON_REVISION) for f, n in (
        ("config.json", 2851), ("tokenizer.json", 277967),
        ("model.safetensors", 102823920), ("default-voice.safetensors", 256136))]


def _groups(items: list[Asset]) -> dict[str, int]:
    """Total bytes per label, in order."""
    sizes: dict[str, int] = {}
    for a in items:
        sizes[a.label] = sizes.get(a.label, 0) + a.size
    return sizes


def human(n: int) -> str:
    return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB" if n >= 1e6 else f"{n / 1e3:.0f} kB"


def ensure(items: list[Asset], out: TextIO = sys.stderr) -> int:
    """Download whatever in ``items`` isn't cached yet, saying what and how
    big first. Silent when everything is there. Returns bytes downloaded."""
    missing = [a for a in items if not a.cached()]
    if not missing:
        return 0
    total = sum(a.size for a in missing)
    print(f"Downloading models, {human(total)} (one time only):", file=out)
    for label, size in _groups(missing).items():
        print(f"  {label:36} {human(size):>8}", file=out)
    out.flush()
    for a in missing:
        a.fetch()
    print(file=out)
    return total


def download(items: list[Asset], out: TextIO = sys.stdout) -> None:
    """``macvoice download``: fetch what's missing and say when it's all there."""
    have = [label for label in _groups(items) if all(a.cached() for a in items if a.label == label)]
    total = human(sum(a.size for a in items))
    if len(have) == len(_groups(items)):
        print(f"All models are already downloaded ({total}). Next: macvoice run", file=out)
        return
    if have:
        print("Already downloaded: " + ", ".join(have), file=out)
    ensure(items, out)
    print(f"All models ready ({total}). Next: macvoice run", file=out)
