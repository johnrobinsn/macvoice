"""Synthesize the conversational test prompts in tests/fixtures/prompts/.

Each TURN_ITEMS entry gives a complete request and a request split in two
(start, continuation) that the benchmarks join with a pause. Spoken by four
Phonon voices, resampled to 16 kHz, silence trimmed to 50 ms at each end. Committed so tests and benchmarks don't
depend on TTS output staying bit-identical across ptts versions.

    uv run python scripts/make_prompt_fixtures.py
"""

import json
import sys
from pathlib import Path

import numpy as np

from macvoice import SAMPLE_RATE
from macvoice.audio import resample, write_wav
from macvoice.bench import TURN_ITEMS, TURN_VOICES
from macvoice.tts import Phonon
from macvoice.turn import trim_trailing_silence

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "prompts"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    clips = []
    for voice in TURN_VOICES:
        engine = Phonon(voice=voice)
        name = (voice or "default").lower()
        for i, (complete, (head, tail)) in enumerate(TURN_ITEMS):
            for kind, text in (("complete", complete), ("head", head), ("tail", tail)):
                audio = np.concatenate(list(engine.stream(text)))
                audio = resample(audio, engine.sample_rate, SAMPLE_RATE)
                # Phonon pads ~0.1-0.2 s before and ~0.5-0.7 s after; cut both to
                # 50 ms so the pauses the benchmarks insert are the real pauses.
                audio = trim_trailing_silence(trim_trailing_silence(audio)[::-1])[::-1].copy()
                peak = float(np.max(np.abs(audio))) or 1.0
                audio = audio * min(1.0, 0.9 / peak)
                wav = f"{name}_{i}_{kind}.wav"
                write_wav(OUT / wav, audio, SAMPLE_RATE)
                clips.append({"wav": wav, "voice": name, "item": i, "kind": kind, "text": text})
    manifest = {"source": "Phonon 31M English (gradium/phonon-31m-en, Apache-2.0) via ptts",
                "sample_rate_hz": SAMPLE_RATE, "clips": clips}
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"{len(clips)} clips -> {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
