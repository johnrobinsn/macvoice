"""The real models, and the whole loop on real audio."""

import subprocess
import sys

import numpy as np
import pytest

from macvoice.agent import CollectingOut, Responder, simulate
from macvoice.audio import resample
from macvoice.bench import normalize, wer
from macvoice.endpointing import Endpointer
from macvoice.llm import Echo

pytestmark = pytest.mark.models


def test_parakeet_accuracy_on_librispeech(parakeet, librispeech):
    hyps = [parakeet.transcribe(c["audio"]) for c in librispeech]
    assert wer([c["transcript"] for c in librispeech], hyps) < 0.06


def test_parakeet_is_fast(parakeet, librispeech):
    import time
    audio = librispeech[0]["audio"]  # 4.8 s
    parakeet.transcribe(audio)
    t0 = time.perf_counter()
    parakeet.transcribe(audio)
    assert time.perf_counter() - t0 < 0.5


def test_parakeet_on_silence_returns_no_words(parakeet):
    text = parakeet.transcribe(np.zeros(16000, dtype=np.float32))
    assert not any(ch.isalnum() for ch in text)


def test_parakeet_rejects_unknown_precision():
    from macvoice.asr import Parakeet
    with pytest.raises(ValueError):
        Parakeet(precision="q3")


def test_phonon_streams_quickly(phonon):
    import time
    list(phonon.stream("Warm up."))
    t0 = time.perf_counter()
    stream = phonon.stream("The quick brown fox jumps over the lazy dog, twice.")
    first = next(iter(stream))
    ttfa = time.perf_counter() - t0
    rest = list(stream)
    assert phonon.sample_rate == 24000
    assert first.dtype == np.float32 and len(first) > 0
    assert ttfa < 0.15
    total = sum(len(c) for c in [first, *rest]) / phonon.sample_rate
    assert 1.5 < total < 8


def test_phonon_round_trip_through_parakeet(phonon, parakeet):
    text = "Please remind me to water the plants tomorrow morning."
    audio = np.concatenate(list(phonon.stream(text)))
    heard = parakeet.transcribe(resample(audio, phonon.sample_rate, 16000))
    assert normalize(heard) == normalize(text)


def test_full_loop_on_conversational_prompts(vad, smart_turn, parakeet, phonon, prompts):
    """Every complete prompt: one turn, ended by Smart Turn ~300 ms after
    speech, transcribed correctly, answered with audio well under a second
    after the user stopped (echo LLM)."""
    from macvoice.bench import TURN_ITEMS

    responder = Responder(parakeet, Echo(), phonon)
    totals = []
    for (voice, item, kind), audio in sorted(prompts.items()):
        if kind != "complete":
            continue
        vad.reset()
        out = CollectingOut()
        results = simulate(audio, vad, Endpointer(smart_turn), responder, out)
        assert len(results) == 1, (voice, item)
        event, m = results[0]
        assert event.reason == "detector"
        assert wer([TURN_ITEMS[item][0]], [m.transcript]) <= 0.15, (voice, m.transcript)
        assert len(out.audio) / phonon.sample_rate > 1.0
        totals.append(m.speech_end_to_first_audio_ms)
    assert len(totals) == 32
    assert np.median(totals) < 600


def test_filled_pause_does_not_end_the_turn(vad, smart_turn, prompts):
    """'I'd like to book a table for, um,' + 2 s + 'four people at seven.'
    is one turn with Smart Turn, two with a 1.5 s silence timer."""
    from macvoice.endpointing import EndpointConfig

    gap = np.zeros(32000, dtype=np.float32)
    audio = np.concatenate([prompts[("default", 0, "head")], gap, prompts[("default", 0, "tail")]])
    vad.reset()
    assert len(simulate(audio, vad, Endpointer(smart_turn))) == 1
    vad.reset()
    assert len(simulate(audio, vad, Endpointer(None, EndpointConfig(max_silence_ms=1500)))) == 2


def _cli(*args, timeout=300):
    return subprocess.run([sys.executable, "-m", "macvoice.cli", *args], capture_output=True,
                          text=True, timeout=timeout)


def test_cli_simulate_and_say(tmp_path):
    from tests.conftest import FIXTURES

    r = _cli("simulate", "--llm", "echo", str(FIXTURES / "prompts/harper_2_complete.wav"),
             "-o", str(tmp_path / "r.wav"))
    assert r.returncode == 0, r.stderr
    assert "brain: echo" in r.stdout
    assert "How many days are in a leap year?" in r.stdout
    assert "first audio" in r.stdout
    assert (tmp_path / "r.wav").stat().st_size > 10_000

    r = _cli("say", "Testing the speaker now.", "-o", str(tmp_path / "s.wav"))
    assert r.returncode == 0, r.stderr
    assert "first audio" in r.stdout

    r = _cli("transcribe", str(tmp_path / "s.wav"))
    assert r.returncode == 0, r.stderr
    assert normalize(r.stdout.split("\t")[-1]) == ["testing", "the", "speaker", "now"]
