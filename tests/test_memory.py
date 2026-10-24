import subprocess
import sys

import pytest

from macvoice.memory import footprint_mb, peak_footprint_mb, torch_loaded
from tests.conftest import APPLE_SILICON


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS footprint API")
def test_footprint_is_sane():
    now, peak = footprint_mb(), peak_footprint_mb()
    assert 5 < now < 100_000
    assert peak >= now * 0.99


def test_torch_loaded_reflects_sys_modules(monkeypatch):
    monkeypatch.delitem(sys.modules, "torch", raising=False)
    assert not torch_loaded()
    monkeypatch.setitem(sys.modules, "torch", object())
    assert torch_loaded()


def _run(code: str) -> str:
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                          check=True, timeout=600).stdout.strip()


def test_importing_everything_does_not_import_torch():
    out = _run("import sys, macvoice.agent, macvoice.asr, macvoice.bench, macvoice.cli, "
               "macvoice.endpointing, macvoice.llm, macvoice.tts, macvoice.turn, macvoice.vad; "
               "print('torch' in sys.modules, 'transformers' in sys.modules)")
    assert out == "False False"


@pytest.mark.models
@pytest.mark.skipif(not APPLE_SILICON, reason="Apple Silicon")
def test_full_voice_stack_stays_under_2_gb_without_torch():
    """The headline claim: every model loaded and a turn processed, under
    2 GB of physical footprint (GPU memory included), with no PyTorch."""
    code = """
import sys
from macvoice.agent import CollectingOut, Responder, simulate
from macvoice.asr import Parakeet
from macvoice.bench import load_prompts
from macvoice.endpointing import Endpointer
from macvoice.llm import Echo
from macvoice.memory import peak_footprint_mb
from macvoice.tts import Phonon
from macvoice.turn import SmartTurn
from macvoice.vad import SileroVAD
vad = SileroVAD()
responder = Responder(Parakeet(), Echo(), Phonon())
detector = SmartTurn()
for (voice, item, kind), audio in sorted(load_prompts().items())[:12]:
    if kind == "complete":
        vad.reset()
        simulate(audio, vad, Endpointer(detector), responder, CollectingOut())
print(round(peak_footprint_mb()), 'torch' in sys.modules)
"""
    peak, torch = _run(code).split()
    assert torch == "False"
    assert int(peak) < 2048, f"peak footprint {peak} MB"
