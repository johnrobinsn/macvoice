"""Shared fixtures.

Tests marked ``models`` load the real models from the Hugging Face cache
(and Silero's ONNX file from ~/.cache/macvoice). They run on Apple Silicon
by default and are skipped elsewhere; ``-m "not models"`` skips them
anywhere.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import numpy as np
import pytest

FIXTURES = Path(__file__).parent / "fixtures"
APPLE_SILICON = sys.platform == "darwin" and platform.machine() == "arm64"


def pytest_collection_modifyitems(config, items):
    if APPLE_SILICON:
        return
    skip = pytest.mark.skip(reason="needs Apple Silicon (MLX, Metal)")
    for item in items:
        if "models" in item.keywords:
            item.add_marker(skip)


def tone(freq: float, seconds: float, rate: int = 16000, amp: float = 0.5) -> np.ndarray:
    t = np.arange(int(seconds * rate)) / rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.fixture(scope="session")
def vad():
    from macvoice.vad import SileroVAD
    return SileroVAD()


@pytest.fixture(scope="session")
def smart_turn():
    from macvoice.turn import SmartTurn
    return SmartTurn()


@pytest.fixture(scope="session")
def parakeet():
    from macvoice.asr import Parakeet
    return Parakeet()


@pytest.fixture(scope="session")
def phonon():
    from macvoice.tts import Phonon
    return Phonon()


@pytest.fixture(scope="session")
def prompts():
    from macvoice.bench import load_prompts
    return load_prompts()


@pytest.fixture(scope="session")
def librispeech():
    from macvoice.bench import load_fixtures
    return load_fixtures()
