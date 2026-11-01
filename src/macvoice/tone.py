"""The thinking tone: a soft blip once a second while the agent works on
a reply and has nothing to say yet, so a long silence doesn't sound like
it stopped listening. It plays through the same output as speech, so the
echo canceller removes it from the mic like any reply."""

from __future__ import annotations

import numpy as np


def thinking_tone(sample_rate: int, period_s: float = 1.0, blip_s: float = 0.12,
                  freq_hz: float = 660.0, level_db: float = -32.0) -> np.ndarray:
    """One period: a short sine blip with smooth edges, then silence."""
    n = int(min(blip_s, period_s) * sample_rate)
    t = np.arange(n) / sample_rate
    envelope = np.sin(np.pi * np.arange(n) / n) ** 2  # no clicks at either end
    blip = (10 ** (level_db / 20)) * envelope * np.sin(2 * np.pi * freq_hz * t)
    out = np.zeros(int(period_s * sample_rate), dtype=np.float32)
    out[:n] = blip
    return out
