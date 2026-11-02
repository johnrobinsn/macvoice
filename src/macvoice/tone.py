"""Soft cues that play through the same output as speech, so the echo
canceller removes them from the mic like any reply.

- The thinking tone: a blip once a second while the agent works on a reply
  and has nothing to say yet, so a long silence doesn't sound like it
  stopped listening.
- The done cue: two rising blips when a reply you talked over finishes, so
  you know the agent is free again."""

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


def done_tone(sample_rate: int, level_db: float = -30.0) -> np.ndarray:
    """Two short rising blips, about a third of a second in all."""
    first = thinking_tone(sample_rate, period_s=0.16, blip_s=0.1, freq_hz=660, level_db=level_db)
    second = thinking_tone(sample_rate, period_s=0.16, blip_s=0.1, freq_hz=880, level_db=level_db)
    return np.concatenate([first, second])
