"""Turn endpointing: when has the user finished speaking?

A fixed silence timer has to choose between two bad options. Short, and it
cuts people off when they pause to think ("I'd like to book a table for,
um..."). Long, and every reply waits for it. The usual fix, and the one
here, is to ask an end-of-turn model after a *short* silence:

* speech, then ``check_silence_ms`` of silence -> run the detector on the
  turn's audio, trimmed to where speech stopped;
* P(complete) >= ``threshold`` -> the turn is over now;
* otherwise keep listening; the turn ends anyway after ``max_silence_ms``,
  and if speech resumes first it continues the same turn.

With ``detector=None`` the endpointer is a plain silence timer that ends the
turn after ``max_silence_ms``, the baseline the benchmarks compare against.

The class is pure (no audio devices, no threads, no clock): feed it 32 ms
chunks with their VAD probability, in order, and it reports turn events in
audio time. Everything is unit-testable with synthetic probabilities.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from macvoice import SAMPLE_RATE
from macvoice.turn import trim_trailing_silence
from macvoice.vad import CHUNK

CHUNK_MS = CHUNK * 1000 / SAMPLE_RATE  # 32 ms


@dataclass
class EndpointConfig:
    speech_prob: float = 0.5        # VAD prob that counts as speech
    silence_prob: float = 0.35      # below this counts as silence (hysteresis between)
    start_ms: float = 96            # consecutive speech needed to open a turn
    pre_roll_ms: float = 320        # audio kept from before the onset
    min_voiced_ms: float = 250      # shorter turns are dropped as noise
    check_silence_ms: float = 300   # silence before the turn detector is asked
    threshold: float = 0.5          # P(complete) that ends the turn early
    max_silence_ms: float = 3000    # the turn ends after this much silence regardless
    tail_ms: float = 200            # silence kept after speech in the audio handed to ASR
    max_turn_s: float = 30.0


@dataclass
class TurnEvent:
    kind: str                       # "start" | "end" | "discard" (too short: noise)
    time_s: float                   # audio time at which the event fired
    speech_end_s: float = 0.0       # audio time of the last speech chunk ("end" only)
    audio: np.ndarray | None = None  # the turn, pre-roll to speech end + tail ("end" only)
    p_complete: float | None = None  # the detector's last answer, if it was asked
    reason: str = ""                # "detector" | "silence" | "max_turn"
    detector_calls: int = 0


@dataclass
class _Turn:
    chunks: list[np.ndarray] = field(default_factory=list)
    voiced: int = 0                 # speech chunks in the turn
    silence: int = 0                # consecutive non-speech chunks at the end
    last_speech_idx: int = 0        # index into chunks of the last speech chunk
    asked: bool = False             # detector already asked during this pause
    p_complete: float | None = None
    detector_calls: int = 0


class Endpointer:
    def __init__(self, detector: Callable[[np.ndarray], float] | None = None,
                 config: EndpointConfig | None = None) -> None:
        self.detector = detector
        self.cfg = config or EndpointConfig()
        c = self.cfg
        n = lambda ms: max(1, int(round(ms / CHUNK_MS)))  # noqa: E731
        self._start_n = n(c.start_ms)
        self._check_n = n(c.check_silence_ms)
        self._max_silence_n = n(c.max_silence_ms)
        self._min_voiced_n = n(c.min_voiced_ms)
        self._tail_n = int(round(c.tail_ms / CHUNK_MS))
        self._max_turn_n = n(c.max_turn_s * 1000)
        self._pre_roll: deque[np.ndarray] = deque(maxlen=n(c.pre_roll_ms))
        self.reset()

    def reset(self) -> None:
        self._pre_roll.clear()
        self._onset = 0
        self._speaking = False
        self._turn: _Turn | None = None
        self._idx = 0  # chunks seen

    @property
    def in_turn(self) -> bool:
        return self._turn is not None

    def _now(self) -> float:
        return self._idx * CHUNK_MS / 1000

    def push(self, chunk: np.ndarray, prob: float) -> TurnEvent | None:
        """Feed one 512-sample chunk and its VAD probability."""
        c = self.cfg
        self._idx += 1
        if prob >= c.speech_prob:
            self._speaking = True
        elif prob < c.silence_prob:
            self._speaking = False
        speech = self._speaking

        if self._turn is None:
            self._pre_roll.append(chunk)
            self._onset = self._onset + 1 if speech else 0
            if self._onset < self._start_n:
                return None
            self._turn = _Turn(chunks=list(self._pre_roll), voiced=self._onset)
            self._turn.last_speech_idx = len(self._turn.chunks) - 1
            self._pre_roll.clear()
            self._onset = 0
            return TurnEvent("start", self._now())

        t = self._turn
        t.chunks.append(chunk)
        if speech:
            t.voiced += 1
            t.silence = 0
            t.last_speech_idx = len(t.chunks) - 1
            t.asked = False
            if len(t.chunks) >= self._max_turn_n:
                return self._end("max_turn")
            return None

        t.silence += 1
        if self.detector is not None and not t.asked and t.silence >= self._check_n:
            t.asked = True
            t.detector_calls += 1
            speech_audio = np.concatenate(t.chunks[: t.last_speech_idx + 1])
            t.p_complete = float(self.detector(trim_trailing_silence(speech_audio)))
            if t.p_complete >= c.threshold:
                return self._end("detector")
        if t.silence >= self._max_silence_n:
            return self._end("silence")
        if len(t.chunks) >= self._max_turn_n:
            return self._end("max_turn")
        return None

    def _end(self, reason: str) -> TurnEvent:
        t = self._turn
        self._turn = None
        self._speaking = False
        if t.voiced < self._min_voiced_n:
            return TurnEvent("discard", self._now(), reason=reason)
        speech_end_s = (self._idx - (len(t.chunks) - 1 - t.last_speech_idx)) * CHUNK_MS / 1000
        audio = np.concatenate(t.chunks[: t.last_speech_idx + 1 + self._tail_n])
        return TurnEvent("end", self._now(), speech_end_s=speech_end_s, audio=audio,
                         p_complete=t.p_complete, reason=reason, detector_calls=t.detector_calls)
