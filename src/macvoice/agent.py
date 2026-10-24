"""The voice loop.

    mic --32 ms chunks--> VAD --> Endpointer --turn audio--> Responder
                                                   ASR -> LLM -> sentences -> TTS -> speaker

:class:`Responder` handles one finished turn synchronously and records
where the time went. :class:`VoiceAgent` runs it live: the microphone
thread feeds chunks, replies run on a worker thread, and the mic is muted
while the agent speaks (half duplex) unless ``barge_in`` is on, in which
case the user starting to talk stops the reply. Use barge-in with
headphones: on laptop speakers the mic hears the agent.

:func:`simulate` drives the same pieces from a recording with no devices,
for benchmarks and tests.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np

from macvoice import SAMPLE_RATE
from macvoice.audio import frames
from macvoice.endpointing import Endpointer, TurnEvent
from macvoice.llm import LLM, sentences
from macvoice.tts import TTS
from macvoice.vad import CHUNK

log = logging.getLogger("macvoice")


class AudioOut(Protocol):
    def write(self, audio: np.ndarray) -> None: ...
    def stop(self) -> None: ...
    @property
    def busy(self) -> bool: ...


@dataclass
class TurnMetrics:
    """Milliseconds unless noted. ``endpoint_ms`` is how long after the user
    stopped speaking the turn was declared over (silence waited plus the
    detector's compute); the other times run from that moment."""

    transcript: str = ""
    reply: str = ""
    endpoint_ms: float = 0.0
    endpoint_reason: str = ""
    p_complete: float | None = None
    asr_ms: float = 0.0
    llm_first_sentence_ms: float | None = None   # after ASR
    tts_first_audio_ms: float | None = None      # after the first sentence
    reply_audio_s: float = 0.0
    tts_synth_ms: float = 0.0
    cancelled: bool = False
    timeline: dict[str, float] = field(default_factory=dict)

    @property
    def speech_end_to_first_audio_ms(self) -> float | None:
        """What the user feels: from the end of their speech to the first
        sound of the reply."""
        if self.llm_first_sentence_ms is None or self.tts_first_audio_ms is None:
            return None
        return self.endpoint_ms + self.asr_ms + self.llm_first_sentence_ms + self.tts_first_audio_ms


class Responder:
    def __init__(self, asr, llm: LLM, tts: TTS) -> None:
        self.asr, self.llm, self.tts = asr, llm, tts
        self.history: list[dict] = []

    def respond(self, audio: np.ndarray, out: AudioOut, cancel: threading.Event | None = None,
                metrics: TurnMetrics | None = None) -> TurnMetrics:
        m = metrics or TurnMetrics()
        cancel = cancel or threading.Event()
        t0 = time.perf_counter()
        m.transcript = self.asr.transcribe(audio)
        m.asr_ms = (time.perf_counter() - t0) * 1000
        if not m.transcript or not any(ch.isalnum() for ch in m.transcript):
            return m  # Parakeet returns "." for breath and noise
        self.history.append({"role": "user", "content": m.transcript})

        t_llm = time.perf_counter()
        spoken: list[str] = []
        try:
            for sentence in sentences(self.llm.reply(self.history)):
                if cancel.is_set():
                    break
                if m.llm_first_sentence_ms is None:
                    m.llm_first_sentence_ms = (time.perf_counter() - t_llm) * 1000
                spoken.append(sentence)
                t_tts = time.perf_counter()
                for chunk in self.tts.stream(sentence):
                    if cancel.is_set():
                        break
                    if m.tts_first_audio_ms is None:
                        m.tts_first_audio_ms = (time.perf_counter() - t_tts) * 1000
                    m.reply_audio_s += len(chunk) / self.tts.sample_rate
                    out.write(chunk)
                m.tts_synth_ms += (time.perf_counter() - t_tts) * 1000
        finally:
            m.cancelled = cancel.is_set()
            m.reply = " ".join(spoken)
            if spoken:
                self.history.append({"role": "assistant", "content": m.reply})
        return m


class VoiceAgent:
    """Live loop. ``feed`` takes 512-sample 16 kHz chunks from any thread;
    :meth:`run` wires it to the default microphone and speaker."""

    def __init__(self, vad: Callable[[np.ndarray], float], endpointer: Endpointer,
                 responder: Responder, out: AudioOut, barge_in: bool = False,
                 echo_guard_s: float = 0.3,
                 on_turn: Callable[[TurnMetrics], None] | None = None) -> None:
        self.vad, self.endpointer, self.responder, self.out = vad, endpointer, responder, out
        self.barge_in, self.echo_guard_s = barge_in, echo_guard_s
        self.on_turn = on_turn
        self._turns: queue.Queue[tuple[TurnEvent, float] | None] = queue.Queue()
        self._cancel = threading.Event()
        self._replying = threading.Event()
        self._muted_until = 0.0
        self._worker = threading.Thread(target=self._work, name="macvoice-responder", daemon=True)
        self._worker.start()

    @property
    def speaking(self) -> bool:
        return self._replying.is_set() or self.out.busy

    def feed(self, chunk: np.ndarray, now: float | None = None) -> TurnEvent | None:
        now = time.monotonic() if now is None else now
        if self.speaking:
            self._muted_until = now + self.echo_guard_s
            if not self.barge_in:
                return None
        elif now < self._muted_until:
            return None  # the room's echo of the reply's last words
        event = self.endpointer.push(chunk, self.vad(chunk))
        if event is None:
            return None
        if event.kind == "start" and self.speaking and self.barge_in:
            log.info("barge-in: stopping the reply")
            self._cancel.set()
            self.out.stop()
        elif event.kind == "end":
            self._turns.put((event, now))
        return event

    def _work(self) -> None:
        while (item := self._turns.get()) is not None:
            event, fired = item
            self._cancel = threading.Event()
            self._replying.set()
            try:
                m = TurnMetrics(endpoint_ms=(event.time_s - event.speech_end_s) * 1000,
                                endpoint_reason=event.reason, p_complete=event.p_complete)
                # Time the turn waited in the queue counts against the reply.
                m.timeline["queued_ms"] = (time.monotonic() - fired) * 1000
                self.responder.respond(event.audio, self.out, self._cancel, m)
                if self.on_turn:
                    self.on_turn(m)
            except Exception:
                log.exception("reply failed")
            finally:
                self._replying.clear()

    def close(self) -> None:
        self._cancel.set()
        self._turns.put(None)
        self._worker.join(timeout=5)

    def run(self, input_device=None, output_device=None) -> None:  # pragma: no cover - devices
        """Blocking: microphone in, speaker out, until Ctrl-C."""
        import sounddevice as sd

        chunks: queue.Queue[np.ndarray] = queue.Queue()

        def on_audio(indata, n, t, status):
            chunks.put(indata[:, 0].copy())

        with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32", blocksize=CHUNK,
                            device=input_device, callback=on_audio):
            log.info("listening (Ctrl-C to quit)")
            try:
                while True:
                    self.feed(chunks.get())
            except KeyboardInterrupt:
                pass
            finally:
                self.close()


class SpeakerOut:  # pragma: no cover - devices
    """Plays chunks on an output device as they arrive; ``stop`` flushes."""

    def __init__(self, sample_rate: int, device=None) -> None:
        import sounddevice as sd

        self._q: queue.Queue[np.ndarray] = queue.Queue()
        self._buf = np.zeros(0, dtype=np.float32)
        self._lock = threading.Lock()
        self._stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32",
                                       device=device, callback=self._callback)
        self._stream.start()

    def _callback(self, outdata, n, t, status) -> None:
        with self._lock:
            while len(self._buf) < n and not self._q.empty():
                self._buf = np.concatenate([self._buf, self._q.get_nowait()])
            take = self._buf[:n]
            self._buf = self._buf[n:]
        outdata[: len(take), 0] = take
        outdata[len(take):, 0] = 0

    def write(self, audio: np.ndarray) -> None:
        self._q.put(np.asarray(audio, dtype=np.float32))

    def stop(self) -> None:
        with self._lock:
            self._buf = np.zeros(0, dtype=np.float32)
            while not self._q.empty():
                self._q.get_nowait()

    @property
    def busy(self) -> bool:
        with self._lock:
            return len(self._buf) > 0 or not self._q.empty()


class CollectingOut:
    """An AudioOut that keeps what it's given (tests, benchmarks)."""

    def __init__(self) -> None:
        self.chunks: list[np.ndarray] = []
        self.stopped = 0

    def write(self, audio: np.ndarray) -> None:
        self.chunks.append(np.asarray(audio, dtype=np.float32))

    def stop(self) -> None:
        self.stopped += 1

    @property
    def busy(self) -> bool:
        return False

    @property
    def audio(self) -> np.ndarray:
        return np.concatenate(self.chunks) if self.chunks else np.zeros(0, dtype=np.float32)


def simulate(audio: np.ndarray, vad: Callable[[np.ndarray], float], endpointer: Endpointer,
             responder: Responder | None = None, out: AudioOut | None = None,
             trailing_silence_s: float = 2.5) -> list[tuple[TurnEvent, TurnMetrics]]:
    """Run a 16 kHz recording through VAD, endpointing and (optionally) the
    responder, as fast as the models allow. Endpoint latency is measured in
    audio time (the silence waited) plus the detector's wall time on the
    chunk that ended the turn, so results don't depend on playback speed."""
    audio = np.concatenate([np.asarray(audio, dtype=np.float32),
                            np.zeros(int(trailing_silence_s * SAMPLE_RATE), dtype=np.float32)])
    out = out or CollectingOut()
    results = []
    for chunk in frames(audio, CHUNK):
        prob = vad(chunk)
        t0 = time.perf_counter()
        event = endpointer.push(chunk, prob)
        push_ms = (time.perf_counter() - t0) * 1000
        if event is None or event.kind != "end":
            continue
        m = TurnMetrics(endpoint_ms=(event.time_s - event.speech_end_s) * 1000 + push_ms,
                        endpoint_reason=event.reason, p_complete=event.p_complete)
        m.timeline["detector_ms"] = push_ms
        if responder is not None:
            responder.respond(event.audio, out, metrics=m)
        results.append((event, m))
    return results
