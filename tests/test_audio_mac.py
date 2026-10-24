"""Apple voice processing (--aec).

The ``devices`` tests use the real speaker and microphone, and play sound.
Run them in a quiet room:

    MACVOICE_DEVICE_TESTS=1 uv run pytest -m devices

macOS's echo canceller removes everything the Mac plays through its
speakers, not just our engine's output (``afplay`` is cancelled too), so a
"person" can't be simulated by playing a clip on the same speakers. The
automated loop test mixes the person into the echo-cancelled mic signal
instead; :func:`test_talk_over_it` (``MACVOICE_HUMAN_TESTS=1 ... -s``) asks
you to do it for real.
"""

import json
import os
import queue
import re
import subprocess
import sys
import threading
import time

import numpy as np
import pytest

from macvoice import SAMPLE_RATE
from macvoice.audio import read_wav
from macvoice.vad import CHUNK
from tests.conftest import FIXTURES

pytestmark = pytest.mark.skipif(sys.platform != "darwin", reason="AVFoundation is macOS only")

PROMPTS = FIXTURES / "prompts"
REPLY = PROMPTS / "default_6_complete.wav"      # what the agent "says"
QUESTION = "default_6_complete.wav"            # "Who painted the Mona Lisa?"
PERSON = "harper_0_complete.wav"                # someone talking over the reply, another voice
TEXT = {c["wav"]: c["text"] for c in json.loads((PROMPTS / "manifest.json").read_text())["clips"]}
LONG_ANSWER = "Here is a long answer so that there is time to interrupt it. " * 6


def words(text: str) -> set[str]:
    return set(re.findall(r"[a-z']+", text.lower()))


def dbfs(x: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(np.square(x))) + 1e-9))


def test_busy_until_stopped():
    from macvoice.audio_mac import VoiceProcessingIO

    io = VoiceProcessingIO(24000)  # engine not started: nothing plays, nothing completes
    assert not io.busy
    io.write(np.zeros(2400, dtype=np.float32))
    io.write(np.zeros(0, dtype=np.float32))  # ignored
    assert io.busy
    io.stop()
    assert not io.busy


class Recorder:
    def __init__(self):
        self.chunks, self.lock = [], threading.Lock()

    def __call__(self, chunk):
        assert chunk.shape == (CHUNK,) and chunk.dtype == np.float32
        with self.lock:
            self.chunks.append(chunk)

    def mark(self) -> int:
        with self.lock:
            return len(self.chunks)

    def between(self, a: int, b: int | None = None) -> np.ndarray:
        with self.lock:
            return np.concatenate(self.chunks[a:b])


@pytest.fixture
def running_io():
    from macvoice.audio_mac import VoiceProcessingIO

    io = VoiceProcessingIO(24000)
    rec = Recorder()
    io.start(rec)
    time.sleep(1.0)  # voice processing settles
    yield io, rec
    io.close()


def longest_speech_ms(vad, audio: np.ndarray) -> float:
    """The longest run of speech the VAD hears; a barge-in needs 250 ms."""
    audio = audio[: len(audio) // CHUNK * CHUNK]
    run = best = 0
    for c in audio.reshape(-1, CHUNK):
        run = run + 1 if vad(c) > 0.5 else 0
        best = max(best, run)
    return best * CHUNK * 1000 / SAMPLE_RATE


@pytest.mark.devices
@pytest.mark.models
def test_own_reply_is_cancelled(running_io, vad):
    io, rec = running_io
    reply, sr = read_wav(REPLY, 24000)
    reply = np.tile(reply, 2) * 0.7
    start = rec.mark()
    io.write(reply)
    assert io.busy
    time.sleep(len(reply) / sr + 0.3)
    assert not io.busy
    mic = rec.between(start)
    assert len(mic) > 0.9 * len(reply) / sr * SAMPLE_RATE  # the mic kept running at 16 kHz
    assert dbfs(mic) < -50, f"echo at {dbfs(mic):.1f} dBFS"
    # A little echo can leak while the canceller adapts to the start of the
    # reply; it must stay well short of what stops a reply.
    assert longest_speech_ms(vad, mic) < 200, "the VAD heard the agent's own voice"


@pytest.mark.devices
@pytest.mark.models
def test_other_mac_audio_is_cancelled_too(running_io, vad):
    """Voice processing references everything going to the speakers, so a
    video or music playing on the Mac doesn't trigger the agent either."""
    _, rec = running_io
    start = rec.mark()
    subprocess.run(["afplay", str(PROMPTS / PERSON)], check=True)
    assert longest_speech_ms(vad, rec.between(start)) < 200


class MixingMic:
    """The echo-cancelled microphone with a "person" added digitally, after
    echo cancellation: the reply still plays out loud and is still cancelled,
    but the person's words arrive as they would from someone in the room."""

    def __init__(self, io):
        self.io, self._person = io, queue.Queue()
        self._current, self._pos = None, 0

    def say(self, audio: np.ndarray) -> float:
        self._person.put(audio)
        return len(audio) / SAMPLE_RATE

    def _next(self) -> np.ndarray:
        out = np.zeros(CHUNK, dtype=np.float32)
        if self._current is None and not self._person.empty():
            self._current, self._pos = self._person.get(), 0
        if self._current is not None:
            piece = self._current[self._pos:self._pos + CHUNK]
            out[: len(piece)] = piece
            self._pos += CHUNK
            if self._pos >= len(self._current):
                self._current = None
        return out

    def start(self, on_chunk):
        self.io.start(lambda chunk: on_chunk(chunk + self._next()))

    def close(self):
        self.io.close()


class LongReply:
    name = "long"

    def reply(self, history):
        yield LONG_ANSWER


def start_agent(vad, smart_turn, parakeet, phonon, mic_factory, turns: list, max_turns: int):
    from macvoice.agent import Responder, VoiceAgent
    from macvoice.audio_mac import VoiceProcessingIO
    from macvoice.endpointing import Endpointer

    io = VoiceProcessingIO(phonon.sample_rate)
    mic = mic_factory(io)
    agent = VoiceAgent(vad, Endpointer(smart_turn), Responder(parakeet, LongReply(), phonon), io,
                       barge_in=True, echo_guard_s=0.0, on_turn=turns.append)
    runner = threading.Thread(target=agent.run, args=(mic,), kwargs={"max_turns": max_turns},
                              daemon=True)
    runner.start()
    return agent, io, mic, runner


def wait_for(cond, timeout: float) -> bool:
    deadline = time.time() + timeout
    while not cond() and time.time() < deadline:
        time.sleep(0.05)
    return cond()


@pytest.mark.devices
@pytest.mark.models
def test_live_loop_does_not_interrupt_itself_and_stops_when_interrupted(
        vad, smart_turn, parakeet, phonon):
    turns = []
    agent, io, mic, runner = start_agent(vad, smart_turn, parakeet, phonon, MixingMic, turns,
                                         max_turns=2)
    time.sleep(1.5)

    question = read_wav(PROMPTS / QUESTION, SAMPLE_RATE)[0]
    time.sleep(mic.say(question))
    assert wait_for(lambda: io.busy, 10), "no reply started"
    # The reply (about 20 s) plays on the laptop speakers; with echo
    # cancellation the agent must not hear it and cut itself off.
    time.sleep(4.0)
    assert agent.interruptions == 0, "the agent interrupted itself"
    assert io.busy, f"the reply stopped early: {turns}"

    person = read_wav(PROMPTS / PERSON, SAMPLE_RATE)[0]
    t0 = time.time()
    mic.say(person)
    assert wait_for(lambda: not io.busy, 2.0), "talking over the reply didn't stop it"
    stopped_after = time.time() - t0
    assert wait_for(lambda: len(turns) == 2, 10), "the interruption didn't become a turn"
    io.stop()  # skip the second (long) reply
    runner.join(timeout=5)
    assert not runner.is_alive()

    assert agent.interruptions == 1
    assert stopped_after < 0.5, f"barge-in took {stopped_after:.2f} s"
    first, second = turns
    assert words(first.transcript) >= {"mona", "lisa"}, first.transcript
    assert len(words(second.transcript) & words(TEXT[PERSON])) >= 4, second.transcript
    assert "interrupt" not in second.transcript.lower(), "the agent heard itself"


@pytest.mark.devices
@pytest.mark.models
@pytest.mark.skipif(not os.environ.get("MACVOICE_HUMAN_TESTS"), reason="needs you: MACVOICE_HUMAN_TESTS=1 -s")
def test_talk_over_it(vad, smart_turn, parakeet, phonon):
    turns = []
    agent, io, _, runner = start_agent(vad, smart_turn, parakeet, phonon, lambda io: io, turns,
                                       max_turns=2)
    time.sleep(1.5)
    print("\n>>> Ask a question out loud. When it starts its (long) answer, wait a moment "
          "and then talk over it.", flush=True)
    runner.join(timeout=60)
    for m in turns:
        print(f"    you: {m.transcript!r}  reply {m.reply_audio_s:.1f} s  cancelled={m.cancelled}")
    assert len(turns) == 2 and agent.interruptions == 1
    assert turns[1].transcript and "interrupt" not in turns[1].transcript.lower()
