"""Responder, VoiceAgent and simulate with fake models (no devices)."""

import threading
import time

import numpy as np
import pytest

from macvoice.agent import CollectingOut, Responder, TurnMetrics, VoiceAgent, simulate
from macvoice.endpointing import EndpointConfig, Endpointer
from macvoice.vad import CHUNK
from tests.conftest import tone


class FakeASR:
    name = "fake"

    def __init__(self, text="what time is it"):
        self.text, self.calls = text, []

    def transcribe(self, audio):
        self.calls.append(audio)
        return self.text


class FakeLLM:
    name = "fake"

    def __init__(self, reply="It is noon right now. Anything else?"):
        self.reply_text, self.histories = reply, []

    def reply(self, history):
        self.histories.append([dict(m) for m in history])
        yield from self.reply_text.split(" ")
        # tokens without spaces: sentences() must still split on punctuation


class SpacedLLM(FakeLLM):
    def reply(self, history):
        self.histories.append([dict(m) for m in history])
        for word in self.reply_text.split(" "):
            yield word + " "


class FakeTTS:
    name = "fake"
    sample_rate = 24000

    def __init__(self, chunks=3, block=None):
        self.chunks, self.spoken, self.block = chunks, [], block

    def stream(self, text):
        self.spoken.append(text)
        for _ in range(self.chunks):
            if self.block is not None:
                self.block.wait(2)
            yield np.ones(2400, dtype=np.float32)


def energy_vad(chunk):
    return 0.95 if np.sqrt(np.mean(chunk**2)) > 0.05 else 0.02


def test_responder_runs_asr_llm_tts_and_records_metrics():
    asr, llm, tts, out = FakeASR(), SpacedLLM(), FakeTTS(), CollectingOut()
    r = Responder(asr, llm, tts)
    m = r.respond(np.zeros(16000, dtype=np.float32), out)
    assert m.transcript == "what time is it"
    assert tts.spoken == ["It is noon right now.", "Anything else?"]
    assert m.reply == "It is noon right now. Anything else?"
    assert len(out.chunks) == 6
    assert m.reply_audio_s == pytest.approx(6 * 2400 / 24000)
    assert m.asr_ms >= 0 and m.llm_first_sentence_ms is not None and m.tts_first_audio_ms is not None
    assert r.history == [{"role": "user", "content": "what time is it"},
                         {"role": "assistant", "content": "It is noon right now. Anything else?"}]


def test_responder_keeps_conversation_history():
    llm = SpacedLLM()
    r = Responder(FakeASR(), llm, FakeTTS())
    r.respond(np.zeros(10), CollectingOut())
    r.respond(np.zeros(10), CollectingOut())
    assert [m["role"] for m in llm.histories[1]] == ["user", "assistant", "user"]


@pytest.mark.parametrize("text", ["", ".", " ... "])
def test_responder_ignores_wordless_transcripts(text):
    llm, tts = SpacedLLM(), FakeTTS()
    m = Responder(FakeASR(text), llm, tts).respond(np.zeros(10), CollectingOut())
    assert llm.histories == [] and tts.spoken == []
    assert m.speech_end_to_first_audio_ms is None


def test_responder_stops_when_cancelled():
    cancel = threading.Event()

    class CancellingOut(CollectingOut):
        def write(self, audio):
            super().write(audio)
            cancel.set()

    llm = SpacedLLM()
    r = Responder(FakeASR(), llm, FakeTTS())
    out = CancellingOut()
    m = r.respond(np.zeros(10), out, cancel)
    assert m.cancelled and len(out.chunks) == 1
    assert r.history[-1] == {"role": "assistant", "content": "It is noon right now."}


def test_speech_end_to_first_audio_adds_the_stages():
    m = TurnMetrics(endpoint_ms=300, asr_ms=60, llm_first_sentence_ms=200, tts_first_audio_ms=5)
    assert m.speech_end_to_first_audio_ms == 565


def test_simulate_finds_each_utterance():
    audio = np.concatenate([tone(300, 1.0), np.zeros(32000, dtype=np.float32), tone(300, 0.8)])
    asr = FakeASR()
    results = simulate(audio, energy_vad, Endpointer(None, EndpointConfig(max_silence_ms=1000)),
                       Responder(asr, SpacedLLM(), FakeTTS()))
    assert len(results) == 2
    for event, m in results:
        assert event.reason == "silence"
        assert m.endpoint_ms == pytest.approx(1000, abs=40)
        assert m.speech_end_to_first_audio_ms is not None
    assert len(asr.calls) == 2


def test_simulate_without_responder_reports_endpoints_only():
    results = simulate(tone(300, 1.0), energy_vad, Endpointer(lambda a: 0.99))
    assert len(results) == 1
    event, m = results[0]
    assert event.reason == "detector" and m.transcript == ""
    assert m.timeline["detector_ms"] >= 0


class FakeOut(CollectingOut):
    def __init__(self):
        super().__init__()
        self._busy = False

    @property
    def busy(self):
        return self._busy


def make_agent(barge_in=False, tts=None, out=None, max_silence_ms=300):
    out = out or FakeOut()
    turns = []
    done = threading.Event()

    def on_turn(m):
        turns.append(m)
        done.set()

    agent = VoiceAgent(energy_vad, Endpointer(None, EndpointConfig(max_silence_ms=max_silence_ms)),
                       Responder(FakeASR(), SpacedLLM(), tts or FakeTTS()), out,
                       barge_in=barge_in, echo_guard_s=0.3, on_turn=on_turn)
    return agent, out, turns, done


def speak(agent, seconds, now, loud=True):
    """Feed ``seconds`` of tone or silence with a simulated clock."""
    sig = tone(300, seconds) if loud else np.zeros(int(seconds * 16000), dtype=np.float32)
    for i in range(len(sig) // CHUNK):
        agent.feed(sig[i * CHUNK:(i + 1) * CHUNK], now=now + i * 0.032)
    return now + len(sig) // CHUNK * 0.032


def test_agent_replies_on_worker_thread():
    agent, out, turns, done = make_agent()
    t = speak(agent, 1.0, 0.0)
    speak(agent, 0.5, t, loud=False)
    assert done.wait(2)
    assert turns[0].transcript == "what time is it"
    assert len(out.chunks) == 6
    agent.close()


def test_agent_is_deaf_while_speaking_without_barge_in():
    agent, out, turns, done = make_agent()
    out._busy = True
    speak(agent, 1.0, 0.0)
    speak(agent, 0.5, 1.0, loud=False)
    time.sleep(0.1)
    assert turns == [] and not agent.endpointer.in_turn
    agent.close()


def test_agent_ignores_the_echo_right_after_speaking():
    agent, out, turns, done = make_agent()
    out._busy = True
    agent.feed(np.zeros(CHUNK, dtype=np.float32), now=10.0)
    out._busy = False
    speak(agent, 0.25, 10.05)                 # inside the 0.3 s echo guard
    assert not agent.endpointer.in_turn
    speak(agent, 0.5, 10.5)                   # after it
    assert agent.endpointer.in_turn
    agent.close()


def test_barge_in_stops_the_reply():
    gate = threading.Event()
    agent, out, turns, done = make_agent(barge_in=True, tts=FakeTTS(chunks=50, block=gate))
    t = speak(agent, 1.0, 0.0)
    t = speak(agent, 0.5, t, loud=False)
    deadline = time.time() + 2
    while not agent.speaking and time.time() < deadline:
        time.sleep(0.01)
    assert agent.speaking
    speak(agent, 0.3, t + 1)                  # the user talks over the reply
    gate.set()
    assert done.wait(2)
    assert turns[0].cancelled
    assert out.stopped == 1
    agent.close()


def test_agent_survives_a_failing_reply():
    class Boom(FakeASR):
        def transcribe(self, audio):
            raise RuntimeError("boom")

    out = FakeOut()
    agent = VoiceAgent(energy_vad, Endpointer(None, EndpointConfig(max_silence_ms=300)),
                       Responder(Boom(), SpacedLLM(), FakeTTS()), out)
    t = speak(agent, 1.0, 0.0)
    speak(agent, 0.5, t, loud=False)
    time.sleep(0.2)
    assert not agent.speaking
    agent.close()
