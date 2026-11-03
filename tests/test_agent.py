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


@pytest.mark.parametrize("tail", ["😊", "**", "—", ":)"])
def test_responder_skips_fragments_with_nothing_to_say(tail):
    """Qwen sometimes ends a reply with an emoji; the splitter hands it over
    alone, and Phonon raises on it, which used to kill the reply."""
    class StrictTTS(FakeTTS):
        def stream(self, text):
            if not any(ch.isalnum() for ch in text):
                raise ValueError("nothing to synthesize: the text is empty")
            return super().stream(text)

    tts = StrictTTS()
    llm = SpacedLLM(f"That is a great idea! {tail}")
    m = Responder(FakeASR(), llm, tts).respond(np.zeros(10), CollectingOut())
    assert tts.spoken == ["That is a great idea!"]
    assert m.reply == "That is a great idea!"


def test_a_tts_failure_skips_that_sentence_not_the_reply(caplog):
    class FlakyTTS(FakeTTS):
        def stream(self, text):
            if "noon" in text:
                raise RuntimeError("synth exploded")
            return super().stream(text)

    tts, out = FlakyTTS(), CollectingOut()
    m = Responder(FakeASR(), SpacedLLM(), tts).respond(np.zeros(10), out)
    assert tts.spoken == ["Anything else?"]
    assert len(out.chunks) == 3
    assert m.reply == "Anything else?"            # only what was actually said
    assert "TTS failed on 'It is noon right now.' (synth exploded)" in caplog.text


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


def test_barge_in_stops_the_speech_not_the_brain():
    """Talking over a reply stops it speaking; the brain isn't cancelled,
    and the whole reply is kept as text."""
    gate = threading.Event()
    agent, out, turns, done = make_agent(barge_in=True, tts=FakeTTS(chunks=50, block=gate))
    cancels = []
    agent.responder.llm.cancel = lambda: cancels.append(1)
    t = speak(agent, 1.0, 0.0)
    t = speak(agent, 0.5, t, loud=False)
    assert wait_for(lambda: agent.speaking)
    speak(agent, 0.9, t + 1)                  # the user talks over the reply (past the 0.5 s grace)
    gate.set()
    assert done.wait(2)
    m = turns[0]
    assert m.interrupted and not m.cancelled and cancels == []
    assert out.stopped == 1
    assert m.reply == "It is noon right now. Anything else?"        # all of it, as text
    assert agent.responder.tts.spoken == ["It is noon right now."]  # the second sentence wasn't said
    agent.close()


def test_what_you_say_while_it_works_is_queued_as_the_next_turn():
    gate = threading.Event()

    class SlowLLM(SpacedLLM):
        def reply(self, history):
            gate.wait(2)                      # busy in a tool
            yield from super().reply(history)

    out, turns, notes = FakeOut(), [], []
    asr = FakeASR()
    agent = VoiceAgent(energy_vad, Endpointer(None, EndpointConfig(max_silence_ms=300)),
                       Responder(asr, SlowLLM(), FakeTTS()), out, barge_in=True, echo_guard_s=0.0,
                       on_turn=turns.append, on_note=notes.append)
    t = speak(agent, 1.0, 0.0)
    t = speak(agent, 0.5, t, loud=False)
    assert wait_for(lambda: agent.busy)
    asr.text = "and add a footer"
    t = speak(agent, 0.9, t + 1)
    speak(agent, 0.5, t, loud=False)          # a whole turn, heard mid-reply
    assert wait_for(lambda: any('queued: "and add a footer"' in n for n in notes))
    assert turns == []                        # the first turn is still working: nothing cancelled
    gate.set()
    assert wait_for(lambda: len(turns) == 2)
    assert turns[0].interrupted and not turns[0].cancelled
    assert turns[1].transcript == "and add a footer" and not turns[1].interrupted
    assert len(asr.calls) == 2                # transcribed once, on arrival
    agent.close()


def test_talking_while_it_thinks_keeps_the_reply_silent():
    gate = threading.Event()

    class SlowLLM(SpacedLLM):
        def reply(self, history):
            gate.wait(2)                      # a coding agent working, nothing said yet
            yield from super().reply(history)

    out, turns, done = FakeOut(), [], threading.Event()
    tts = FakeTTS()
    agent = VoiceAgent(energy_vad, Endpointer(None, EndpointConfig(max_silence_ms=300)),
                       Responder(FakeASR(), SlowLLM(), tts), out, barge_in=True, echo_guard_s=0.0,
                       on_turn=lambda m: (turns.append(m), done.set()))
    t = speak(agent, 1.0, 0.0)
    t = speak(agent, 0.5, t, loud=False)
    assert wait_for(lambda: agent.state == "thinking")
    speak(agent, 0.9, t + 1)                  # talking while it works
    assert agent.state in ("hearing", "working")
    gate.set()
    assert done.wait(2)
    assert turns[0].interrupted and tts.spoken == [] and out.chunks == []
    assert turns[0].reply == "It is noon right now. Anything else?"
    agent.close()


def test_saying_stop_cancels_the_brain():
    gate = threading.Event()

    class SilentBrain(SpacedLLM):
        cancelled = 0

        def reply(self, history):
            gate.wait(2)
            yield "Too late."

        def cancel(self):
            SilentBrain.cancelled += 1
            gate.set()

    out, turns, done, notes = FakeOut(), [], threading.Event(), []
    asr = FakeASR("go")
    agent = VoiceAgent(energy_vad, Endpointer(None, EndpointConfig(max_silence_ms=300)),
                       Responder(asr, SilentBrain(), FakeTTS()), out, barge_in=True, echo_guard_s=0.0,
                       on_turn=lambda m: (turns.append(m), done.set()), on_note=notes.append)
    t = speak(agent, 1.0, 0.0)
    t = speak(agent, 0.5, t, loud=False)
    assert wait_for(lambda: agent.busy)
    asr.text = "Stop."
    t = speak(agent, 0.9, t + 1)
    speak(agent, 0.5, t, loud=False)
    assert done.wait(2)
    assert turns[0].cancelled and SilentBrain.cancelled == 1
    assert "stopping the agent" in notes[-1]
    time.sleep(0.1)
    assert len(turns) == 1                    # "stop" isn't sent on as a prompt
    agent.close()


def test_the_next_turn_is_not_hushed():
    """A reply you talked over doesn't carry over: the queued turn after it
    speaks, and gets its own grace period."""
    gate = threading.Event()

    class SlowFirst(SpacedLLM):
        def reply(self, history):
            if len(self.histories) == 0:
                gate.wait(2)
            yield from super().reply(history)

    out, turns, tts = FakeOut(), [], FakeTTS()
    asr = FakeASR()
    agent = VoiceAgent(energy_vad, Endpointer(None, EndpointConfig(max_silence_ms=300)),
                       Responder(asr, SlowFirst(), tts), out, barge_in=True, echo_guard_s=0.0,
                       on_turn=turns.append)
    t = speak(agent, 1.0, 0.0)
    t = speak(agent, 0.5, t, loud=False)
    assert wait_for(lambda: agent.busy)
    asr.text = "and add a footer"
    t = speak(agent, 0.9, t + 1)              # hushes the first reply, and is queued
    speak(agent, 0.5, t, loud=False)
    assert wait_for(lambda: agent._hearing == 0)
    gate.set()
    assert wait_for(lambda: len(turns) == 2)
    assert turns[0].interrupted and not turns[1].interrupted
    assert tts.spoken == ["It is noon right now.", "Anything else?"]   # all of the second reply
    agent.close()


def test_a_late_stop_leaves_the_next_reply_alone():
    agent, out, turns, done = make_agent(barge_in=True)
    notes = []
    agent.on_note = notes.append
    stops = []
    agent._stop_reply = lambda: stops.append(1)
    agent._replying.set()
    agent._turn_no = 7                        # the reply "stop" is said over
    agent.responder.asr.text = "stop"
    t = speak(agent, 0.6, 0.0, loud=False)
    t = speak(agent, 0.5, t)
    agent._turn_no = 8                        # ... has finished; the next one is running
    speak(agent, 0.5, t, loud=False)
    assert wait_for(lambda: any("already finished" in n for n in notes))
    assert stops == [] and turns == []
    agent._replying.clear()
    agent.close()


@pytest.mark.parametrize("text, stop", [
    ("Stop.", True), ("stop it", True), ("Cancel that!", True), ("stop, stop", True),
    ("stop using tabs", False), ("don't stop", False), ("bus stop", False), ("", False),
])
def test_is_stop(text, stop):
    from macvoice.agent import is_stop
    assert is_stop(text) is stop


def test_done_cue_after_a_reply_you_talked_over():
    from macvoice.tone import done_tone

    gate = threading.Event()
    cue = done_tone(24000)
    out, turns, done = FakeOut(), [], threading.Event()

    class SlowLLM(SpacedLLM):
        def reply(self, history):
            gate.wait(2)
            yield from super().reply(history)

    agent = VoiceAgent(energy_vad, Endpointer(None, EndpointConfig(max_silence_ms=300)),
                       Responder(FakeASR(), SlowLLM(), FakeTTS()), out, barge_in=True, echo_guard_s=0.0,
                       on_turn=lambda m: (turns.append(m), done.set()), done_tone=cue)
    t = speak(agent, 1.0, 0.0)
    t = speak(agent, 0.5, t, loud=False)
    assert wait_for(lambda: agent.busy)
    t = speak(agent, 0.6, t + 1, loud=False)  # past the reply's first 0.5 s
    speak(agent, 0.4, t)                      # hushes it; no silence after, so not a turn
    gate.set()
    assert done.wait(2)
    assert wait_for(lambda: not agent.busy)
    assert len(out.chunks) == 1 and len(out.chunks[0]) == len(cue)
    agent.close()


def test_a_question_is_still_spoken_after_barge_in():
    from macvoice.acp import ALLOW

    class AskingLLM(SpacedLLM):
        def reply(self, history):
            gate.wait(2)
            decision = agent.ask("The agent wants to run ls. Allow?")
            yield f"You said {'yes' if decision == ALLOW else 'no'}."

    gate, done = threading.Event(), threading.Event()
    out, asr, tts = FakeOut(), FakeASR("list the files"), FakeTTS()
    agent = VoiceAgent(energy_vad, Endpointer(None, EndpointConfig(max_silence_ms=300)),
                       Responder(asr, AskingLLM(), tts), out, barge_in=True, echo_guard_s=0.0,
                       on_turn=lambda m: done.set())
    t = speak(agent, 1.0, 0.0)
    t = speak(agent, 0.5, t, loud=False)
    assert wait_for(lambda: agent.busy)
    t = speak(agent, 0.6, t + 1, loud=False)
    speak(agent, 0.4, t)                      # hush it
    gate.set()
    assert wait_for(lambda: agent.state == "asking")
    assert tts.spoken == ["The agent wants to run ls. Allow?"]
    asr.text = "yes"
    t = speak(agent, 0.5, 20.0)
    speak(agent, 0.5, t, loud=False)
    assert done.wait(2)
    assert tts.spoken == ["The agent wants to run ls. Allow?"]   # the reply itself stays silent
    agent.close()


def test_short_sounds_do_not_stop_the_reply():
    agent, out, turns, done = make_agent(barge_in=True, max_silence_ms=3000)   # no turns end
    out._busy = True
    speak(agent, 0.45, 0.0)                   # the reply's first 0.5 s: echo leaking, ignored
    speak(agent, 0.3, 0.45, loud=False)
    speak(agent, 0.15, 0.75)                  # a cough
    speak(agent, 0.3, 0.9, loud=False)
    assert out.stopped == 0 and agent.interruptions == 0
    speak(agent, 0.3, 1.2)                    # someone actually talking
    assert out.stopped == 1 and agent.interruptions == 1
    agent.close()


def test_talking_on_over_a_stopped_reply_is_one_interruption():
    agent, out, turns, done = make_agent(barge_in=True)
    out._busy = True                          # the stopped reply's audio winding down
    speak(agent, 2.5, 0.0)
    assert out.stopped == 1 and agent.interruptions == 1
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


def wait_for(cond, timeout=2.0):
    deadline = time.time() + timeout
    while not cond() and time.time() < deadline:
        time.sleep(0.01)
    return cond()


def test_agent_state_follows_the_turn():
    """listening -> hearing (in the turn) -> thinking (reply under way, no
    audio yet) -> speaking, including the gaps between sentences ->
    listening."""
    gate = threading.Event()
    agent, out, turns, done = make_agent(tts=FakeTTS(block=gate))
    silence = np.zeros(CHUNK, dtype=np.float32)
    assert agent.state == "listening"
    t = speak(agent, 0.5, 0.0)
    assert agent.state == "hearing"
    speak(agent, 0.5, t, loud=False)
    assert wait_for(lambda: agent.state == "thinking")
    out._busy = True                          # the first sentence playing
    agent.feed(silence, now=5.0)
    assert agent.state == "speaking"
    out._busy = False                         # between sentences
    agent.feed(silence, now=5.1)
    assert agent.state == "speaking"
    gate.set()
    assert done.wait(2)
    assert wait_for(lambda: agent.state == "listening")
    agent.close()


def test_agent_reports_the_mic_level():
    agent, *_ = make_agent()
    agent.feed(tone(300, 0.032, amp=0.5)[:CHUNK], now=0.0)
    assert agent.level_db == pytest.approx(-9.0, abs=0.5)   # a 0.5 sine's RMS is -9 dBFS
    agent.feed(np.zeros(CHUNK, dtype=np.float32), now=0.1)
    assert agent.level_db == pytest.approx(-100, abs=1)
    agent.close()


# ── questions (permission requests) ──────────────────────────────────

def asking_agent(asr_text="yes please", keys=None, timeout_s=2.0):
    out = FakeOut()
    notes = []
    agent = VoiceAgent(energy_vad, Endpointer(None, EndpointConfig(max_silence_ms=300)),
                       Responder(FakeASR(asr_text), SpacedLLM(), FakeTTS()), out,
                       echo_guard_s=0.3, on_note=notes.append, keys=keys, answer_timeout_s=timeout_s)
    return agent, out, notes


def ask_in_background(agent, *args):
    result = {}
    t = threading.Thread(target=lambda: result.setdefault("answer", agent.ask(*args)))
    t.start()
    assert wait_for(lambda: agent.state == "asking")
    return t, result


def test_ask_speaks_the_question_and_takes_a_spoken_answer():
    agent, out, notes = asking_agent("Yes, go ahead.")
    t, result = ask_in_background(agent, "Claude wants to run ls. Allow?", "? Claude wants to use ls")
    assert agent.responder.tts.spoken == ["Claude wants to run ls. Allow?"]
    speak(agent, 0.5, 10.0)
    speak(agent, 0.5, 10.5, loud=False)
    t.join(2)
    assert result["answer"] == "allow"
    assert notes[0].startswith("? Claude wants to use ls") and "you : Yes, go ahead." in notes[1]
    assert agent.state == "listening"
    agent.close()


def test_ask_ignores_what_it_hears_while_the_question_plays():
    agent, out, notes = asking_agent("yes")
    out._busy = True                                       # the question is still playing
    t, result = ask_in_background(agent, "Allow?")
    speak(agent, 0.5, 10.0)                                # its own voice, leaking back
    speak(agent, 0.5, 10.5, loud=False)
    assert "answer" not in result
    out._busy = False
    speak(agent, 0.5, 20.0)
    speak(agent, 0.5, 20.5, loud=False)
    t.join(2)
    assert result["answer"] == "allow"
    agent.close()


class FakeKeys:
    def __init__(self):
        self.on_key, self.stopped = None, 0

    def start(self, on_key):
        self.on_key = on_key

    def stop(self):
        self.stopped += 1


@pytest.mark.parametrize("key, answer", [("y", "allow"), ("n", "reject"), ("a", "always")])
def test_ask_takes_a_key(key, answer):
    keys = FakeKeys()
    agent, out, notes = asking_agent(keys=keys)
    t, result = ask_in_background(agent, "Allow?")
    keys.on_key("x")                                       # not an answer: ignored
    keys.on_key(key)
    t.join(2)
    assert result["answer"] == answer and keys.stopped >= 1
    agent.close()


def test_an_unclear_answer_is_asked_again_then_declined():
    agent, out, notes = asking_agent("banana", timeout_s=0.5)
    t, result = ask_in_background(agent, "Allow?")
    speak(agent, 0.5, 10.0)
    speak(agent, 0.5, 10.5, loud=False)
    t.join(3)
    assert result["answer"] is None
    assert agent.responder.tts.spoken == ["Allow?", "Sorry, was that a yes or a no?"]
    assert "unclear" in " ".join(notes) and "no answer: declined" in notes[-1]
    agent.close()


def test_a_permission_request_mid_reply_is_answered_by_voice():
    """End to end: the brain asks permission while replying; the user's
    spoken answer goes back, then the reply carries on."""
    from macvoice.acp import ALLOW

    class AskingLLM(SpacedLLM):
        def reply(self, history):
            self.histories.append(history)
            decision = agent.ask("The agent wants to run ls. Allow?")
            yield f"You said {'yes' if decision == ALLOW else 'no'}. Done now."

    out = FakeOut()
    done = threading.Event()
    asr = FakeASR("list the files")
    agent = VoiceAgent(energy_vad, Endpointer(None, EndpointConfig(max_silence_ms=300)),
                       Responder(asr, AskingLLM(), FakeTTS()), out, echo_guard_s=0.3,
                       on_turn=lambda m: done.set())
    t = speak(agent, 0.5, 0.0)
    speak(agent, 0.5, t, loud=False)                       # the request
    assert wait_for(lambda: agent.state == "asking")
    asr.text = "yes"
    t = speak(agent, 0.5, 10.0)
    speak(agent, 0.5, t, loud=False)                       # the answer
    assert done.wait(2)
    assert agent.responder.tts.spoken[-2:] == ["You said yes.", "Done now."]
    agent.close()


# ── thinking tone ─────────────────────────────────────────────────────

def test_thinking_tone_plays_while_waiting_and_stops_for_speech():
    from macvoice.tone import thinking_tone

    gate = threading.Event()

    class SlowLLM(SpacedLLM):
        def reply(self, history):
            gate.wait(2)
            yield from super().reply(history)

    out, done = FakeOut(), threading.Event()
    tone_audio = thinking_tone(24000, period_s=0.2)
    agent = VoiceAgent(energy_vad, Endpointer(None, EndpointConfig(max_silence_ms=300)),
                       Responder(FakeASR(), SlowLLM(), FakeTTS()), out, echo_guard_s=0.3,
                       on_turn=lambda m: done.set(), thinking_tone=tone_audio, thinking_tone_after_s=0.05)
    t = speak(agent, 0.5, 0.0)
    speak(agent, 0.5, t, loud=False)
    assert wait_for(lambda: len(out.chunks) >= 2)          # tone periods, nothing said yet
    assert all(len(c) == len(tone_audio) for c in out.chunks)
    assert agent.state == "thinking"
    stops = out.stopped
    gate.set()
    assert done.wait(2)
    assert out.stopped == stops + 1                        # speech cut the tone off
    n = len(out.chunks)
    time.sleep(0.2)
    assert len(out.chunks) == n                            # and it doesn't come back
    agent.close()


def test_no_tone_for_quick_replies():
    from macvoice.tone import thinking_tone

    out, done = FakeOut(), threading.Event()
    agent = VoiceAgent(energy_vad, Endpointer(None, EndpointConfig(max_silence_ms=300)),
                       Responder(FakeASR(), SpacedLLM(), FakeTTS()), out, echo_guard_s=0.3,
                       on_turn=lambda m: done.set(), thinking_tone=thinking_tone(24000),
                       thinking_tone_after_s=5.0)
    t = speak(agent, 0.5, 0.0)
    speak(agent, 0.5, t, loud=False)
    assert done.wait(2)
    assert all(len(c) == 2400 for c in out.chunks)         # speech only
    agent.close()


def test_thinking_tone_is_quiet_and_click_free():
    from macvoice.tone import thinking_tone

    t = thinking_tone(24000)
    assert len(t) == 24000 and t.dtype == np.float32
    assert 20 * np.log10(np.abs(t).max()) == pytest.approx(-32, abs=0.5)
    assert abs(t[0]) < 1e-4 and abs(t[int(0.12 * 24000) - 1]) < 1e-3   # fades in and out


# ── console: typed turns, interrupt, mute, markdown ──────────────────

def test_typed_prompt_is_answered_like_a_spoken_one():
    out, done, turns = FakeOut(), threading.Event(), []
    asr, tts = FakeASR(), FakeTTS()
    agent = VoiceAgent(energy_vad, Endpointer(None, EndpointConfig(max_silence_ms=300)),
                       Responder(asr, SpacedLLM(), tts), out,
                       on_turn=lambda m: (turns.append(m), done.set()))
    agent.submit_text("what time is it")
    assert done.wait(2)
    m = turns[0]
    assert m.typed and m.transcript == "what time is it" and asr.calls == []
    assert tts.spoken == ["It is noon right now.", "Anything else?"]
    agent.close()


def test_interrupt_stops_a_reply_and_cancels_the_brain():
    gate = threading.Event()

    class SilentBrain(SpacedLLM):
        cancelled = 0

        def reply(self, history):
            gate.wait(2)                               # a coding agent busy in a tool
            yield "Too late."

        def cancel(self):
            SilentBrain.cancelled += 1
            gate.set()

    out, done, turns = FakeOut(), threading.Event(), []
    agent = VoiceAgent(energy_vad, Endpointer(None, EndpointConfig(max_silence_ms=300)),
                       Responder(FakeASR(), SilentBrain(), FakeTTS()), out,
                       on_turn=lambda m: (turns.append(m), done.set()))
    assert agent.interrupt() is False                  # nothing to stop
    agent.submit_text("go")
    assert wait_for(lambda: agent.busy)
    assert agent.interrupt() is True
    assert done.wait(2) and turns[0].cancelled and SilentBrain.cancelled == 1
    assert out.stopped >= 1
    agent.close()


def test_interrupt_declines_an_open_question():
    agent, out, notes = asking_agent()
    t, result = ask_in_background(agent, "Allow?")
    assert agent.interrupt() is True
    t.join(2)
    assert result["answer"] == "reject"
    agent.close()


def test_muted_replies_are_text_only():
    out, done, turns = FakeOut(), threading.Event(), []
    tts = FakeTTS()
    r = Responder(FakeASR(), SpacedLLM(), tts)
    r.speak = False
    m = r.respond_text("hi", out)
    assert tts.spoken == [] and out.chunks == []
    assert m.reply == "It is noon right now. Anything else?"


def test_markdown_is_shown_but_not_read_aloud():
    class MarkdownLLM(SpacedLLM):
        def reply(self, history):
            yield from ["I fixed **two** bugs in `app.py`.\n\n```python\n",
                        "print('secret code')\n```\n\n- Run the [tests](http://x).\n"]

    tts = FakeTTS()
    m = Responder(FakeASR(), MarkdownLLM(), tts).respond_text("fix it", CollectingOut())
    assert tts.spoken == ["I fixed two bugs in app.py.", "Run the tests."]
    assert "```python" in m.text and "**two**" in m.text      # the screen gets the original


def test_a_failing_chunk_does_not_stop_the_listening(caplog):
    import queue as q
    agent = VoiceAgent.__new__(VoiceAgent)
    calls = []

    def feed(chunk):
        calls.append(chunk)
        raise TypeError("boom")
    agent.feed = feed
    chunks = q.Queue()
    chunks.put("a")
    chunks.put("b")
    agent._feed_from(chunks)
    agent._feed_from(chunks)
    assert calls == ["a", "b"] and "still listening" in caplog.text
