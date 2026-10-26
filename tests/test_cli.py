"""How ``macvoice run`` picks its audio (no devices)."""

import argparse

import pytest

from macvoice import cli


class FakeVPIO:
    def __init__(self, rate, fail_start=False):
        self.rate, self.fail_start, self.closed = rate, fail_start, False
        self.on_chunk = None

    def start(self, on_chunk):
        if self.fail_start:
            raise RuntimeError("could not start the audio engine")
        self.on_chunk = on_chunk

    def close(self):
        self.closed = True


@pytest.fixture
def devices(monkeypatch):
    made = []

    class Mic:
        def __init__(self, device=None):
            self.device = device
            made.append(("mic", device))

    class Speaker:
        def __init__(self, rate, device=None):
            made.append(("speaker", device))

    monkeypatch.setattr("macvoice.agent.SoundDeviceMic", Mic)
    monkeypatch.setattr("macvoice.agent.SpeakerOut", Speaker)
    return made


def use_vpio(monkeypatch, **kw):
    made = []

    def factory(rate):
        made.append(FakeVPIO(rate, **kw))
        return made[-1]

    monkeypatch.setattr("macvoice.audio_mac.VoiceProcessingIO", factory)
    return made


def run_args(*argv):
    return cli.build_parser().parse_args(["run", *argv])


def test_echo_cancellation_is_the_default(monkeypatch, devices):
    vpio = use_vpio(monkeypatch)
    args = run_args()
    assert args.aec is True
    mic, out, barge_in, guard, mode = cli.open_audio(args, 24000, note=lambda m: None)
    assert out is vpio[0] and barge_in and guard == 0.0
    assert "echo cancellation on" in mode and devices == []
    # chunks reach whoever the agent hands to start()
    got = []
    mic.start(got.append)
    vpio[0].on_chunk("chunk")
    assert got == ["chunk"]
    mic.close()
    assert vpio[0].closed


def test_no_aec_is_half_duplex(monkeypatch, devices):
    vpio = use_vpio(monkeypatch)
    mic, out, barge_in, guard, mode = cli.open_audio(run_args("--no-aec"), 24000)
    assert vpio == [] and not barge_in and guard == 0.3
    assert "half duplex" in mode
    assert devices == [("speaker", None), ("mic", None)]


def test_no_aec_with_barge_in_is_the_headphones_mode(monkeypatch, devices):
    use_vpio(monkeypatch)
    _, _, barge_in, _, mode = cli.open_audio(run_args("--no-aec", "--barge-in"), 24000)
    assert barge_in and "headphones" in mode


def test_choosing_a_device_turns_echo_cancellation_off(monkeypatch, devices):
    vpio = use_vpio(monkeypatch)
    notes = []
    _, _, barge_in, _, _ = cli.open_audio(run_args("--input-device", "USB Mic"), 24000, note=notes.append)
    assert vpio == [] and not barge_in
    assert ("mic", "USB Mic") in devices
    assert notes == ["echo cancellation uses the system default mic and speaker, so it's off"]


def test_falls_back_to_half_duplex_when_echo_cancellation_will_not_start(monkeypatch, devices):
    use_vpio(monkeypatch, fail_start=True)
    notes = []
    _, _, barge_in, _, mode = cli.open_audio(run_args(), 24000, note=notes.append)
    assert not barge_in and "half duplex" in mode
    assert "echo cancellation didn't start (could not start the audio engine)" in notes[0]
