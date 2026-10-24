"""Microphone and speaker through Apple's voice processing (echo cancellation).

Laptop speakers leak into the laptop microphone, so without echo
cancellation the agent hears itself and barge-in only works on headphones.
macOS ships the same echo canceller FaceTime and Chrome use
(``echoCancellation: "all"``); it is a mode of ``AVAudioEngine``'s input
node. Turn it on, play the reply through the *same* engine, and the
microphone signal arrives with that reply subtracted. It also does noise
suppression and automatic gain.

:class:`VoiceProcessingIO` is both the agent's microphone and its
:class:`~macvoice.agent.AudioOut`. It needs PyObjC (``AVFoundation``) and is
macOS only.
"""

from __future__ import annotations

import logging
import threading
from typing import Callable

import numpy as np

from macvoice import SAMPLE_RATE
from macvoice.audio import Chunker, StreamResampler
from macvoice.vad import CHUNK

log = logging.getLogger("macvoice")


class VoiceProcessingIO:
    def __init__(self, out_rate: int, duck_others: bool = False) -> None:
        import AVFoundation as AV

        self._AV = AV
        self.engine = AV.AVAudioEngine.alloc().init()
        # The output side must exist before voice processing is switched on,
        # or the engine fails to start (kAudioUnitErr_FailedInitialization).
        self.engine.outputNode()
        mixer = self.engine.mainMixerNode()
        self.input = self.engine.inputNode()
        ok, err = self.input.setVoiceProcessingEnabled_error_(True, None)
        if not ok:
            raise RuntimeError(f"could not enable voice processing: {err}")
        if not duck_others and hasattr(AV, "AVAudioVoiceProcessingOtherAudioDuckingConfiguration"):
            # By default macOS turns other apps' audio way down while voice
            # processing runs (macOS 14+ lets us ask for the minimum).
            self.input.setVoiceProcessingOtherAudioDuckingConfiguration_(
                AV.AVAudioVoiceProcessingOtherAudioDuckingConfiguration(
                    enableAdvancedDucking=False,
                    duckingLevel=AV.AVAudioVoiceProcessingOtherAudioDuckingLevelMin))

        self.out_rate = out_rate
        self._out_fmt = AV.AVAudioFormat.alloc().initWithCommonFormat_sampleRate_channels_interleaved_(
            AV.AVAudioPCMFormatFloat32, float(out_rate), 1, False)
        self.player = AV.AVAudioPlayerNode.alloc().init()
        self.engine.attachNode_(self.player)
        self.engine.connect_to_format_(self.player, mixer, self._out_fmt)

        self._lock = threading.Lock()
        self._pending = 0       # scheduled buffers not yet consumed
        self._generation = 0    # bumped by stop() so stale completions are ignored
        self._on_chunk: Callable[[np.ndarray], None] | None = None

    # -- microphone ---------------------------------------------------------

    def start(self, on_chunk: Callable[[np.ndarray], None]) -> None:
        """Start the engine; ``on_chunk`` gets 512-sample 16 kHz float32
        chunks of echo-cancelled microphone audio, on an audio thread."""
        fmt = self.input.outputFormatForBus_(0)  # hardware rate; often 9 channels, use the first
        resample = StreamResampler(int(fmt.sampleRate()), SAMPLE_RATE)
        chunk = Chunker(CHUNK)
        self._on_chunk = on_chunk

        def tap(buf, when):
            n = buf.frameLength()
            if not n:
                return
            mic = np.frombuffer(buf.floatChannelData()[0].as_buffer(n), dtype=np.float32)
            for c in chunk(resample(mic)):
                on_chunk(c)

        self.input.installTapOnBus_bufferSize_format_block_(0, 1600, fmt, tap)
        self.engine.prepare()
        ok, err = self.engine.startAndReturnError_(None)
        if not ok:
            raise RuntimeError(f"could not start the audio engine: {err}")
        self.player.play()
        log.info("voice processing on (%s Hz mic)", int(fmt.sampleRate()))

    def close(self) -> None:
        self.input.removeTapOnBus_(0)
        self.player.stop()
        self.engine.stop()

    # -- speaker (AudioOut) ---------------------------------------------------

    def write(self, audio: np.ndarray) -> None:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if not len(audio):
            return
        buf = self._AV.AVAudioPCMBuffer.alloc().initWithPCMFormat_frameCapacity_(self._out_fmt, len(audio))
        np.frombuffer(buf.floatChannelData()[0].as_buffer(len(audio)), dtype=np.float32)[:] = audio
        buf.setFrameLength_(len(audio))
        with self._lock:
            self._pending += 1
            gen = self._generation

        def done():
            with self._lock:
                if gen == self._generation:
                    self._pending -= 1

        self.player.scheduleBuffer_completionHandler_(buf, done)

    def stop(self) -> None:
        """Drop everything queued (barge-in)."""
        with self._lock:
            self._generation += 1
            self._pending = 0
        self.player.stop()
        self.player.play()

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._pending > 0
