"""Smart Turn v3.2 end-of-turn detector (pipecat-ai, BSD-2-Clause).

An audio model (Whisper-tiny encoder + a linear head, 8 MB int8 ONNX,
~10 ms on CPU) that judges from prosody whether the speaker is done. It
needs no transcript, so it can run before ASR.

One thing matters more than anything else when using it: give it the
audio **up to where speech stops**, not the trailing silence. Silence at
the end pushes it toward "complete" (in vibr8's tests the AUC fell from
0.93 with no trailing silence to 0.62 with one second), which defeats the
point of asking. :func:`trim_trailing_silence` does that cut.
"""

from __future__ import annotations

import numpy as np

from macvoice import SAMPLE_RATE
from macvoice.features import whisper_log_mel

HF_REPO = "pipecat-ai/smart-turn-v3"
ONNX_FILENAME = "smart-turn-v3.2-cpu.onnx"
WINDOW_S = 8


class SmartTurn:
    """``detector(audio) -> P(turn complete)``. ``session`` is injectable
    for tests."""

    def __init__(self, session=None) -> None:
        if session is None:
            import onnxruntime as ort
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(repo_id=HF_REPO, filename=ONNX_FILENAME)
            opts = ort.SessionOptions()
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
        self._session = session

    def __call__(self, audio: np.ndarray) -> float:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if len(audio) == 0:
            return 0.0
        n = WINDOW_S * SAMPLE_RATE
        # The reference keeps the last 8 s, zero-padded at the front.
        audio = audio[-n:] if len(audio) > n else np.pad(audio, (n - len(audio), 0))
        feats = whisper_log_mel(audio, n)
        out = self._session.run(None, {"input_features": feats})[0]
        return float(np.asarray(out).reshape(-1)[0])


def trim_trailing_silence(audio: np.ndarray, frame: int = 320, below_peak_db: float = 35.0,
                          keep_s: float = 0.05) -> np.ndarray:
    """Cut 16 kHz audio just after the last 20 ms frame within
    ``below_peak_db`` of the loudest frame, keeping ``keep_s`` of tail. The
    threshold is relative to the peak, so mic gain doesn't matter."""
    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if len(audio) < frame:
        return audio
    n = len(audio) // frame
    rms_db = 20 * np.log10(np.sqrt(np.mean(audio[: n * frame].reshape(n, frame) ** 2, axis=1)) + 1e-9)
    voiced = np.nonzero(rms_db > rms_db.max() - below_peak_db)[0]
    if len(voiced) == 0:
        return audio
    end = (voiced[-1] + 1) * frame + int(keep_s * SAMPLE_RATE)
    return audio[: min(end, len(audio))]
