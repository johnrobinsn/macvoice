"""Silero VAD (v6) on ONNX Runtime (no PyTorch).

The official ``silero-vad`` package wraps its ONNX model in a small
PyTorch class (``OnnxWrapper``) that keeps the recurrent state and a
64-sample context between calls. This is the same logic in NumPy; its
probabilities match the PyTorch JIT model to ~1e-5
(``tests/test_vad.py``).
"""

from __future__ import annotations

import hashlib
import threading
import urllib.request
from pathlib import Path

import numpy as np

CHUNK = 512  # samples at 16 kHz = 32 ms
_CONTEXT = 64
VERSION = "v6.2.3"
URL = f"https://github.com/snakers4/silero-vad/raw/{VERSION}/src/silero_vad/data/silero_vad.onnx"
SHA256 = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3"
SIZE = 2327524


def cache_path() -> Path:
    return Path.home() / ".cache/macvoice/silero_vad.onnx"


def model_path() -> Path:
    """silero_vad.onnx (MIT), downloaded once (~2 MB) and checked against
    its SHA-256, since it comes from a git tag rather than a pinned commit."""
    dest = cache_path()
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(URL, tmp)
        digest = hashlib.sha256(tmp.read_bytes()).hexdigest()
        if digest != SHA256:
            tmp.unlink()
            raise RuntimeError(f"{URL} has SHA-256 {digest}, expected {SHA256}")
        tmp.replace(dest)
    return dest


class SileroVAD:
    """``vad(chunk) -> P(speech)`` for consecutive 512-sample 16 kHz chunks.

    ``session`` is injectable for tests; by default the ONNX model runs on
    one CPU thread (~0.1 ms per chunk on an M-series core)."""

    def __init__(self, session=None) -> None:
        if session is None:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 1
            opts.intra_op_num_threads = 1
            session = ort.InferenceSession(str(model_path()), sess_options=opts,
                                           providers=["CPUExecutionProvider"])
        self._session = session
        self._lock = threading.Lock()
        self.reset()

    def reset(self) -> None:
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, _CONTEXT), dtype=np.float32)

    def __call__(self, chunk: np.ndarray) -> float:
        x = np.asarray(chunk, dtype=np.float32).reshape(1, -1)
        if x.shape[1] != CHUNK:
            raise ValueError(f"SileroVAD expects {CHUNK}-sample chunks at 16 kHz, got {x.shape[1]}")
        with self._lock:
            inp = np.concatenate([self._context, x], axis=1)
            out, state = self._session.run(None, {"input": inp, "state": self._state,
                                                  "sr": np.array(16000, dtype=np.int64)})
            self._state = state
            self._context = inp[:, -_CONTEXT:]
        return float(np.asarray(out).reshape(-1)[0])
