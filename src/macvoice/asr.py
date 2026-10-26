"""Speech recognition.

Default: NVIDIA Parakeet TDT 0.6B v2 on MLX (``parakeet-mlx``), with the
encoder quantized to 8 bits. On a 10k-utterance English test set the 8-bit
model's WER was statistically indistinguishable from bf16 while using
~0.9 GB less memory and running ~23% faster.

Baseline (``--asr whisper``, needs the ``baseline`` extra): Whisper
large-v3 on PyTorch-MPS in fp16 with distil-large-v3 as the speculative
decoding assistant, the configuration the article starts from.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Protocol, TypeVar

import numpy as np

from macvoice import SAMPLE_RATE

T = TypeVar("T")
log = logging.getLogger("macvoice")

PARAKEET_MODEL = "mlx-community/parakeet-tdt-0.6b-v2"
PARAKEET_REVISION = "8ae155301e23d820d82aa60d24817c900e69e487"
# The same model with the 8-bit encoder already applied
# (scripts/make_parakeet_q8.py): a 0.7 GB download instead of the 2.3 GB
# fp32 checkpoint, and nothing to convert at load.
PARAKEET_Q8 = "johnrobinsn/parakeet-tdt-0.6b-v2-mlx-q8"
PARAKEET_Q8_REVISION = "aad991e02731e67298700cf1f2b86ddaa45dd18c"
PRECISIONS = ("bf16", "fp16", "q8", "q6", "q4")
GROUP_SIZE = 64


class ASR(Protocol):
    name: str

    def transcribe(self, audio: np.ndarray) -> str: ...


class MLXThread:
    """Runs callables on one long-lived thread.

    MLX streams are thread-local: a model loaded on one thread raises
    ``There is no Stream(gpu, 0) in current thread`` when called from
    another. Pinning load and inference to one thread avoids that no matter
    which thread the caller is on."""

    def __init__(self, name: str, cache_mb: int = 256) -> None:
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"mlx-{name}")
        self.run(lambda: _set_cache_limit(cache_mb))

    def run(self, fn: Callable[[], T]) -> T:
        return self._pool.submit(fn).result()


def _set_cache_limit(mb: int) -> None:
    """Bound MLX's cache of freed Metal buffers. Unbounded, it keeps GBs
    that the process footprint pays for."""
    import mlx.core as mx

    setter = getattr(mx, "set_cache_limit", None) or mx.metal.set_cache_limit
    setter(mb * 2**20)


class Parakeet:
    def __init__(self, model_id: str = PARAKEET_MODEL, precision: str = "q8") -> None:
        if precision not in PRECISIONS:
            raise ValueError(f"precision must be one of {PRECISIONS}, got {precision!r}")
        self.model_id, self.precision = model_id, precision
        self.name = f"parakeet-{precision}"
        self._thread = MLXThread("parakeet")
        self._model = self._thread.run(self._load)

    def _load(self):
        import mlx.core as mx

        if self.model_id == PARAKEET_MODEL and self.precision == "q8":
            try:
                model = load_quantized(*model_files(PARAKEET_Q8, PARAKEET_Q8_REVISION), bits=8)
                mx.eval(model.parameters())
                return model
            except OSError as e:  # offline, or the copy is gone: convert the original
                log.warning("couldn't get %s (%s); converting %s instead", PARAKEET_Q8, e, PARAKEET_MODEL)
        revision = PARAKEET_REVISION if self.model_id == PARAKEET_MODEL else None
        model = convert(*model_files(self.model_id, revision), precision=self.precision)
        mx.eval(model.parameters())
        return model

    def transcribe(self, audio: np.ndarray) -> str:
        def run() -> str:
            import mlx.core as mx
            from parakeet_mlx.audio import get_logmel

            mel = get_logmel(mx.array(np.asarray(audio, dtype=np.float32)), self._model.preprocessor_config)
            results = self._model.generate(mel)
            return results[0].text.strip() if results else ""

        return self._thread.run(run)


def model_files(repo: str, revision: str | None) -> tuple[str, str]:
    """``(config.json, model.safetensors)`` paths, downloaded if needed."""
    from huggingface_hub import hf_hub_download

    return tuple(hf_hub_download(repo, f, revision=revision) for f in ("config.json", "model.safetensors"))


def convert(config: str, weights: str, precision: str = "q8"):
    """The published checkpoint at ``precision``: bf16 is what parakeet-mlx
    loads; fp16 casts (M1 GPUs have no native bf16); qN quantizes the
    encoder (~99% of the weights, group size 64). The decoder and joint
    network stay in bf16."""
    import json

    import mlx.core as mx
    import mlx.nn as nn
    from mlx.utils import tree_flatten, tree_unflatten
    from parakeet_mlx.utils import from_config

    with open(config) as f:
        model = from_config(json.load(f))
    model.load_weights(weights)
    model.update(tree_unflatten([(k, v.astype(mx.bfloat16)) for k, v in tree_flatten(model.parameters())]))
    if precision == "fp16":
        model.set_dtype(mx.float16)
    elif precision.startswith("q"):
        nn.quantize(model.encoder, group_size=GROUP_SIZE, bits=int(precision[1:]))
    return model


def load_quantized(config: str, weights: str, bits: int = 8):
    """A model saved after :func:`convert` quantized its encoder. Quantizing
    the freshly built model first gives it the saved weights' shapes; its
    random initial weights are lazy and never computed."""
    import json

    import mlx.nn as nn
    from parakeet_mlx.utils import from_config

    with open(config) as f:
        model = from_config(json.load(f))
    nn.quantize(model.encoder, group_size=GROUP_SIZE, bits=bits)
    model.load_weights(weights)
    return model


class WhisperHF:
    """Baseline: Whisper large-v3, PyTorch-MPS fp16, speculative decoding."""

    name = "whisper-large-v3"

    def __init__(self, model_id: str = "openai/whisper-large-v3",
                 assistant_id: str | None = "distil-whisper/distil-large-v3") -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoModelForSpeechSeq2Seq, AutoProcessor

        self._torch = torch
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        dtype = torch.float16 if self.device == "mps" else torch.float32
        self._processor = AutoProcessor.from_pretrained(model_id)
        self._model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id, torch_dtype=dtype, attn_implementation="sdpa").to(self.device).eval()
        self._assistant = None
        if assistant_id:
            # Decoder only: the assistant reuses the main model's encoder
            # output. Loading the full seq2seq assistant runs a second
            # encoder per call and makes speculative decoding slower.
            self._assistant = AutoModelForCausalLM.from_pretrained(
                assistant_id, torch_dtype=dtype, attn_implementation="sdpa").to(self.device).eval()
        self._dtype = dtype

    def transcribe(self, audio: np.ndarray) -> str:
        inputs = self._processor(np.asarray(audio, dtype=np.float32), sampling_rate=SAMPLE_RATE,
                                 return_tensors="pt")
        feats = inputs.input_features.to(self.device, dtype=self._dtype)
        kwargs = {"language": "en", "task": "transcribe"}
        if self._assistant is not None:
            kwargs["assistant_model"] = self._assistant
        with self._torch.inference_mode():
            ids = self._model.generate(feats, **kwargs)
        return self._processor.batch_decode(ids, skip_special_tokens=True)[0].strip()


def load(name: str = "parakeet", precision: str = "q8") -> ASR:
    if name == "parakeet":
        return Parakeet(precision=precision)
    if name == "whisper":
        return WhisperHF()
    raise ValueError(f"unknown ASR {name!r} (parakeet | whisper)")
