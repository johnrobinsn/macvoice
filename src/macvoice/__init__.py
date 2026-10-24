"""macvoice: a local voice agent loop for Apple Silicon.

    mic -> Silero VAD (ONNX) -> Smart Turn v3.2 (ONNX) -> Parakeet TDT (MLX)
        -> your LLM -> Phonon TTS (Rust/Metal) -> speaker

No PyTorch is imported on the default path; see ``macvoice.memory.torch_loaded``.
"""

__version__ = "0.1.0"

SAMPLE_RATE = 16000  # VAD, turn detection and ASR all run at 16 kHz
