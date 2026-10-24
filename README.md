# macvoice

A local voice agent loop for Apple Silicon in under 2 GB, with no PyTorch.

```
 mic ──▶ Silero VAD ──▶ Smart Turn ──▶ Parakeet ──▶ your LLM ──▶ Phonon ──▶ speaker
```

On an M5 MacBook Pro, the voice pipeline uses **1.4 GB** (physical footprint, GPU memory included) and starts speaking **~350 ms after you stop talking**, before LLM time. With a local Qwen3-4B as the brain, the first word of the answer arrives in 0.6–0.8 s. The draft write-up is in [ARTICLE.md](ARTICLE.md).

| Stage | Model | Runtime |
|---|---|---|
| Voice activity | [Silero VAD](https://github.com/snakers4/silero-vad) v6 | ONNX Runtime (CPU) |
| End of turn | [Smart Turn](https://github.com/pipecat-ai/smart-turn) v3.2 | ONNX Runtime (CPU) |
| Speech to text | [Parakeet TDT 0.6B v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2), 8-bit | [MLX](https://github.com/senstella/parakeet-mlx) |
| Text to speech | [Phonon 31M](https://huggingface.co/gradium/phonon-31m-en) English | ptts (Rust, Metal) |
| Brain | any OpenAI-compatible server, or a coding agent over [ACP](https://agentclientprotocol.com) | |

## Quick start

Needs an Apple Silicon Mac, [uv](https://docs.astral.sh/uv/), and ~2.7 GB for the first model download.

```bash
uv sync
uv run macvoice run                  # talk to it; the "echo" brain repeats you back
uv run macvoice run --aec            # with echo cancellation: talk over it to interrupt
```

The first time, macOS asks for microphone access for your terminal. Laptop speakers leak into the laptop mic, so without `--aec` the loop is **half duplex**: the mic is muted while it speaks (with headphones, `--barge-in` lets you interrupt without echo cancellation).

### Echo cancellation (`--aec`)

`--aec` runs the mic and speaker through one `AVAudioEngine` with Apple's voice processing turned on, the same echo canceller FaceTime uses and Chrome uses for `echoCancellation: "all"` ([`audio_mac.py`](src/macvoice/audio_mac.py), PyObjC; no model, about 70 MB). On an M5's speakers it brings the agent's own voice at the mic from −21 dBFS down to −78 dBFS, below the room's noise, so you can interrupt it by talking. A few things to know:

- It cancels everything the Mac plays, not just the agent: a video playing in the background won't trigger it either.
- Talking over the reply stops it after 250 ms of speech, so a cough doesn't. Speech in the first 0.5 s of a reply is ignored: the first reply of a session leaks a syllable or two of echo while the canceller learns the room.
- It uses the system default mic and speaker, and turns other apps' audio down only slightly while it runs.

### Pick a brain

```bash
# A local model on MLX (fully offline; ~2.6 GB in its own process)
uvx --from mlx-lm mlx_lm.server --model mlx-community/Qwen3-4B-Instruct-2507-4bit --prompt-cache-size 1
uv run macvoice run --llm openai

# Any other OpenAI-compatible endpoint (Ollama, LM Studio, llama.cpp, hosted APIs)
uv run macvoice run --llm openai --base-url http://localhost:11434/v1 --model llama3.2

# Claude Code over ACP (npm i -g @agentclientprotocol/claude-agent-acp)
uv run macvoice run --llm acp

# Any other ACP agent
uv run macvoice run --llm acp --acp-command "gemini --experimental-acp"
```

ACP agents keep their own conversation. Tool-permission requests are declined unless you pass `--allow-tools`.

### Other commands

```bash
uv run macvoice simulate recording.wav -o reply.wav   # run a recording through the loop, no devices
uv run macvoice transcribe a.wav b.wav                 # Parakeet only
uv run macvoice say "Hello there" [-o hello.wav]       # Phonon only
uv run macvoice run --voice Harper                     # a Phonon catalog voice
uv run macvoice run --turns 3                          # exit after three replies
```

Each turn prints where the time went:

```
  you : Who painted the Mona Lisa?
  bot : Leonardo da Vinci painted the Mona Lisa.
        endpoint 301 ms (detector, p=0.99) | asr 64 ms | llm 314 ms | tts 10 ms | => first audio 689 ms after you stopped
```

## How turns end

After 300 ms of silence, the turn's audio, trimmed to where speech stopped, goes to Smart Turn. If it says the speaker is done (P ≥ 0.5) the turn ends immediately; otherwise the loop keeps listening for up to 3 s, and speech that resumes continues the same turn. `--no-turn-detector` falls back to a plain 1.5 s silence timer. The logic is a pure class, [`endpointing.py`](src/macvoice/endpointing.py), driven chunk by chunk and unit-tested with synthetic VAD probabilities.

## Benchmarks

```bash
uv run macvoice bench stack --stack new --out results/new.json
uv run macvoice bench turns --out results/turns.json

# The PyTorch baseline (Whisper large-v3 + Kokoro), in its own environment
UV_PROJECT_ENVIRONMENT=.venv-baseline uv sync --extra baseline
.venv-baseline/bin/macvoice bench stack --stack baseline --out results/baseline.json
uv run macvoice bench compare results/baseline.json results/new.json
```

The results from the article (M5, 32 GB, macOS 26.6, on AC power) are in [`results/`](results/).

## Tests

```bash
uv run pytest                 # everything (~20 s on an M5)
uv run pytest -m "not models" # unit tests only, no model downloads
```

Tests marked `devices` play sound and use the real mic and speaker to check echo cancellation, including a full live loop that is interrupted mid-reply; they're opt-in: `MACVOICE_DEVICE_TESTS=1 uv run pytest -m devices` (in a quiet room). macOS cancels any sound the Mac itself plays, so the "person" interrupting is mixed into the echo-cancelled mic signal; `MACVOICE_HUMAN_TESTS=1 ... -s` asks you to do it for real.

Unit tests cover the endpointer, sentence splitting, the OpenAI-compatible client (against a local SSE server), the ACP client (against a scripted fake agent), the live loop's half duplex, echo guard and barge-in, and the NumPy feature ports (pinned to the PyTorch/transformers reference outputs). Tests marked `models` run the real models: Parakeet accuracy, Phonon → Parakeet round trips, the full loop on 32 spoken prompts, the 2 GB ceiling with no PyTorch, and Claude Code over ACP (skipped if `claude-agent-acp` isn't installed).

## Licenses

Code: Apache-2.0. Models and test data have their own licenses; see [NOTICE](NOTICE).
