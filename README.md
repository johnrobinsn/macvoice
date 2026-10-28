# macvoice

A local voice agent loop for Apple Silicon in under 2 GB, with no PyTorch.

The write-up: [A Local Voice Agent on a MacBook in Under 2 GB](https://www.storminthecastle.com/posts/mac_voice/).

```
 mic ──▶ Silero VAD ──▶ Smart Turn ──▶ Parakeet ──▶ your LLM ──▶ Phonon ──▶ speaker
```

On an M5 MacBook Pro, the voice pipeline uses **1.4 GB** (physical footprint, GPU memory included) and starts speaking **~350 ms after you stop talking**, before LLM time. With a local Qwen3-4B as the brain, the first word of the answer arrives in 0.6–0.8 s.

| Stage | Model | Runtime |
|---|---|---|
| Voice activity | [Silero VAD](https://github.com/snakers4/silero-vad) v6 | ONNX Runtime (CPU) |
| End of turn | [Smart Turn](https://github.com/pipecat-ai/smart-turn) v3.2 | ONNX Runtime (CPU) |
| Speech to text | [Parakeet TDT 0.6B v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2), [8-bit](https://huggingface.co/johnrobinsn/parakeet-tdt-0.6b-v2-mlx-q8) | [MLX](https://github.com/senstella/parakeet-mlx) |
| Text to speech | [Phonon 31M](https://huggingface.co/gradium/phonon-31m-en) English | ptts (Rust, Metal) |
| Brain | any OpenAI-compatible server, or a coding agent over [ACP](https://agentclientprotocol.com) | |

## Quick start

Needs an Apple Silicon Mac and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run macvoice download             # the models, 0.85 GB, once

# In another terminal, a local LLM (optional; ~2.6 GB, fully offline):
uvx --from mlx-lm mlx_lm.server --model mlx-community/Qwen3-4B-Instruct-2507-4bit --prompt-cache-size 1

uv run macvoice run                  # talk to it; talk over it to interrupt
```

After the download the voice pipeline runs offline. (`run` downloads anything still missing itself, so `download` is only there to get the wait out of the way first.) While it runs, the bottom line of the terminal shows what it's doing and how loud your mic is:

```
● hearing you  ▕██████████████▉         ▏
```

It goes `listening` → `hearing you` → `thinking` → `speaking`; each finished turn prints above it (`--no-status` turns it off).

At startup it prints which brain it's using. By default that's the LLM server on `localhost:8080` if one is running; otherwise an "echo" brain that repeats what you say, so you can try the audio loop with no LLM at all.

**New to running a local LLM?** [docs/llm-setup.md](docs/llm-setup.md) walks through it step by step (Qwen3-4B on MLX, about 2 minutes plus a 2.1 GB download), along with Ollama, LM Studio, hosted APIs and Claude Code over ACP.

The first time, macOS asks for microphone access for your terminal.

### Echo cancellation (on by default)

Laptop speakers leak into the laptop mic, so a voice agent hears itself. macvoice runs the mic and speaker through one `AVAudioEngine` with Apple's voice processing turned on, the same echo canceller FaceTime uses and Chrome uses for `echoCancellation: "all"` ([`audio_mac.py`](src/macvoice/audio_mac.py), PyObjC; no model, about 70 MB). On an M5's speakers it brings the agent's own voice at the mic from −21 dBFS down to −78 dBFS, below the room's noise, so you can interrupt it by talking. A few things to know:

- It cancels everything the Mac plays, not just the agent: a video playing in the background won't trigger it either.
- Talking over the reply stops it after 250 ms of speech, so a cough doesn't. Speech in the first 0.5 s of a reply is ignored: the first reply of a session leaks a syllable or two of echo while the canceller learns the room.
- It uses the system default mic and speaker, and turns other apps' audio down only slightly while it runs. Choosing a device with `--input-device` or `--output-device` turns it off, and if voice processing won't start on your audio hardware macvoice says so and runs half duplex.
- `--no-aec` turns it off: the loop is then **half duplex**, the mic muted while it speaks. With headphones, `--no-aec --barge-in` lets you interrupt without echo cancellation.

The `ready:` line at startup says which mode you're in.

### Pick a brain

Setup details for each option are in [docs/llm-setup.md](docs/llm-setup.md).

```bash
# Default: the local server on port 8080 (mlx_lm.server above), else echo
uv run macvoice run

# Any other OpenAI-compatible endpoint: Ollama, LM Studio, llama.cpp, or a hosted API
uv run macvoice run --base-url http://localhost:11434/v1 --model llama3.2

# Just the audio loop, even if a server is running
uv run macvoice run --llm echo

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
  you : How many days are in a leap year?
  bot : A leap year has 366 days.
        endpoint 301 ms (detector, p=0.99) | asr 54 ms | llm 404 ms | tts 10 ms | => first audio 769 ms after you stopped
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

The results from [the article](https://www.storminthecastle.com/posts/mac_voice/) (M5, 32 GB, macOS 26.6, on AC power) are in [`results/`](results/).

## Tests

```bash
uv run pytest                 # everything (~20 s on an M5)
uv run pytest -m "not models" # unit tests only, no model downloads
```

Tests marked `devices` play sound and use the real mic and speaker to check echo cancellation, including a full live loop that is interrupted mid-reply; they're opt-in: `MACVOICE_DEVICE_TESTS=1 uv run pytest -m devices` (in a quiet room). macOS cancels any sound the Mac itself plays, so the "person" interrupting is mixed into the echo-cancelled mic signal; `MACVOICE_HUMAN_TESTS=1 ... -s` asks you to do it for real.

Unit tests cover the model list and first-run download, the status line, the endpointer, sentence splitting, the OpenAI-compatible client and brain selection (against a local SSE server), the ACP client (against a scripted fake agent), the live loop's half duplex, echo guard and barge-in rules, the streaming resampler for the voice-processing mic, and the NumPy feature ports (pinned to the PyTorch/transformers reference outputs). Tests marked `models` run the real models: Parakeet accuracy, the published 8-bit Parakeet matching a local conversion (when the original checkpoint is cached), Phonon → Parakeet round trips, the full loop on 32 spoken prompts, the 2 GB ceiling with no PyTorch, and Claude Code over ACP (skipped if `claude-agent-acp` isn't installed).

## Models

Every model is pinned to a commit (Silero to a release tag plus its SHA-256). The published Parakeet checkpoint is fp32 and 2.3 GB; macvoice runs it in bf16 with the encoder quantized to 8 bits, so it downloads [that conversion](https://huggingface.co/johnrobinsn/parakeet-tdt-0.6b-v2-mlx-q8) (0.7 GB) instead. It is made by [`scripts/make_parakeet_q8.py`](scripts/make_parakeet_q8.py) with the same code macvoice falls back to if the copy can't be downloaded, and a `models` test checks the two are identical. Other `--precision` values download the original and convert it at load.

## Licenses

Code: Apache-2.0. Models and test data have their own licenses; see [NOTICE](NOTICE).
