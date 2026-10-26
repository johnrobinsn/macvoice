# Setting up a brain

macvoice handles listening and speaking; the replies come from an LLM you run (or point it at). This page gets you from nothing to a talking assistant in a few minutes.

macvoice looks for an LLM server on `http://localhost:8080/v1` when it starts and prints what it found:

```
brain: mlx-community/Qwen3-4B-Instruct-2507-4bit at http://localhost:8080/v1
```

If nothing is running there, it falls back to an "echo" brain that repeats what you say, and tells you so. That's handy for checking your mic and speakers, but you'll want a real model.

## Recommended: Qwen3-4B on MLX (local, offline)

This is the setup the numbers in the README come from: fast enough that the answer starts about 0.6–0.8 s after you stop talking, and small enough that the whole assistant fits in about 4 GB.

**1. Start the model server** in its own terminal and leave it running:

```bash
uvx --from mlx-lm mlx_lm.server \
  --model mlx-community/Qwen3-4B-Instruct-2507-4bit \
  --prompt-cache-size 1
```

- `uvx` (part of [uv](https://docs.astral.sh/uv/), which you already have for macvoice) runs Apple's [`mlx-lm`](https://github.com/ml-explore/mlx-lm) without installing anything globally.
- The first start downloads the model (about 2.1 GB) into `~/.cache/huggingface`. After that it starts in a few seconds and works offline.
- It listens on port 8080, which is where macvoice looks by default.
- Keep `--prompt-cache-size 1`. Without it the server caches every conversation's prompt and grew from 2.6 GB to over 4 GB in a few turns in my testing.

**2. Check it's answering** (optional):

```bash
curl -s http://localhost:8080/v1/models
```

You should see the model name in the reply.

**3. Run macvoice** in another terminal:

```bash
uv run macvoice run
```

The `brain:` line at startup should name the Qwen model. Ask it something.

**Why this model?** It's an *instruct* model, so it answers directly instead of "thinking out loud" first (a reasoning model's hidden thinking adds seconds of silence before it speaks). It's Apache-2.0 licensed, and at 4 bits it's quick on any Apple Silicon Mac. macvoice tells it to keep answers to one to three short sentences, since they're spoken.

**Using a different MLX model:** swap the `--model` id for another one from [mlx-community](https://huggingface.co/mlx-community). Prefer instruct (non-thinking) models; bigger ones answer better but start talking later and need more memory.

## Other local servers

Anything that speaks the OpenAI chat-completions API works ("OpenAI-compatible" is the protocol; nothing is sent to OpenAI). Point macvoice at it with `--base-url`, and name the model with `--model` if the server hosts more than one:

| Server | Command |
|---|---|
| [Ollama](https://ollama.com) | `uv run macvoice run --base-url http://localhost:11434/v1 --model <model>` |
| [LM Studio](https://lmstudio.ai) (start its local server) | `uv run macvoice run --base-url http://localhost:1234/v1` |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) `llama-server` | `uv run macvoice run --base-url http://localhost:8080/v1` |

When you pass `--base-url` and nothing answers there, macvoice stops with an error instead of falling back to echo.

## A hosted API

The same flag works for hosted services that offer an OpenAI-compatible endpoint:

```bash
uv run macvoice run --base-url https://api.example.com/v1 --model <model> --api-key "$YOUR_API_KEY"
```

Expect more latency than a local model: the network round trip comes before the first word.

## Claude Code (or another coding agent) over ACP

macvoice can also talk to a coding agent through the [Agent Client Protocol](https://agentclientprotocol.com). Replies take longer (about 3 s to the first word in my testing), but the agent can actually do things.

**1. Install the Claude Code ACP adapter** (needs Node.js):

```bash
npm install -g @agentclientprotocol/claude-agent-acp
```

**2. Make sure Claude Code is signed in.** The adapter uses your existing Claude Code login (run `claude` once and sign in if you haven't), or an `ANTHROPIC_API_KEY` in the environment.

**3. Run macvoice with it:**

```bash
uv run macvoice run --llm acp
```

The agent runs in `~/.cache/macvoice/acp` and keeps its own conversation history. When it asks permission to run a tool, macvoice declines unless you pass `--allow-tools`; think twice before letting a voice command edit files.

Other ACP agents work the same way: `--acp-command "gemini --experimental-acp"`, for example.

## Troubleshooting

- **It just repeats what I say.** That's the echo fallback: no server answered on `localhost:8080` when macvoice started. Start the model server first (step 1), then macvoice.
- **`no LLM server at ...`** You passed `--base-url` (or `--llm server`) and nothing answered there. Check the server is running and the port matches.
- **The first answer is slow, later ones are fast.** Normal: the first request loads the model into GPU memory and warms up.
- **Long pauses before it speaks.** Check you're not running a reasoning ("thinking") model, which generates hidden text before the answer.
- **Just want to test the audio?** `uv run macvoice run --llm echo` skips the LLM even if a server is running.
