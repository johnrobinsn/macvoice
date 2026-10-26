"""``macvoice`` command line."""

from __future__ import annotations

import argparse
import logging
import sys
import time

import numpy as np

from macvoice import SAMPLE_RATE


def _add_model_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--asr", default="parakeet", choices=["parakeet", "whisper"])
    p.add_argument("--precision", default="q8", help="Parakeet precision: bf16 fp16 q8 q6 q4")
    p.add_argument("--tts", default="phonon", choices=["phonon", "kokoro"])
    p.add_argument("--voice", default=None, help="TTS voice (Phonon catalog name, e.g. Harper)")
    p.add_argument("--llm", default=None, choices=["server", "echo", "acp", "openai"],
                   help="server: an OpenAI-compatible endpoint (--base-url); echo: repeat what you "
                        "said; acp: a coding agent over ACP (default command claude-agent-acp). "
                        "Default: the local server on port 8080 if one is running, else echo")
    p.add_argument("--base-url", default=None, help="OpenAI-compatible base URL: a local server "
                   "(mlx_lm.server, Ollama, LM Studio, llama.cpp) or a hosted API "
                   "(default http://localhost:8080/v1)")
    p.add_argument("--model", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--acp-command", default=None,
                   help='ACP agent command line, e.g. "gemini --experimental-acp" (default claude-agent-acp)')
    p.add_argument("--allow-tools", action="store_true",
                   help="let the ACP agent run tools it asks permission for (declined by default)")
    p.add_argument("--no-turn-detector", action="store_true",
                   help="end turns on silence alone (--max-silence-ms)")
    p.add_argument("--max-silence-ms", type=float, default=None,
                   help="end a turn after this much silence regardless "
                        "(default 3000 with the turn detector, 1500 without)")
    p.add_argument("--threshold", type=float, default=0.5, help="Smart Turn P(complete) to end a turn")


def _assets(args, need_tts: bool = True):
    from macvoice import models

    return models.assets(args.asr, args.precision, args.tts,
                         turn_detector=not getattr(args, "no_turn_detector", False), need_tts=need_tts)


def _build(args, need_tts: bool = True):
    from macvoice import asr, llm, models, tts
    from macvoice.agent import Responder
    from macvoice.endpointing import EndpointConfig, Endpointer
    from macvoice.turn import SmartTurn
    from macvoice.vad import SileroVAD

    models.ensure(_assets(args, need_tts))
    t0 = time.perf_counter()
    vad = SileroVAD()
    detector = None if args.no_turn_detector else SmartTurn()
    max_silence = args.max_silence_ms or (1500 if detector is None else 3000)
    endpointer = Endpointer(detector, EndpointConfig(max_silence_ms=max_silence, threshold=args.threshold))
    try:
        brain, about = llm.load(args.llm, args.base_url, args.model, args.api_key,
                                args.acp_command, args.allow_tools)
    except RuntimeError as e:
        sys.exit(f"macvoice: {e}")
    print(f"brain: {about}", flush=True)
    responder = Responder(asr.load(args.asr, args.precision), brain,
                          tts.load(args.tts, args.voice) if need_tts else None)
    logging.getLogger("macvoice").info("models loaded in %.1f s", time.perf_counter() - t0)
    return vad, endpointer, responder


def _print_turn(m, emit=print) -> None:
    total = m.speech_end_to_first_audio_ms
    lines = ["", f"  you : {m.transcript}"]
    if m.reply:
        lines.append(f"  bot : {m.reply}")
    parts = [f"endpoint {m.endpoint_ms:.0f} ms ({m.endpoint_reason}"
             + (f", p={m.p_complete:.2f})" if m.p_complete is not None else ")"),
             f"asr {m.asr_ms:.0f} ms"]
    if m.llm_first_sentence_ms is not None:
        parts.append(f"llm {m.llm_first_sentence_ms:.0f} ms")
    if m.tts_first_audio_ms is not None:
        parts.append(f"tts {m.tts_first_audio_ms:.0f} ms")
    if total is not None:
        parts.append(f"=> first audio {total:.0f} ms after you stopped")
    lines.append("        " + " | ".join(parts))
    emit("\n".join(lines))


class _StartedMic:
    """A mic that's already running; ``start`` just says where chunks go.
    Starting it early is how :func:`open_audio` finds out it works."""

    def __init__(self, io) -> None:
        self.io, self._target = io, None
        io.start(lambda c: self._target and self._target(c))

    def start(self, on_chunk) -> None:
        self._target = on_chunk

    def close(self) -> None:
        self.io.close()


def open_audio(args, rate: int, note=print):
    """``(mic, out, barge_in, echo_guard_s, mode)``. Echo cancellation
    (Apple voice processing) unless ``--no-aec``; it needs the system
    default devices, and if it won't start the loop runs half duplex."""
    from macvoice.agent import SoundDeviceMic, SpeakerOut

    aec = args.aec
    if aec and (args.input_device or args.output_device):
        note("echo cancellation uses the system default mic and speaker, so it's off")
        aec = False
    if aec:
        try:
            # One engine is both mic and speaker, so it knows what to cancel.
            from macvoice.audio_mac import VoiceProcessingIO

            io = VoiceProcessingIO(rate)
            return _StartedMic(io), io, True, 0.0, "echo cancellation on: talk over me any time"
        except Exception as e:  # noqa: BLE001 - any device or OS refusal: fall back
            note(f"echo cancellation didn't start ({e}); running half duplex")
    out = SpeakerOut(rate, device=args.output_device)
    mic = SoundDeviceMic(args.input_device)
    if args.barge_in:
        return mic, out, True, 0.3, "no echo cancellation: use headphones to talk over me"
    return mic, out, False, 0.3, "half duplex: the mic is muted while I talk"


def cmd_run(args) -> None:
    from macvoice.agent import VoiceAgent
    from macvoice.memory import footprint_mb, torch_loaded

    vad, endpointer, responder = _build(args)
    mic, out, barge_in, echo_guard_s, mode = open_audio(args, responder.tts.sample_rate)
    from macvoice.status import StatusLine

    fp = footprint_mb()
    print(f"ready: footprint {fp:.0f} MB, torch loaded: {torch_loaded()}  ({mode})")
    print("Talk to me. Ctrl-C to quit.", flush=True)
    agent = VoiceAgent(vad, endpointer, responder, out, barge_in=barge_in, echo_guard_s=echo_guard_s)
    # -v logs to the same terminal, which would tear the live line.
    status = StatusLine(lambda: (agent.state, agent.level_db), barge_in=barge_in,
                        enabled=False if args.no_status or args.verbose else None)
    agent.on_turn = lambda m: _print_turn(m, status.print)
    status.start()
    try:
        agent.run(mic, max_turns=args.turns)
    finally:
        status.stop()


def cmd_simulate(args) -> None:
    from macvoice.agent import CollectingOut, simulate
    from macvoice.audio import read_wav, write_wav

    audio, _ = read_wav(args.wav, SAMPLE_RATE)
    vad, endpointer, responder = _build(args)
    out = CollectingOut()
    for _, m in simulate(audio, vad, endpointer, responder, out):
        _print_turn(m)
    if args.out:
        write_wav(args.out, out.audio, responder.tts.sample_rate)
        print(f"reply audio -> {args.out}")


def cmd_download(args) -> None:
    from macvoice import models

    models.download(_assets(args))


def cmd_transcribe(args) -> None:
    from macvoice import asr, models
    from macvoice.audio import read_wav

    if args.asr == "parakeet":
        models.ensure(models.parakeet(args.precision))
    model = asr.load(args.asr, args.precision)
    for path in args.wav:
        audio, _ = read_wav(path, SAMPLE_RATE)
        t0 = time.perf_counter()
        text = model.transcribe(audio)
        print(f"{path}\t{(time.perf_counter() - t0) * 1000:.0f} ms\t{text}")


def cmd_say(args) -> None:
    from macvoice import models, tts
    from macvoice.audio import write_wav

    if args.tts == "phonon":
        models.ensure(models.phonon())
    engine = tts.load(args.tts, args.voice)
    t0 = time.perf_counter()
    first = None
    chunks = []
    for chunk in engine.stream(args.text):
        first = first if first is not None else time.perf_counter() - t0
        chunks.append(chunk)
    total = time.perf_counter() - t0
    audio = np.concatenate(chunks)
    secs = len(audio) / engine.sample_rate
    print(f"first audio {first * 1000:.0f} ms, {secs:.1f} s of audio in {total * 1000:.0f} ms "
          f"(RTF {total / secs:.3f})")
    if args.out:
        write_wav(args.out, audio, engine.sample_rate)
    else:
        import sounddevice as sd
        sd.play(audio, engine.sample_rate, blocking=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="macvoice", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="talk to it: microphone in, speaker out")
    _add_model_args(p)
    p.add_argument("--aec", action=argparse.BooleanOptionalAction, default=True,
                   help="Apple voice processing (echo cancellation), on by default: interrupt replies "
                        "by talking, even on laptop speakers. --no-aec: half duplex, the mic is muted "
                        "while it talks")
    p.add_argument("--barge-in", action="store_true",
                   help="with --no-aec: interrupt replies by talking anyway (use headphones)")
    p.add_argument("--input-device", default=None)
    p.add_argument("--output-device", default=None)
    p.add_argument("--turns", type=int, default=None, help="exit after this many replies")
    p.add_argument("--no-status", action="store_true", help="no live status line, just the turn summaries")
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("download", help="download the models now (about 0.85 GB, once), "
                                        "so run starts listening right away")
    p.add_argument("--asr", default="parakeet", choices=["parakeet", "whisper"])
    p.add_argument("--precision", default="q8", help="Parakeet precision: bf16 fp16 q8 q6 q4")
    p.add_argument("--tts", default="phonon", choices=["phonon", "kokoro"])
    p.set_defaults(fn=cmd_download)

    p = sub.add_parser("simulate", help="run a WAV recording through the loop, no devices")
    _add_model_args(p)
    p.add_argument("wav")
    p.add_argument("-o", "--out", help="write the reply audio to this WAV")
    p.set_defaults(fn=cmd_simulate)

    p = sub.add_parser("transcribe", help="transcribe WAV files")
    p.add_argument("--asr", default="parakeet", choices=["parakeet", "whisper"])
    p.add_argument("--precision", default="q8")
    p.add_argument("wav", nargs="+")
    p.set_defaults(fn=cmd_transcribe)

    p = sub.add_parser("say", help="speak text (or write it to a WAV)")
    p.add_argument("--tts", default="phonon", choices=["phonon", "kokoro"])
    p.add_argument("--voice", default=None)
    p.add_argument("-o", "--out")
    p.add_argument("text")
    p.set_defaults(fn=cmd_say)

    p = sub.add_parser("bench", help="benchmarks (see macvoice.bench)")
    p.add_argument("rest", nargs=argparse.REMAINDER)
    p.set_defaults(fn=lambda a: __import__("macvoice.bench", fromlist=["main"]).main(a.rest))

    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(name)s %(message)s", stream=sys.stderr)
    args.fn(args)


if __name__ == "__main__":
    main()
