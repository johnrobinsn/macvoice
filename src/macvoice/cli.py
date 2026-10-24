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
    p.add_argument("--llm", default="echo", choices=["echo", "openai", "acp"],
                   help="echo: repeat what you said; openai: any OpenAI-compatible server; "
                        "acp: a coding agent over ACP (default command claude-agent-acp)")
    p.add_argument("--base-url", default=None, help="OpenAI-compatible base URL "
                   "(default http://localhost:8080/v1, mlx_lm.server)")
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


def _build(args, need_tts: bool = True):
    from macvoice import asr, llm, tts
    from macvoice.agent import Responder
    from macvoice.endpointing import EndpointConfig, Endpointer
    from macvoice.turn import SmartTurn
    from macvoice.vad import SileroVAD

    t0 = time.perf_counter()
    vad = SileroVAD()
    detector = None if args.no_turn_detector else SmartTurn()
    max_silence = args.max_silence_ms or (1500 if detector is None else 3000)
    endpointer = Endpointer(detector, EndpointConfig(max_silence_ms=max_silence, threshold=args.threshold))
    responder = Responder(asr.load(args.asr, args.precision),
                          llm.load(args.llm, args.base_url, args.model, args.api_key,
                                   args.acp_command, args.allow_tools),
                          tts.load(args.tts, args.voice) if need_tts else None)
    logging.getLogger("macvoice").info("models loaded in %.1f s", time.perf_counter() - t0)
    return vad, endpointer, responder


def _print_turn(m) -> None:
    total = m.speech_end_to_first_audio_ms
    print(f"\n  you : {m.transcript}")
    if m.reply:
        print(f"  bot : {m.reply}")
    parts = [f"endpoint {m.endpoint_ms:.0f} ms ({m.endpoint_reason}"
             + (f", p={m.p_complete:.2f})" if m.p_complete is not None else ")"),
             f"asr {m.asr_ms:.0f} ms"]
    if m.llm_first_sentence_ms is not None:
        parts.append(f"llm {m.llm_first_sentence_ms:.0f} ms")
    if m.tts_first_audio_ms is not None:
        parts.append(f"tts {m.tts_first_audio_ms:.0f} ms")
    if total is not None:
        parts.append(f"=> first audio {total:.0f} ms after you stopped")
    print("        " + " | ".join(parts), flush=True)


def cmd_run(args) -> None:
    from macvoice.agent import SoundDeviceMic, SpeakerOut, VoiceAgent
    from macvoice.memory import footprint_mb, torch_loaded

    vad, endpointer, responder = _build(args)
    if args.aec:
        # One engine is both mic and speaker, so it knows what to cancel.
        from macvoice.audio_mac import VoiceProcessingIO

        if args.input_device or args.output_device:
            sys.exit("--aec uses the system default microphone and speaker")
        out = mic = VoiceProcessingIO(responder.tts.sample_rate)
        barge_in, echo_guard_s = True, 0.0
        mode = "  (echo cancellation on: talk over me any time)"
    else:
        out = SpeakerOut(responder.tts.sample_rate, device=args.output_device)
        mic = SoundDeviceMic(args.input_device)
        barge_in, echo_guard_s = args.barge_in, 0.3
        mode = "" if barge_in else "  (half duplex: the mic is muted while I talk; --aec to talk over me)"
    fp = footprint_mb()
    print(f"ready: footprint {fp:.0f} MB, torch loaded: {torch_loaded()}" + mode)
    agent = VoiceAgent(vad, endpointer, responder, out, barge_in=barge_in,
                       echo_guard_s=echo_guard_s, on_turn=_print_turn)
    agent.run(mic, max_turns=args.turns)


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


def cmd_transcribe(args) -> None:
    from macvoice import asr
    from macvoice.audio import read_wav

    model = asr.load(args.asr, args.precision)
    for path in args.wav:
        audio, _ = read_wav(path, SAMPLE_RATE)
        t0 = time.perf_counter()
        text = model.transcribe(audio)
        print(f"{path}\t{(time.perf_counter() - t0) * 1000:.0f} ms\t{text}")


def cmd_say(args) -> None:
    from macvoice import tts
    from macvoice.audio import write_wav

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


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="macvoice", description=__doc__)
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("run", help="talk to it: microphone in, speaker out")
    _add_model_args(p)
    p.add_argument("--aec", action="store_true",
                   help="Apple voice processing (echo cancellation): interrupt replies by talking, "
                        "even on laptop speakers")
    p.add_argument("--barge-in", action="store_true",
                   help="interrupt replies by talking, without echo cancellation (use headphones)")
    p.add_argument("--input-device", default=None)
    p.add_argument("--output-device", default=None)
    p.add_argument("--turns", type=int, default=None, help="exit after this many replies")
    p.set_defaults(fn=cmd_run)

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

    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(asctime)s %(name)s %(message)s", stream=sys.stderr)
    args.fn(args)


if __name__ == "__main__":
    main()
