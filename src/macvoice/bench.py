"""Benchmarks behind the article. Each run is a fresh process so memory
numbers aren't polluted by an earlier stack.

    macvoice bench stack --stack new      --out results/new.json
    macvoice bench stack --stack baseline --out results/baseline.json   # needs the baseline extra
    macvoice bench compare results/baseline.json results/new.json
    macvoice bench turns --out results/turns.json

``stack``: load each component, then measure ASR (latency, WER) on the
bundled LibriSpeech clips, TTS (first audio, real-time factor), and the
whole loop on the synthesized conversational prompts in
tests/fixtures/prompts (end of speech -> first reply audio, echo LLM).

``turns``: endpointing policies on synthesized hesitations (see
:func:`turns`).
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from macvoice import SAMPLE_RATE

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "librispeech"

TTS_SENTENCES = [
    "The capital of France is Paris.",
    "Sure, I can help you with that.",
    "It looks like it will rain tomorrow afternoon, so bring an umbrella.",
    "An octopus has three hearts and blue blood.",
    "Your meeting with the design team starts in fifteen minutes.",
    "I couldn't find that file, but there's one with a similar name in your downloads folder.",
    "A leap year has three hundred sixty six days.",
    "Okay, the timer is set for ten minutes.",
    "Here's a quick summary: the tests pass, and the build is ready to ship.",
    "Good morning! You have two new messages.",
]


# ── WER ───────────────────────────────────────────────────────────────

_NUMBERS = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five",
            "6": "six", "7": "seven", "8": "eight", "9": "nine", "10": "ten"}


def normalize(text: str) -> list[str]:
    """Lowercase words without punctuation; hyphens split words, apostrophes
    stay, single digits become words. Enough for LibriSpeech, which has no
    numerals or abbreviations beyond these."""
    text = text.lower().replace("-", " ")
    text = re.sub(r"[^a-z0-9' ]+", " ", text)
    words = [w.strip("'") for w in text.split()]
    return [_NUMBERS.get(w, w) for w in words if w]


def edit_distance(ref: list[str], hyp: list[str]) -> int:
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def wer(refs: list[str], hyps: list[str]) -> float:
    errors = sum(edit_distance(normalize(r), normalize(h)) for r, h in zip(refs, hyps))
    words = sum(len(normalize(r)) for r in refs)
    return errors / max(words, 1)


# ── helpers ───────────────────────────────────────────────────────────


def pct(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


def load_fixtures() -> list[dict]:
    from macvoice.audio import read_wav

    manifest = json.loads((FIXTURES / "manifest.json").read_text())
    clips = []
    for c in manifest["clips"]:
        audio, _ = read_wav(FIXTURES / c["wav"], SAMPLE_RATE)
        clips.append({**c, "audio": audio})
    return clips


def _footprint() -> float | None:
    from macvoice.memory import footprint_mb

    fp = footprint_mb()
    return None if fp is None else round(fp, 1)


def _timed(fn):
    t0 = time.perf_counter()
    value = fn()
    return value, (time.perf_counter() - t0) * 1000


# ── stack ─────────────────────────────────────────────────────────────


def stack(name: str, repeats: int = 3) -> dict:
    from macvoice import asr, llm, tts
    from macvoice.agent import CollectingOut, Responder, simulate
    from macvoice.endpointing import EndpointConfig, Endpointer
    from macvoice.memory import peak_footprint_mb, torch_loaded
    from macvoice.turn import SmartTurn
    from macvoice.vad import CHUNK, SileroVAD

    new = name == "new"
    r: dict = {"stack": name, "machine": platform.machine(), "macos": platform.mac_ver()[0],
               "python": platform.python_version(), "footprint_mb": {"start": _footprint()},
               "load_ms": {}}

    vad, r["load_ms"]["vad"] = _timed(SileroVAD)
    detector = None
    if new:
        detector, r["load_ms"]["turn"] = _timed(SmartTurn)
    r["footprint_mb"]["vad+turn"] = _footprint()
    model, r["load_ms"]["asr"] = _timed(lambda: asr.load("parakeet" if new else "whisper"))
    r["asr"] = model.name
    r["footprint_mb"]["+asr"] = _footprint()
    voice, r["load_ms"]["tts"] = _timed(lambda: tts.load("phonon" if new else "kokoro"))
    r["tts"] = voice.name
    r["footprint_mb"]["+tts"] = _footprint()

    clips = load_fixtures()

    # Component latencies
    chunk = clips[0]["audio"][:CHUNK]
    vad_ms = [_timed(lambda: vad(chunk))[1] for _ in range(200)]
    r["vad_ms_per_chunk_p50"] = pct(vad_ms, 50)
    if detector is not None:
        turn_ms = [_timed(lambda c=c: detector(c["audio"]))[1] for c in clips]
        r["turn_ms_p50"] = pct(turn_ms, 50)

    model.transcribe(clips[0]["audio"])  # warm up
    lat, hyps = [], []
    for rep in range(repeats):
        for c in clips:
            text, ms = _timed(lambda c=c: model.transcribe(c["audio"]))
            lat.append(ms)
            if rep == 0:
                hyps.append(text)
    r["asr_ms_p50"], r["asr_ms_p95"] = pct(lat, 50), pct(lat, 95)
    r["asr_wer"] = wer([c["transcript"] for c in clips], hyps)
    r["asr_audio_s"] = sum(len(c["audio"]) for c in clips) / SAMPLE_RATE
    r["asr_rtfx"] = r["asr_audio_s"] * repeats / (sum(lat) / 1000)
    r["footprint_mb"]["after_asr"] = _footprint()

    list(voice.stream(TTS_SENTENCES[0]))  # warm up
    first, rtf = [], []
    for s in TTS_SENTENCES:
        t0 = time.perf_counter()
        ttfa, n = None, 0
        for ch in voice.stream(s):
            ttfa = ttfa if ttfa is not None else (time.perf_counter() - t0) * 1000
            n += len(ch)
        first.append(ttfa)
        rtf.append((time.perf_counter() - t0) / (n / voice.sample_rate))
    r["tts_first_audio_ms_p50"], r["tts_first_audio_ms_p95"] = pct(first, 50), pct(first, 95)
    r["tts_rtf_p50"] = pct(rtf, 50)
    r["footprint_mb"]["after_tts"] = _footprint()

    # The whole loop on the conversational prompts (echo LLM: no model time,
    # so this is the cost of the voice pipeline itself). LibriSpeech is read
    # audiobook text cut out of paragraphs, which an end-of-turn model
    # rightly often hears as unfinished, so it is used for ASR only.
    config = EndpointConfig() if new else EndpointConfig(max_silence_ms=1500)
    totals, endpoint, reasons = [], [], {}
    prompts = [a for (_, _, kind), a in sorted(load_prompts().items()) if kind == "complete"]
    for audio in prompts:
        responder = Responder(model, llm.Echo(), voice)
        vad.reset()
        results = simulate(audio, vad, Endpointer(detector, config), responder, CollectingOut())
        for _, m in results:
            if m.speech_end_to_first_audio_ms is not None:
                totals.append(m.speech_end_to_first_audio_ms)
                endpoint.append(m.endpoint_ms)
                reasons[m.endpoint_reason] = reasons.get(m.endpoint_reason, 0) + 1
    r["loop_policy"] = ("smart-turn after 300 ms, max 3000 ms" if new
                        else "fixed 1500 ms silence")
    r["loop_turns"] = len(totals)
    r["loop_endpoint_ms_p50"] = pct(endpoint, 50)
    r["loop_first_audio_ms_p50"], r["loop_first_audio_ms_p95"] = pct(totals, 50), pct(totals, 95)
    r["loop_end_reasons"] = reasons
    r["footprint_mb"]["end"] = _footprint()
    r["peak_footprint_mb"] = peak_footprint_mb()
    r["torch_loaded"] = torch_loaded()
    return r


ROWS = [
    ("ASR model", "asr", "{}"),
    ("TTS model", "tts", "{}"),
    ("PyTorch loaded", "torch_loaded", "{}"),
    ("ASR latency p50 / p95 (ms)", ("asr_ms_p50", "asr_ms_p95"), "{:.0f} / {:.0f}"),
    ("ASR WER (20 LibriSpeech clips)", "asr_wer", "{:.1%}"),
    ("ASR speed (x real time)", "asr_rtfx", "{:.0f}x"),
    ("TTS first audio p50 / p95 (ms)", ("tts_first_audio_ms_p50", "tts_first_audio_ms_p95"), "{:.0f} / {:.0f}"),
    ("TTS real-time factor", "tts_rtf_p50", "{:.3f}"),
    ("Turn detector (ms)", "turn_ms_p50", "{:.0f}"),
    ("Endpointing policy", "loop_policy", "{}"),
    ("Speech end -> turn over p50 (ms)", "loop_endpoint_ms_p50", "{:.0f}"),
    ("Speech end -> first audio p50 / p95 (ms)", ("loop_first_audio_ms_p50", "loop_first_audio_ms_p95"),
     "{:.0f} / {:.0f}"),
    ("Footprint, models loaded (MB)", ("footprint_mb", "+tts"), "{:.0f}"),
    ("Footprint, after the run (MB)", ("footprint_mb", "end"), "{:.0f}"),
    ("Peak footprint (MB)", "peak_footprint_mb", "{:.0f}"),
]


def _cell(r: dict, key, fmt: str) -> str:
    try:
        if isinstance(key, tuple) and key[0] == "footprint_mb":
            return fmt.format(r["footprint_mb"][key[1]])
        if isinstance(key, tuple):
            return fmt.format(*(r[k] for k in key))
        return fmt.format(r[key])
    except (KeyError, TypeError):
        return "-"


def compare(paths: list[str]) -> str:
    runs = [json.loads(Path(p).read_text()) for p in paths]
    header = "| | " + " | ".join(r["stack"] for r in runs) + " |"
    lines = [header, "|---" * (len(runs) + 1) + "|"]
    for label, key, fmt in ROWS:
        lines.append(f"| {label} | " + " | ".join(_cell(r, key, fmt) for r in runs) + " |")
    return "\n".join(lines)


# ── turns ─────────────────────────────────────────────────────────────

# (complete request, (start, continuation) of a turn interrupted by a pause)
TURN_ITEMS = [
    ("What's the weather going to be like tomorrow?",
     ("I'd like to book a table for, um,", "four people at seven.")),
    ("Set a timer for ten minutes.",
     ("Can you remind me to call", "my sister on Sunday?")),
    ("How many days are in a leap year?",
     ("The thing I keep forgetting is, uh,", "where I parked the car.")),
    ("Turn off the lights in the kitchen.",
     ("Send a message to Alex saying that", "I'll be ten minutes late.")),
    ("What time is it in Tokyo right now?",
     ("I need you to find the file with the, um,", "quarterly budget numbers.")),
    ("Play some relaxing music.",
     ("What's the best way to get to", "the airport from here?")),
    ("Who wrote Pride and Prejudice?",
     ("So my question is, hmm,", "how long should I boil an egg?")),
    ("Add milk to my shopping list.",
     ("Could you explain why the sky is blue and", "why sunsets are red?")),
]
TURN_VOICES = [None, "Harper", "Hadley", "Juniper"]  # Phonon: checkpoint default + catalog
TURN_VOICES_NAMES = [v.lower() if v else "default" for v in TURN_VOICES]


PROMPTS = FIXTURES.parent / "prompts"


def load_prompts() -> dict[tuple[str, int, str], np.ndarray]:
    """The synthesized prompts (scripts/make_prompt_fixtures.py), keyed by
    (voice, item, kind) with kind in complete | head | tail."""
    from macvoice.audio import read_wav

    manifest = json.loads((PROMPTS / "manifest.json").read_text())
    return {(c["voice"], c["item"], c["kind"]): read_wav(PROMPTS / c["wav"], SAMPLE_RATE)[0]
            for c in manifest["clips"]}


FILLERS = re.compile(r"\b(um|uh|hmm|er)\b", re.I)


def turn_samples(pauses_s: tuple[float, ...]) -> list[tuple[str, float | None, np.ndarray]]:
    """("complete", None, audio) per prompt, and ("filler" | "bare", p,
    start + p s of silence + continuation) per prompt and pause: "filler" when
    the start ends on um/uh/hmm, "bare" when it just stops mid-clause."""
    prompts = load_prompts()
    samples = []
    for (voice, item, kind), audio in sorted(prompts.items()):
        if kind == "complete":
            samples.append(("complete", None, audio))
        elif kind == "head":
            tail = prompts[(voice, item, "tail")]
            pause_kind = "filler" if FILLERS.search(TURN_ITEMS[item][1][0]) else "bare"
            for p in pauses_s:
                gap = np.zeros(int(p * SAMPLE_RATE), dtype=np.float32)
                samples.append((pause_kind, p, np.concatenate([audio, gap, tail])))
    return samples


def turns(pauses_s: tuple[float, ...] = (1.0, 2.0)) -> dict:
    """Endpointing policies on the synthesized prompts.

    Complete requests measure how soon a policy ends a finished turn (and
    whether it wrongly splits it). Requests interrupted by a 1 s or 2 s
    pause ("I'd like to book a table for, um, ... four people at seven")
    measure how often it cuts the speaker off, split by whether the pause
    follows a filler ("um,") or a bare mid-clause stop ("remind me to
    call"). The speech is TTS, and a TTS voice says a bare fragment with
    sentence-final intonation, so the bare-stop rates overstate what happens
    with people; read the comparison between policies."""
    from macvoice.agent import simulate
    from macvoice.endpointing import EndpointConfig, Endpointer
    from macvoice.turn import SmartTurn
    from macvoice.vad import SileroVAD

    vad, smart = SileroVAD(), SmartTurn()
    policies = {
        "fixed 500 ms": (None, EndpointConfig(max_silence_ms=500)),
        "fixed 1000 ms": (None, EndpointConfig(max_silence_ms=1000)),
        "fixed 1500 ms": (None, EndpointConfig(max_silence_ms=1500)),
        "smart-turn after 200 ms, max 2000 ms": (smart, EndpointConfig(check_silence_ms=200,
                                                                        max_silence_ms=2000)),
        "smart-turn after 300 ms, max 2000 ms": (smart, EndpointConfig(check_silence_ms=300,
                                                                        max_silence_ms=2000)),
        "smart-turn after 300 ms, max 3000 ms (default)": (smart, EndpointConfig(check_silence_ms=300,
                                                                                  max_silence_ms=3000)),
    }
    samples = turn_samples(pauses_s)
    out: dict = {"voices": TURN_VOICES_NAMES, "items": len(TURN_ITEMS), "pauses_s": list(pauses_s),
                 "complete_samples": sum(k == "complete" for k, _, _ in samples), "policies": {}}
    for name, (detector, config) in policies.items():
        latency, split = [], []
        cut = {(k, p): [] for k in ("filler", "bare") for p in pauses_s}
        for kind, pause, audio in samples:
            vad.reset()
            ends = [e for e, _ in simulate(audio, vad, Endpointer(detector, config))]
            if kind == "complete":
                split.append(len(ends) > 1)
                if ends:
                    latency.append((ends[-1].time_s - ends[-1].speech_end_s) * 1000)
            else:
                cut[(kind, pause)].append(len(ends) > 1)
        out["policies"][name] = {
            "complete_endpoint_ms_p50": pct(latency, 50),
            "complete_endpoint_ms_p90": pct(latency, 90),
            "complete_split_rate": float(np.mean(split)),
            **{f"cut_off_{k}_{p:g}s": float(np.mean(v)) for (k, p), v in cut.items()},
        }
    return out


def turns_table(r: dict) -> str:
    pauses = r["pauses_s"]
    cols = [(k, p) for k in ("filler", "bare") for p in pauses]
    names = {"filler": "um + {:g} s pause", "bare": "bare {:g} s pause"}
    lines = ["| Policy | Ends a finished turn after (p50 / p90) | Splits a finished turn | "
             + " | ".join("Cuts off: " + names[k].format(p) for k, p in cols) + " |",
             "|---|---|---|" + "---|" * len(cols)]
    for name, x in r["policies"].items():
        lines.append(f"| {name} | {x['complete_endpoint_ms_p50']:.0f} / {x['complete_endpoint_ms_p90']:.0f} ms"
                     f" | {x['complete_split_rate']:.0%} | "
                     + " | ".join(f"{x[f'cut_off_{k}_{p:g}s']:.0%}" for k, p in cols) + " |")
    return "\n".join(lines)


# ── main ──────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="macvoice bench")
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("stack")
    p.add_argument("--stack", choices=["new", "baseline"], default="new")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--out")
    p.add_argument("--in-process", action="store_true", help=argparse.SUPPRESS)
    p = sub.add_parser("compare")
    p.add_argument("json", nargs="+")
    p = sub.add_parser("turns")
    p.add_argument("--out")
    args = parser.parse_args(argv)

    if args.cmd == "stack":
        if not args.in_process:  # fresh interpreter, so footprints start clean
            cmd = [sys.executable, "-m", "macvoice.bench", "stack", "--stack", args.stack,
                   "--repeats", str(args.repeats), "--in-process"] + (["--out", args.out] if args.out else [])
            raise SystemExit(subprocess.call(cmd))
        result = stack(args.stack, args.repeats)
    elif args.cmd == "compare":
        print(compare(args.json))
        return
    else:
        result = turns()
        print(turns_table(result))

    text = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n")
    print(text if args.cmd == "stack" else "")


if __name__ == "__main__":
    main()
