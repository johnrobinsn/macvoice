import json

import pytest

from macvoice.bench import compare, edit_distance, normalize, turns_table, wer


def test_normalize():
    assert normalize("Nor is Mister Quilter's manner, less-interesting!") == \
        ["nor", "is", "mister", "quilter's", "manner", "less", "interesting"]
    assert normalize("I have 3 cats") == ["i", "have", "three", "cats"]
    assert normalize("  ") == []


def test_edit_distance():
    assert edit_distance(list("abc"), list("abc")) == 0
    assert edit_distance(list("abc"), list("axc")) == 1
    assert edit_distance([], list("ab")) == 2
    assert edit_distance(list("ab"), []) == 2


def test_wer_ignores_case_and_punctuation():
    assert wer(["THE KING HAS FLED"], ["The king has fled."]) == 0
    assert wer(["a b c d"], ["a x c"]) == pytest.approx(0.5)


def test_compare_table(tmp_path):
    run = {"stack": "new", "asr": "parakeet-q8", "tts": "phonon", "torch_loaded": False,
           "asr_ms_p50": 60.2, "asr_ms_p95": 72.9, "asr_wer": 0.044,
           "footprint_mb": {"+tts": 1268.4, "end": 1405.0}, "peak_footprint_mb": 1546.0}
    p = tmp_path / "a.json"
    p.write_text(json.dumps(run))
    table = compare([str(p)])
    assert "| | new |" in table
    assert "| ASR latency p50 / p95 (ms) | 60 / 73 |" in table
    assert "| ASR WER (20 LibriSpeech clips) | 4.4% |" in table
    assert "| Footprint, models loaded (MB) | 1268 |" in table
    assert "| TTS real-time factor | - |" in table          # missing keys don't break it


def test_turns_table():
    r = {"pauses_s": [1.0], "policies": {"p": {
        "complete_endpoint_ms_p50": 288, "complete_endpoint_ms_p90": 300, "complete_split_rate": 0.0,
        "cut_off_filler_1s": 0.0, "cut_off_bare_1s": 0.5}}}
    table = turns_table(r)
    assert "| p | 288 / 300 ms | 0% | 0% | 50% |" in table
