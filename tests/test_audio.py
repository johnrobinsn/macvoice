import numpy as np
import pytest

from macvoice.audio import Chunker, StreamResampler, frames, read_wav, resample, write_wav
from tests.conftest import tone


def dominant_hz(x: np.ndarray, rate: int) -> float:
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    return float(np.argmax(spec) * rate / len(x))


def test_wav_round_trip(tmp_path):
    x = tone(440, 0.5)
    write_wav(tmp_path / "a.wav", x, 16000)
    y, rate = read_wav(tmp_path / "a.wav")
    assert rate == 16000
    assert len(y) == len(x)
    assert np.max(np.abs(y - x)) < 1e-4


def test_write_clips_out_of_range(tmp_path):
    write_wav(tmp_path / "a.wav", np.array([2.0, -2.0], dtype=np.float32), 16000)
    y, _ = read_wav(tmp_path / "a.wav")
    assert y.max() <= 1.0 and y.min() >= -1.0


def test_read_resamples_on_request(tmp_path):
    write_wav(tmp_path / "a.wav", tone(1000, 1.0, rate=24000), 24000)
    y, rate = read_wav(tmp_path / "a.wav", 16000)
    assert rate == 16000
    assert len(y) == 16000
    assert dominant_hz(y, 16000) == pytest.approx(1000, abs=2)


def test_read_rejects_non_16_bit(tmp_path):
    import wave
    with wave.open(str(tmp_path / "a.wav"), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(1)
        w.setframerate(8000)
        w.writeframes(bytes(100))
    with pytest.raises(ValueError, match="16-bit"):
        read_wav(tmp_path / "a.wav")


@pytest.mark.parametrize("src,dst", [(24000, 16000), (48000, 16000), (16000, 24000), (44100, 16000)])
def test_resample_keeps_pitch_and_duration(src, dst):
    x = tone(500, 0.8, rate=src)
    y = resample(x, src, dst)
    assert len(y) == round(len(x) * dst / src)
    assert dominant_hz(y, dst) == pytest.approx(500, abs=3)
    assert np.sqrt(np.mean(y**2)) == pytest.approx(np.sqrt(np.mean(x**2)), rel=0.02)


def test_resample_removes_content_above_new_nyquist():
    x = tone(7000, 1.0, rate=24000) + tone(300, 1.0, rate=24000)
    y = resample(x, 24000, 8000)  # Nyquist 4 kHz: the 7 kHz tone must go, not alias
    spec = np.abs(np.fft.rfft(y))
    freqs = np.fft.rfftfreq(len(y), 1 / 8000)
    assert spec[np.abs(freqs - 300) < 5].max() > 100 * spec[np.abs(freqs - 1000) < 50].max()


def test_resample_identity_and_empty():
    x = tone(440, 0.1)
    assert np.array_equal(resample(x, 16000, 16000), x)
    assert len(resample(np.zeros(0), 24000, 16000)) == 0


def test_frames_pads_last_chunk():
    chunks = frames(np.ones(1100, dtype=np.float32), 512)
    assert [len(c) for c in chunks] == [512, 512, 512]
    assert chunks[2][:76].sum() == 76 and chunks[2][76:].sum() == 0
    assert frames(np.zeros(0), 512) == []


@pytest.mark.parametrize("src", [48000, 44100, 24000])
def test_stream_resampler_matches_whole_clip_resampling(src):
    rng = np.random.default_rng(0)
    x = tone(300, 1.0, src) + tone(1700, 1.0, src, amp=0.2)
    blocks = np.array_split(x, rng.integers(5, 40))
    y = np.concatenate([StreamResampler(src, 16000)(b) for b in [x]])
    y_blocks = np.concatenate(list(map(StreamResampler(src, 16000), blocks)))
    # Same output however the stream is split, aligned in time with resample().
    np.testing.assert_allclose(y_blocks, y[: len(y_blocks)], atol=1e-6)
    ref = resample(x, src, 16000)[: len(y)]
    mid = slice(200, len(y) - 200)  # away from the clip edges
    assert np.max(np.abs(y[mid] - ref[mid])) < 0.02
    assert len(y) >= 16000 - 20


def test_stream_resampler_removes_content_above_new_nyquist():
    x = tone(12000, 1.0, 48000)  # would alias to 4 kHz
    y = StreamResampler(48000, 16000)(x)
    assert np.sqrt(np.mean(y[100:] ** 2)) < 0.01


def test_stream_resampler_keeps_pitch():
    rs = StreamResampler(48000, 16000)
    y = np.concatenate([rs(b) for b in np.array_split(tone(440, 2.0, 48000), 20)])
    assert dominant_hz(y, 16000) == pytest.approx(440, abs=2)


def test_chunker_regroups_blocks():
    chunk = Chunker(512)
    x = np.arange(3000, dtype=np.float32)
    out = [c for b in np.array_split(x, 7) for c in chunk(b)]
    assert [len(c) for c in out] == [512] * 5
    np.testing.assert_array_equal(np.concatenate(out), x[:2560])
