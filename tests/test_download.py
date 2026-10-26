"""The model list, ``macvoice download`` and the first-run download."""

import hashlib
import io
import os

import pytest

from macvoice import asr, models, vad


def labels(items):
    return list(dict.fromkeys(a.label for a in items))


def test_default_models_are_under_a_gigabyte():
    items = models.assets()
    assert labels(items) == ["voice activity (Silero VAD)", "end of turn (Smart Turn)",
                             "speech to text (Parakeet, 8-bit)", "text to speech (Phonon)"]
    assert 0.8e9 < sum(a.size for a in items) < 0.9e9
    assert {a.repo for a in items if "Parakeet" in a.label} == {asr.PARAKEET_Q8}


def test_every_hub_file_is_pinned_to_a_commit():
    for a in models.assets() + models.parakeet("bf16"):
        if a.repo:
            assert a.revision and len(a.revision) == 40, a


def test_other_precisions_download_the_original_checkpoint():
    items = models.parakeet("q4")
    assert {a.repo for a in items} == {asr.PARAKEET_MODEL}
    assert sum(a.size for a in items) > 2.4e9


def test_options_leave_out_what_is_not_loaded():
    assert "end of turn (Smart Turn)" not in labels(models.assets(turn_detector=False))
    assert not any("Parakeet" in label for label in labels(models.assets("whisper")))
    assert "text to speech (Phonon)" not in labels(models.assets(need_tts=False))
    assert "text to speech (Phonon)" not in labels(models.assets(tts_name="kokoro"))


@pytest.fixture
def fake_hub(monkeypatch):
    """Assets are 'cached' once fetched; records the fetches."""
    have, fetched = set(), []
    monkeypatch.setattr(models.Asset, "cached", lambda self: (self.repo, self.filename) in have)

    def fetch(self):
        fetched.append(self.filename)
        have.add((self.repo, self.filename))

    monkeypatch.setattr(models.Asset, "fetch", fetch)
    return have, fetched


def test_ensure_downloads_only_what_is_missing(fake_hub):
    have, fetched = fake_hub
    items = models.assets()
    have.update((a.repo, a.filename) for a in items if "Parakeet" not in a.label)
    out = io.StringIO()
    got = models.ensure(items, out)
    assert fetched == ["config.json", "model.safetensors"]
    assert got == 36176 + 736486649
    assert "Downloading models, 737 MB (one time only)" in out.getvalue()
    assert "speech to text (Parakeet, 8-bit)" in out.getvalue()
    assert "Phonon" not in out.getvalue()


def test_ensure_is_silent_when_everything_is_there(fake_hub):
    have, fetched = fake_hub
    have.update((a.repo, a.filename) for a in models.assets())
    out = io.StringIO()
    assert models.ensure(models.assets(), out) == 0
    assert out.getvalue() == "" and fetched == []


def test_download_command(fake_hub):
    have, fetched = fake_hub
    have.update((a.repo, a.filename) for a in models.silero())
    out = io.StringIO()
    models.download(models.assets(), out)
    text = out.getvalue()
    assert text.startswith("Already downloaded: voice activity (Silero VAD)\n")
    assert "All models ready (851 MB). Next: macvoice run" in text
    out = io.StringIO()
    models.download(models.assets(), out)
    assert out.getvalue() == "All models are already downloaded (851 MB). Next: macvoice run\n"


def test_cli_download_uses_the_options(monkeypatch):
    from macvoice import cli

    seen = []
    monkeypatch.setattr(models, "download", lambda items: seen.append(items))
    cli.main(["download", "--precision", "bf16"])
    assert {a.repo for a in seen[0] if "Parakeet" in a.label} == {asr.PARAKEET_MODEL}


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


def fake_urlretrieve(content):
    def retrieve(url, dest):
        with open(dest, "wb") as f:
            f.write(content)
    return retrieve


def test_silero_is_checked_against_its_checksum(home, monkeypatch):
    monkeypatch.setattr(vad.urllib.request, "urlretrieve", fake_urlretrieve(b"not the model"))
    with pytest.raises(RuntimeError, match="SHA-256"):
        vad.model_path()
    assert list((home / ".cache/macvoice").iterdir()) == []


def test_silero_download_lands_in_the_cache(home, monkeypatch):
    monkeypatch.setattr(vad.urllib.request, "urlretrieve", fake_urlretrieve(b"model"))
    monkeypatch.setattr(vad, "SHA256", hashlib.sha256(b"model").hexdigest())
    path = vad.model_path()
    assert path == home / ".cache/macvoice/silero_vad.onnx" and path.read_bytes() == b"model"
    assert models.silero()[0].cached()


def test_parakeet_falls_back_to_converting_when_the_8bit_copy_is_unavailable(monkeypatch, caplog):
    pytest.importorskip("mlx.core")
    calls = []

    def model_files(repo, revision):
        calls.append(repo)
        if repo == asr.PARAKEET_Q8:
            raise OSError("offline")
        return "config.json", "model.safetensors"

    class Model:
        def parameters(self):
            return {}

    monkeypatch.setattr(asr, "model_files", model_files)
    monkeypatch.setattr(asr, "convert", lambda c, w, precision: Model())
    asr.Parakeet()
    assert calls == [asr.PARAKEET_Q8, asr.PARAKEET_MODEL]
    assert "converting mlx-community/parakeet-tdt-0.6b-v2 instead" in caplog.text


@pytest.mark.models
def test_cached_files_have_the_listed_sizes():
    """The sizes are typed in (so ``download`` can say how big before asking
    the Hub); the real files must match. Runs after the model tests have
    cached everything."""
    from huggingface_hub import try_to_load_from_cache

    models.ensure(models.assets())
    for a in models.assets():
        path = try_to_load_from_cache(a.repo, a.filename, revision=a.revision) if a.repo else vad.cache_path()
        assert os.path.getsize(path) == a.size, a


@pytest.mark.models
def test_published_8bit_parakeet_matches_converting_it_here(parakeet, librispeech):
    """The Hub copy is what ``convert`` makes from the pinned original.
    Skipped unless the 2.5 GB original is already cached."""
    from huggingface_hub import try_to_load_from_cache

    if not isinstance(try_to_load_from_cache(asr.PARAKEET_MODEL, "model.safetensors",
                                             revision=asr.PARAKEET_REVISION), str):
        pytest.skip("the fp32 original isn't cached")
    import mlx.core as mx
    from mlx.utils import tree_flatten

    published = dict(tree_flatten(parakeet._thread.run(lambda: parakeet._model.parameters())))

    def converted():
        model = asr.convert(*asr.model_files(asr.PARAKEET_MODEL, asr.PARAKEET_REVISION), precision="q8")
        params = dict(tree_flatten(model.parameters()))
        return all(mx.array_equal(params[k], v).item() for k, v in published.items()) and \
            params.keys() == published.keys()

    assert parakeet._thread.run(converted)
