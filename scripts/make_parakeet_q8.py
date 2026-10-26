"""Make (and optionally publish) the 8-bit Parakeet that macvoice downloads.

The published MLX checkpoint (mlx-community/parakeet-tdt-0.6b-v2) is fp32,
2.3 GB; macvoice runs it in bf16 with the encoder quantized to 8 bits,
which is 0.7 GB. This does that conversion once (``macvoice.asr.convert``,
the same code ``--precision q8`` falls back to) and writes a Hugging Face
model folder, so users download the small file and skip the conversion.

    uv run python scripts/make_parakeet_q8.py OUT_DIR
    uv run python scripts/make_parakeet_q8.py OUT_DIR --push johnrobinsn/parakeet-tdt-0.6b-v2-mlx-q8

Parakeet is CC BY 4.0: the model card credits NVIDIA, links the original
and its license, and says what was changed.
"""

import argparse
import json
import shutil
from pathlib import Path

from macvoice.asr import GROUP_SIZE, PARAKEET_MODEL, PARAKEET_REVISION, convert

CARD = f"""---
license: cc-by-4.0
library_name: mlx
pipeline_tag: automatic-speech-recognition
language: en
base_model: nvidia/parakeet-tdt-0.6b-v2
tags:
- parakeet
- mlx
- speech-recognition
---

# Parakeet TDT 0.6B v2, 8-bit encoder (MLX)

[NVIDIA's Parakeet TDT 0.6B v2](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)
English speech recognition model, for [parakeet-mlx](https://github.com/senstella/parakeet-mlx)
on Apple Silicon. 0.7 GB instead of 2.3 GB.

Made from the MLX conversion
[{PARAKEET_MODEL}](https://huggingface.co/{PARAKEET_MODEL}) at commit
`{PARAKEET_REVISION}` by
[`scripts/make_parakeet_q8.py`](https://github.com/johnrobinsn/macvoice/blob/main/scripts/make_parakeet_q8.py)
in [macvoice](https://github.com/johnrobinsn/macvoice).

## Changes from the original

- All weights cast to bfloat16.
- The encoder (about 99% of the weights) quantized to 8 bits: MLX affine
  quantization, group size {GROUP_SIZE}. The decoder and joint network stay
  in bfloat16.

On macvoice's tests the transcripts match the bf16 model's; on a 10k-utterance
English test set the 8-bit model's word error rate was statistically
indistinguishable from bf16.

## Loading

The weights are already quantized, so build the model, quantize its encoder
to get the same shapes, then load:

```python
import json
import mlx.nn as nn
from huggingface_hub import hf_hub_download
from parakeet_mlx.utils import from_config

repo = "johnrobinsn/parakeet-tdt-0.6b-v2-mlx-q8"
model = from_config(json.load(open(hf_hub_download(repo, "config.json"))))
nn.quantize(model.encoder, group_size={GROUP_SIZE}, bits=8)
model.load_weights(hf_hub_download(repo, "model.safetensors"))
```

`quantization.json` records the same settings.

## License

[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/), as the original.
The model was created by NVIDIA; see the
[original model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v2)
for its training data, intended use and limitations. This is a modified
version, not endorsed by NVIDIA.
"""


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", type=Path)
    p.add_argument("--push", metavar="REPO", help="upload to this Hugging Face repo (public)")
    args = p.parse_args()

    import mlx.core as mx
    from huggingface_hub import HfApi, hf_hub_download
    from mlx.utils import tree_flatten

    config, weights = (hf_hub_download(PARAKEET_MODEL, f, revision=PARAKEET_REVISION)
                       for f in ("config.json", "model.safetensors"))
    model = convert(config, weights, precision="q8")
    args.out.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(args.out / "model.safetensors"), dict(tree_flatten(model.parameters())),
                        metadata={"format": "mlx"})
    shutil.copyfile(config, args.out / "config.json")
    (args.out / "quantization.json").write_text(json.dumps(
        {"modules": ["encoder"], "mode": "affine", "group_size": GROUP_SIZE, "bits": 8,
         "other_dtype": "bfloat16", "source": f"{PARAKEET_MODEL}@{PARAKEET_REVISION}"}, indent=2) + "\n")
    (args.out / "README.md").write_text(CARD)
    size = (args.out / "model.safetensors").stat().st_size / 1e6
    print(f"wrote {args.out} (model.safetensors {size:.0f} MB)")

    if args.push:
        api = HfApi()
        api.create_repo(args.push, private=False, exist_ok=True)
        info = api.upload_folder(repo_id=args.push, folder_path=args.out,
                                 commit_message="8-bit encoder Parakeet TDT 0.6B v2 for MLX")
        print(f"pushed {info.oid} to https://huggingface.co/{args.push}")


if __name__ == "__main__":
    main()
