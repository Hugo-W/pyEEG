"""Download distilgpt2 from ModelScope (no HuggingFace Hub access needed).

Downloads the essential files for loading via GPT2LMHeadModel.from_pretrained:
config.json, tokenizer files, and pytorch_model.bin (~334 MB).

Usage::

    uv run --extra features python tests/download_distilgpt2.py
"""
import os
import urllib.request
from pathlib import Path

MODEL_ID = "Intel/distilgpt2-wikitext2"
BASE_URL = "https://modelscope.ai/api/v1/models/{model}/repo?Revision=master&FilePath={file}"
OUT_DIR = Path.home() / ".cache" / "huggingface" / "distilgpt2"

# Essential files (skip training artifacts, README, etc.)
FILES = [
    "config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.json",
    "pytorch_model.bin",
]


def _file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for fname in FILES:
        out_path = OUT_DIR / fname
        if out_path.exists() and out_path.stat().st_size > 0:
            print(f"  [skip] {fname} ({_file_size_mb(out_path):.1f} MB)")
            continue

        url = BASE_URL.format(model=MODEL_ID, file=fname)
        print(f"  [download] {fname} ...")
        urllib.request.urlretrieve(url, out_path)
        print(f"    done ({_file_size_mb(out_path):.1f} MB)")

    print(f"\nSaved distilgpt2 to {OUT_DIR}")
    print("Total size: {:.1f} MB".format(
        sum(f.stat().st_size for f in OUT_DIR.iterdir() if f.is_file()) / (1024 * 1024)
    ))


if __name__ == "__main__":
    main()
