"""
Download training data for tokenizer experiments.

Datasets:
  - TinyStories : ~2GB children's stories (roneneldan/TinyStories on HuggingFace)
  - TinyShakespeare : ~1MB for quick smoke tests

Usage:
  python data/download.py --dataset tinystories
  python data/download.py --dataset tinyshakespeare
  python data/download.py --dataset all
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

DATA_DIR = Path(__file__).parent

TINYSHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
    "tinyshakespeare/input.txt"
)


def download_tinyshakespeare(dest: Path = DATA_DIR / "raw" / "tinyshakespeare"):
    dest.mkdir(parents=True, exist_ok=True)
    out = dest / "input.txt"
    if out.exists():
        print(f"[download] {out} already exists, skipping")
        return
    import urllib.request
    print(f"[download] TinyShakespeare → {out}")
    urllib.request.urlretrieve(TINYSHAKESPEARE_URL, out)
    print(f"  {out.stat().st_size / 1e6:.1f} MB")


def download_tinystories(dest: Path = DATA_DIR / "raw" / "tinystories"):
    dest.mkdir(parents=True, exist_ok=True)
    # Check if already downloaded
    existing = list(dest.glob("*.txt")) + list(dest.glob("*.json"))
    if existing:
        print(f"[download] TinyStories already in {dest} ({len(existing)} files)")
        return

    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError(
            "Install `datasets` package: pip install datasets"
        )

    print(f"[download] TinyStories from HuggingFace → {dest}")
    ds = load_dataset("roneneldan/TinyStories", split="train")

    # Write shards of ~100MB each
    shard_size = 100_000  # documents per shard
    shard_idx  = 0
    buf: list[str] = []

    for i, item in enumerate(ds):
        buf.append(item["text"])
        if len(buf) >= shard_size:
            out = dest / f"train_{shard_idx:04d}.txt"
            out.write_text("\n<|endoftext|>\n".join(buf), encoding="utf-8")
            print(f"  shard {shard_idx}: {out.stat().st_size / 1e6:.0f} MB")
            shard_idx += 1
            buf = []
        if (i + 1) % 10_000 == 0:
            print(f"  processed {i+1:,} documents ...")

    if buf:
        out = dest / f"train_{shard_idx:04d}.txt"
        out.write_text("\n<|endoftext|>\n".join(buf), encoding="utf-8")
        print(f"  shard {shard_idx}: {out.stat().st_size / 1e6:.0f} MB")

    # Validation split
    ds_val = load_dataset("roneneldan/TinyStories", split="validation")
    val_texts = [item["text"] for item in ds_val]
    val_out = dest / "val_0000.txt"
    val_out.write_text("\n<|endoftext|>\n".join(val_texts), encoding="utf-8")
    print(f"  val shard: {val_out.stat().st_size / 1e6:.0f} MB")
    print(f"[download] TinyStories done → {dest}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["tinystories", "tinyshakespeare", "all"],
        default="tinystories",
    )
    args = parser.parse_args()

    if args.dataset in ("tinyshakespeare", "all"):
        download_tinyshakespeare()

    if args.dataset in ("tinystories", "all"):
        download_tinystories()


if __name__ == "__main__":
    main()
