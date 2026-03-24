"""
Train the EntropyTokenizer on TinyStories, then tokenize all shards.

Steps:
  1. Train EntropyTokenizer on a subset of the corpus (first N MB)
  2. Save tokenizer to data/entropy/tokenizer.bin
  3. Encode all train shards → data/entropy/train_NNNN.bin
  4. Encode val shard        → data/entropy/val_0000.bin

Usage:
  python data/prepare_entropy.py [options]

  --train_mb           : MB of text to train tokenizer on (default: 500)
  --vocab_size         : target vocabulary size (default: 50257)
  --boundary_percentile: percentile of h_i used as cut threshold (default: 80)
  --no_pmi             : disable PMI vocabulary filter
  --min_count          : minimum token frequency for vocabulary (default: 10)
  --smoke              : quick test: 5 MB corpus, vocab=8192, 1 shard output
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer.entropy_tokenizer import EntropyTokenizer
from tokenizer.tokenizer_utils import write_datafile

DATA_DIR = Path(__file__).parent


def load_training_corpus(raw_dir: Path, max_mb: float) -> bytes:
    """Read up to max_mb MB from raw text shards as a single bytes object."""
    max_bytes = int(max_mb * 1e6)
    parts: list[bytes] = []
    total = 0
    for shard in sorted(raw_dir.glob("train_*.txt")):
        text     = shard.read_text(encoding="utf-8", errors="replace")
        encoded  = text.encode("utf-8", errors="replace")
        needed   = max_bytes - total
        if needed <= 0:
            break
        if len(encoded) > needed:
            encoded = encoded[:needed]
        parts.append(encoded)
        total += len(encoded)
        if total >= max_bytes:
            break
    corpus = b"".join(parts)
    print(f"[corpus] loaded {total / 1e6:.1f} MB for tokenizer training",
          flush=True)
    return corpus


def encode_shard(
    shard: Path, out: Path, tok: EntropyTokenizer
) -> None:
    if out.exists():
        print(f"  skip {out.name} (already exists)", flush=True)
        return
    text = shard.read_text(encoding="utf-8", errors="replace")
    docs = text.split("<|endoftext|>")
    all_tokens: list[int] = []
    for doc in docs:
        doc = doc.strip()
        if doc:
            all_tokens.extend(tok.encode_with_eot(doc))
    write_datafile(all_tokens, out, vocab_size=tok.vocab_size)


def prepare_entropy(
    raw_dir: Path,
    out_dir: Path,
    train_mb: float              = 500.0,
    vocab_size: int              = 50257,
    boundary_percentile: int     = 80,
    use_pmi: bool                = True,
    min_count: int               = 10,
    smoke: bool                  = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tok_path = out_dir / "tokenizer.bin"

    if smoke:
        train_mb   = 5.0
        vocab_size = 8192
        print("[prepare_entropy] smoke mode: 5 MB corpus, vocab=8192", flush=True)

    # ── Train or load tokenizer ───────────────────────────────────────────
    tok = EntropyTokenizer(
        vocab_size           = vocab_size,
        boundary_percentile  = boundary_percentile,
        use_pmi              = use_pmi,
        min_count            = min_count,
    )

    if tok_path.exists():
        print(f"[prepare_entropy] loading existing tokenizer from {tok_path}",
              flush=True)
        tok.load(str(tok_path))
    else:
        corpus = load_training_corpus(raw_dir, max_mb=train_mb)
        tok.train(corpus, verbose=True)
        tok.save(str(tok_path))

    print(f"[prepare_entropy] vocab_size={tok.vocab_size}, "
          f"eot={tok.eot_token}", flush=True)

    # ── Encode training shards ────────────────────────────────────────────
    shards = sorted(raw_dir.glob("train_*.txt"))
    if smoke:
        shards = shards[:1]

    for shard in shards:
        encode_shard(shard, out_dir / (shard.stem + ".bin"), tok)

    # ── Encode validation shard ───────────────────────────────────────────
    val_shards = sorted(raw_dir.glob("val_*.txt"))
    if val_shards:
        encode_shard(val_shards[0], out_dir / "val_0000.bin", tok)

    print("[prepare_entropy] done", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare TinyStories with EntropyTokenizer"
    )
    parser.add_argument(
        "--data_dir", type=Path,
        default=DATA_DIR / "raw" / "tinystories",
    )
    parser.add_argument(
        "--out_dir", type=Path,
        default=DATA_DIR / "entropy",
    )
    parser.add_argument(
        "--train_mb", type=float, default=500.0,
        help="MB of corpus to train tokenizer on",
    )
    parser.add_argument(
        "--vocab_size", type=int, default=50257,
    )
    parser.add_argument(
        "--boundary_percentile", type=int, default=80,
        help="Percentile of h_i used as boundary threshold (higher → longer tokens)",
    )
    parser.add_argument(
        "--no_pmi", action="store_true",
        help="Disable PMI vocabulary filter",
    )
    parser.add_argument(
        "--min_count", type=int, default=10,
        help="Minimum token frequency to enter vocabulary",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Quick smoke test: 5 MB corpus, vocab=8192, 1 shard",
    )
    args = parser.parse_args()

    prepare_entropy(
        raw_dir              = args.data_dir,
        out_dir              = args.out_dir,
        train_mb             = args.train_mb,
        vocab_size           = args.vocab_size,
        boundary_percentile  = args.boundary_percentile,
        use_pmi              = not args.no_pmi,
        min_count            = args.min_count,
        smoke                = args.smoke,
    )


if __name__ == "__main__":
    main()
