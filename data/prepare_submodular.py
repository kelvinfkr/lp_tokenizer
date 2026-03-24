"""
Train SubmodularTokenizer on TinyStories, then tokenize all shards.

Steps:
  1. Train SubmodularTokenizer on a subset of the corpus (first N MB)
  2. Save tokenizer to data/submodular/tokenizer.bin
  3. Encode all train shards → data/submodular/train_NNNN.bin
  4. Encode val shard        → data/submodular/val_0000.bin

Usage:
  python data/prepare_submodular.py [options]

  --train_mb            : MB of text to train tokenizer on (default: 500)
  --vocab_size          : target vocabulary size (default: 50257)
  --boundary_percentile : entropy-patching threshold percentile (default: 80)
  --sample_mb           : MB of corpus used for DP evaluations (default: 2)
  --candidate_factor    : candidate pool = factor × (vocab_size - 257) (default: 10)
  --min_count           : minimum token frequency (default: 10)
  --recompute_interval  : full DP recompute every N greedy steps (default: 500)
  --smoke               : quick test: 5 MB corpus, vocab=8192, 1 shard
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer.submodular_tokenizer import SubmodularTokenizer
from tokenizer.tokenizer_utils import write_datafile

DATA_DIR = Path(__file__).parent


def load_training_corpus(raw_dir: Path, max_mb: float) -> bytes:
    """Read up to max_mb MB from raw text shards as a single bytes object."""
    max_bytes = int(max_mb * 1e6)
    parts: list[bytes] = []
    total = 0
    for shard in sorted(raw_dir.glob("train_*.txt")):
        encoded = shard.read_text(encoding="utf-8", errors="replace").encode(
            "utf-8", errors="replace"
        )
        needed = max_bytes - total
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


def encode_shard(shard: Path, out: Path, tok: SubmodularTokenizer) -> None:
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


def prepare_submodular(
    raw_dir:             Path,
    out_dir:             Path,
    train_mb:            float = 500.0,
    vocab_size:          int   = 50257,
    boundary_percentile: int   = 80,
    sample_mb:           float = 2.0,
    candidate_factor:    int   = 10,
    min_count:           int   = 10,
    recompute_interval:  int   = 500,
    smoke:               bool  = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    tok_path = out_dir / "tokenizer.bin"

    if smoke:
        train_mb   = 5.0
        vocab_size = 8192
        sample_mb  = 0.5
        print("[prepare_submodular] smoke mode: 5 MB corpus, vocab=8192",
              flush=True)

    tok = SubmodularTokenizer(
        vocab_size           = vocab_size,
        boundary_percentile  = boundary_percentile,
        sample_mb            = sample_mb,
        candidate_factor     = candidate_factor,
        min_count            = min_count,
        recompute_interval   = recompute_interval,
    )

    if tok_path.exists():
        print(f"[prepare_submodular] loading existing tokenizer from {tok_path}",
              flush=True)
        tok.load(str(tok_path))
    else:
        corpus = load_training_corpus(raw_dir, max_mb=train_mb)
        tok.train(corpus, verbose=True)
        tok.save(str(tok_path))

    print(f"[prepare_submodular] vocab_size={tok.vocab_size}, "
          f"eot={tok.eot_token}", flush=True)

    # Encode shards
    shards = sorted(raw_dir.glob("train_*.txt"))
    if smoke:
        shards = shards[:1]
    for shard in shards:
        encode_shard(shard, out_dir / (shard.stem + ".bin"), tok)

    val_shards = sorted(raw_dir.glob("val_*.txt"))
    if val_shards:
        encode_shard(val_shards[0], out_dir / "val_0000.bin", tok)

    print("[prepare_submodular] done", flush=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Prepare TinyStories with SubmodularTokenizer"
    )
    p.add_argument("--data_dir",            type=Path,  default=DATA_DIR / "raw" / "tinystories")
    p.add_argument("--out_dir",             type=Path,  default=DATA_DIR / "submodular")
    p.add_argument("--train_mb",            type=float, default=500.0)
    p.add_argument("--vocab_size",          type=int,   default=50257)
    p.add_argument("--boundary_percentile", type=int,   default=80)
    p.add_argument("--sample_mb",           type=float, default=2.0,
                   help="MB sample for DP evaluations during greedy")
    p.add_argument("--candidate_factor",    type=int,   default=10)
    p.add_argument("--min_count",           type=int,   default=10)
    p.add_argument("--recompute_interval",  type=int,   default=500)
    p.add_argument("--smoke",               action="store_true")
    args = p.parse_args()

    prepare_submodular(
        raw_dir             = args.data_dir,
        out_dir             = args.out_dir,
        train_mb            = args.train_mb,
        vocab_size          = args.vocab_size,
        boundary_percentile = args.boundary_percentile,
        sample_mb           = args.sample_mb,
        candidate_factor    = args.candidate_factor,
        min_count           = args.min_count,
        recompute_interval  = args.recompute_interval,
        smoke               = args.smoke,
    )


if __name__ == "__main__":
    main()
