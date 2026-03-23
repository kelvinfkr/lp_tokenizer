"""
Train the clustering tokenizer on TinyStories, then tokenize all shards.

Steps:
  1. Train ClusteringTokenizer on a subset of the corpus (first N MB)
  2. Save tokenizer to data/clustering/tokenizer.bin
  3. Encode all train shards → data/clustering/train_NNNN.bin
  4. Encode val shard      → data/clustering/val_0000.bin

Usage:
  python data/prepare_clustering.py [options]

  --train_mb     : how many MB of text to train tokenizer on (default: 500)
  --vocab_size   : target vocabulary size (default: 50257)
  --smoke        : quick test with 5MB training corpus, 1 shard output
  --device       : cuda or cpu (default: auto-detect)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer.clustering_tokenizer import ClusteringTokenizer
from tokenizer.tokenizer_utils import write_datafile

DATA_DIR = Path(__file__).parent


def load_training_corpus(raw_dir: Path, max_mb: float) -> str:
    """Read up to max_mb MB from raw text shards for tokenizer training."""
    max_bytes = int(max_mb * 1e6)
    buf: list[str] = []
    total = 0
    for shard in sorted(raw_dir.glob("train_*.txt")):
        text = shard.read_text(encoding="utf-8", errors="replace")
        needed = max_bytes - total
        if needed <= 0:
            break
        if len(text.encode("utf-8")) > needed:
            text = text[:needed]
        buf.append(text)
        total += len(text.encode("utf-8"))
        if total >= max_bytes:
            break
    corpus = "\n".join(buf)
    print(f"[corpus] loaded {total/1e6:.1f} MB for tokenizer training")
    return corpus


def encode_shard(shard: Path, out: Path, tok: ClusteringTokenizer) -> None:
    if out.exists():
        print(f"  skip {out.name} (already exists)")
        return
    text = shard.read_text(encoding="utf-8", errors="replace")
    docs = text.split("<|endoftext|>")
    all_tokens: list[int] = []
    for doc in docs:
        doc = doc.strip()
        if doc:
            all_tokens.extend(tok.encode_with_eot(doc))
    write_datafile(all_tokens, out, vocab_size=tok.vocab_size)


def prepare_clustering(
    raw_dir: Path,
    out_dir: Path,
    train_mb: float = 500.0,
    vocab_size: int = 50257,
    smoke: bool = False,
    device: str | None = None,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    tok_path = out_dir / "tokenizer.bin"

    if smoke:
        train_mb   = 5.0
        vocab_size = 8192
        print("[prepare_clustering] smoke mode: 5MB corpus, vocab=8192")

    # ── Train tokenizer ──────────────────────────────────────────
    if tok_path.exists():
        print(f"[prepare_clustering] loading existing tokenizer from {tok_path}")
        tok = ClusteringTokenizer(vocab_size=vocab_size, device=device)
        tok.load(str(tok_path))
    else:
        corpus = load_training_corpus(raw_dir, max_mb=train_mb)
        kwargs: dict = {"vocab_size": vocab_size, "verbose": True}
        if device:
            kwargs["device"] = device
        tok = ClusteringTokenizer(**kwargs)
        tok.train(corpus)
        tok.save(str(tok_path))

    print(f"[prepare_clustering] vocab_size={tok.vocab_size}, eot={tok.eot_token}")

    # ── Encode training shards ────────────────────────────────────
    shards = sorted(raw_dir.glob("train_*.txt"))
    if smoke:
        shards = shards[:1]

    for shard in shards:
        encode_shard(shard, out_dir / (shard.stem + ".bin"), tok)

    # ── Encode validation shard ───────────────────────────────────
    val_shards = sorted(raw_dir.glob("val_*.txt"))
    if val_shards:
        encode_shard(val_shards[0], out_dir / "val_0000.bin", tok)

    print("[prepare_clustering] done")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir",   type=Path,
                        default=DATA_DIR / "raw" / "tinystories")
    parser.add_argument("--out_dir",    type=Path,
                        default=DATA_DIR / "clustering")
    parser.add_argument("--train_mb",   type=float, default=500.0,
                        help="MB of corpus to train tokenizer on")
    parser.add_argument("--vocab_size", type=int,   default=50257)
    parser.add_argument("--smoke",      action="store_true")
    parser.add_argument("--device",     type=str,   default=None,
                        help="cuda or cpu (default: auto)")
    args = parser.parse_args()

    prepare_clustering(
        raw_dir    = args.data_dir,
        out_dir    = args.out_dir,
        train_mb   = args.train_mb,
        vocab_size = args.vocab_size,
        smoke      = args.smoke,
        device     = args.device,
    )


if __name__ == "__main__":
    main()
