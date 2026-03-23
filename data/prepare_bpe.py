"""
Tokenize TinyStories with BPE (tiktoken gpt2) and write llm.c binary files.

Output:
  data/bpe/train_NNNN.bin  — training shards
  data/bpe/val_0000.bin    — validation shard

Usage:
  python data/prepare_bpe.py [--data_dir data/raw/tinystories] [--smoke]

--smoke : only process the first shard (quick sanity check)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Make project root importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer.bpe_baseline import BPETokenizer
from tokenizer.tokenizer_utils import write_datafile

DATA_DIR = Path(__file__).parent


def prepare_bpe(raw_dir: Path, out_dir: Path, smoke: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = BPETokenizer()
    print(f"[prepare_bpe] vocab_size={tok.n_vocab}, eot={tok.eot_token}")
    print(f"  raw_dir: {raw_dir}")
    print(f"  out_dir: {out_dir}")

    shards = sorted(raw_dir.glob("train_*.txt"))
    if smoke:
        shards = shards[:1]
        print("  [smoke mode] processing 1 training shard only")

    for shard in shards:
        out = out_dir / (shard.stem + ".bin")
        if out.exists():
            print(f"  skip {out.name} (already exists)")
            continue
        text = shard.read_text(encoding="utf-8", errors="replace")
        # Split on document boundary marker and prepend EOT for each doc
        docs = text.split("<|endoftext|>")
        all_tokens: list[int] = []
        for doc in docs:
            doc = doc.strip()
            if doc:
                all_tokens.extend(tok.encode_with_eot(doc))
        write_datafile(all_tokens, out, vocab_size=tok.n_vocab)

    # Validation shard
    val_shards = sorted(raw_dir.glob("val_*.txt"))
    if val_shards:
        val_shard = val_shards[0]
        val_out   = out_dir / "val_0000.bin"
        if not val_out.exists():
            text = val_shard.read_text(encoding="utf-8", errors="replace")
            docs = text.split("<|endoftext|>")
            all_tokens = []
            for doc in docs:
                doc = doc.strip()
                if doc:
                    all_tokens.extend(tok.encode_with_eot(doc))
            write_datafile(all_tokens, val_out, vocab_size=tok.n_vocab)
        else:
            print(f"  skip {val_out.name} (already exists)")

    print("[prepare_bpe] done")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path,
                        default=DATA_DIR / "raw" / "tinystories")
    parser.add_argument("--out_dir",  type=Path,
                        default=DATA_DIR / "bpe")
    parser.add_argument("--smoke", action="store_true",
                        help="Process only 1 shard for quick testing")
    args = parser.parse_args()
    prepare_bpe(args.data_dir, args.out_dir, args.smoke)


if __name__ == "__main__":
    main()
