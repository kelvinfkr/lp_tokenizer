"""
Compare tokenizer quality metrics: BPE vs Clustering.

Prints a side-by-side table of:
  - vocab_size
  - compression_ratio (bytes/token)
  - fertility (tokens/word)
  - encode_speed (MB/s)
  - vocab_coverage (fraction of non-byte tokens)

Usage:
  python eval/tokenizer_metrics.py [--text_file data/raw/tinystories/val_0000.txt]
  python eval/tokenizer_metrics.py --smoke   # uses built-in sample text
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer.tokenizer_utils import TokenizerMetrics
from tokenizer.bpe_baseline import BPETokenizer

SAMPLE_TEXT = """\
Once upon a time, there was a little girl named Lily. She loved to play in the garden.
One day, she found a small caterpillar on a leaf. She named it Charlie.
Charlie grew bigger every day. Lily watched him carefully.
One morning, Charlie made a cocoon. Lily waited and waited.
Finally, a beautiful butterfly came out. Lily was so happy.
She let the butterfly fly away. "Goodbye, Charlie!" she said.
""" * 50  # ~3KB sample


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_file", type=Path, default=None,
                        help="Path to text file for evaluation")
    parser.add_argument("--clustering_tokenizer", type=Path, default=None,
                        help="Path to trained clustering tokenizer .bin")
    parser.add_argument("--smoke", action="store_true",
                        help="Use built-in sample text")
    parser.add_argument("--out_json", type=Path, default=None,
                        help="Write results to JSON")
    args = parser.parse_args()

    # Load evaluation text
    if args.smoke or args.text_file is None:
        text = SAMPLE_TEXT
        print("[eval] using built-in sample text (~3KB)")
    else:
        text = args.text_file.read_text(encoding="utf-8", errors="replace")
        # Use first 2MB for speed
        text = text[:2_000_000]
        print(f"[eval] loaded {len(text)/1e6:.1f} MB from {args.text_file}")

    results = []

    # ── BPE baseline ─────────────────────────────────────────────
    print("\nEvaluating BPE baseline ...")
    bpe = BPETokenizer()
    bpe_result = TokenizerMetrics.evaluate(text, bpe, name="BPE (tiktoken gpt2)")
    results.append(bpe_result)

    # ── Clustering tokenizer ──────────────────────────────────────
    if args.clustering_tokenizer and args.clustering_tokenizer.exists():
        from tokenizer.clustering_tokenizer import ClusteringTokenizer
        print("\nEvaluating Clustering tokenizer ...")
        cls_tok = ClusteringTokenizer()
        cls_tok.load(str(args.clustering_tokenizer))
        cls_result = TokenizerMetrics.evaluate(
            text, cls_tok, name="Clustering (MinHash+LSH)"
        )
        results.append(cls_result)

        # ── Side-by-side comparison ───────────────────────────────
        print("\n" + "═" * 60)
        print("COMPARISON SUMMARY")
        print("═" * 60)
        keys = [
            "vocab_size",
            "compression_ratio_bytes_per_token",
            "fertility_tokens_per_word",
            "encode_speed_mbs",
            "vocab_coverage",
        ]
        fmt = "{:<40} {:>12} {:>12}"
        print(fmt.format("metric", "BPE", "Clustering"))
        print("-" * 64)
        for k in keys:
            bv = bpe_result.get(k, "—")
            cv = cls_result.get(k, "—")
            print(fmt.format(k, str(bv), str(cv)))

        # Relative differences
        print("\nRelative to BPE (positive = clustering better):")
        if bpe_result["compression_ratio_bytes_per_token"] > 0:
            ratio = cls_result["compression_ratio_bytes_per_token"] / \
                    bpe_result["compression_ratio_bytes_per_token"]
            print(f"  compression_ratio: {ratio:.3f}x  "
                  f"({'better' if ratio >= 1 else 'worse'} compression)")
        if bpe_result["fertility_tokens_per_word"] > 0:
            ratio = cls_result["fertility_tokens_per_word"] / \
                    bpe_result["fertility_tokens_per_word"]
            print(f"  fertility:         {ratio:.3f}x  "
                  f"({'better' if ratio <= 1 else 'worse'} fertility)")
        if bpe_result["encode_speed_mbs"] > 0:
            ratio = cls_result["encode_speed_mbs"] / bpe_result["encode_speed_mbs"]
            print(f"  encode_speed:      {ratio:.2f}x  "
                  f"({'faster' if ratio >= 1 else 'slower'})")
    else:
        print("\n[eval] No clustering tokenizer provided. "
              "Run `python data/prepare_clustering.py` first, then pass "
              "--clustering_tokenizer data/clustering/tokenizer.bin")

    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        args.out_json.write_text(json.dumps(results, indent=2))
        print(f"\n[eval] results written to {args.out_json}")


if __name__ == "__main__":
    main()
