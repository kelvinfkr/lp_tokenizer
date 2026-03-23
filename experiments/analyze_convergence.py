"""
分析收敛速度实验结果。

关键问题：
  "在达到相同 val loss (e.g. 3.2 nats) 时，
   BPE vs 聚类分词器各消耗了多少 token / byte？"

公平对比需要把 token 消耗转换为 byte 消耗，因为：
  - BPE：1 token ≈ 3.6 bytes（GPT-2 on English）
  - Clustering：1 token ≈ X bytes（取决于词表质量）

两者消耗相同 byte 数时，谁的 val loss 更低 → 谁更好。

Usage:
  python experiments/analyze_convergence.py \
    --bpe_log experiments/out/convergence/bpe/log.txt \
    --cls_log experiments/out/convergence/clustering/log.txt \
    --target_loss 3.2
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))


def parse_log(log_path: Path) -> list[dict]:
    """
    Parse llm.c training log into a list of checkpoint records.

    Returns list of dicts: {step, train_loss, val_loss}
    """
    records: list[dict] = []
    current: dict = {}

    with open(log_path) as f:
        for line in f:
            m = re.search(r"step\s+(\d+).*train loss\s+([\d.]+)", line)
            if m:
                if current and "val_loss" not in current:
                    records.append(current)
                current = {"step": int(m.group(1)),
                            "train_loss": float(m.group(2))}

            m2 = re.search(r"val loss\s+([\d.]+)", line)
            if m2 and current:
                current["val_loss"] = float(m2.group(1))
                records.append(current)
                current = {}

    if current and "val_loss" in current:
        records.append(current)

    return records


def tokens_consumed(step: int, batch_size: int, seq_len: int,
                    total_batch_size: int) -> int:
    """Total tokens consumed after `step` gradient updates."""
    # Each gradient step processes total_batch_size tokens
    return step * total_batch_size


def bytes_consumed(n_tokens: int, avg_bytes_per_token: float) -> float:
    return n_tokens * avg_bytes_per_token


def find_step_for_target(records: list[dict], target_loss: float) -> dict | None:
    """Find the first checkpoint where val_loss <= target_loss."""
    for r in records:
        if "val_loss" in r and r["val_loss"] <= target_loss:
            return r
    return None


def bpb_at_step(records: list[dict], step: int,
                avg_bytes_per_token: float) -> float | None:
    """BPB at the closest recorded step."""
    closest = min(
        (r for r in records if "val_loss" in r),
        key=lambda r: abs(r["step"] - step),
        default=None
    )
    if closest is None:
        return None
    return closest["val_loss"] / np.log(2) / avg_bytes_per_token


def estimate_avg_bpt_from_tokenizer(tokenizer_bin: Path) -> float:
    """Estimate average bytes per token from a saved tokenizer binary."""
    try:
        from tokenizer.clustering_tokenizer import ClusteringTokenizer
        tok = ClusteringTokenizer()
        tok.load(str(tokenizer_bin))
        lengths = [len(t) for t in tok._vocab
                   if t not in (b"<|endoftext|>", b"<|pad|>")]
        return float(np.mean(lengths)) if lengths else 3.0
    except Exception as e:
        print(f"  [warn] could not load tokenizer: {e}, defaulting to 3.0")
        return 3.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bpe_log",       type=Path, required=True)
    parser.add_argument("--cls_log",       type=Path, required=True)
    parser.add_argument("--target_loss",   type=float, default=3.2,
                        help="Val loss target (nats)")
    parser.add_argument("--batch_size",    type=int,   default=32)
    parser.add_argument("--seq_len",       type=int,   default=1024)
    parser.add_argument("--total_batch_size", type=int, default=16384,
                        help="Tokens per gradient step (grad accumulation)")
    parser.add_argument("--bpe_avg_bpt",   type=float, default=3.6,
                        help="BPE avg bytes per token (GPT-2 ≈ 3.6)")
    parser.add_argument("--cls_tokenizer", type=Path,  default=None,
                        help="Path to clustering tokenizer .bin for auto-computing avg_bpt")
    parser.add_argument("--cls_avg_bpt",   type=float, default=None,
                        help="Clustering avg bytes per token (overrides auto-compute)")
    args = parser.parse_args()

    # Auto-compute clustering avg_bpt
    cls_avg_bpt = args.cls_avg_bpt
    if cls_avg_bpt is None:
        if args.cls_tokenizer and args.cls_tokenizer.exists():
            cls_avg_bpt = estimate_avg_bpt_from_tokenizer(args.cls_tokenizer)
            print(f"Clustering avg bytes/token (from tokenizer.bin): {cls_avg_bpt:.3f}")
        else:
            cls_avg_bpt = 2.8  # rough default
            print(f"Clustering avg bytes/token (default): {cls_avg_bpt:.3f}")

    bpe_records = parse_log(args.bpe_log)
    cls_records = parse_log(args.cls_log)

    if not bpe_records:
        print(f"[error] no records parsed from {args.bpe_log}")
        return
    if not cls_records:
        print(f"[error] no records parsed from {args.cls_log}")
        return

    print(f"\n{'═'*65}")
    print(f" CONVERGENCE COMPARISON  (target val loss = {args.target_loss})")
    print(f"{'═'*65}")

    # ── Per-tokenizer analysis ──────────────────────────────
    results = {}
    for name, records, avg_bpt in [
        ("BPE",        bpe_records, args.bpe_avg_bpt),
        ("Clustering", cls_records, cls_avg_bpt),
    ]:
        target_rec = find_step_for_target(records, args.target_loss)
        final_rec  = max((r for r in records if "val_loss" in r),
                         key=lambda r: r["step"], default=None)

        print(f"\n── {name}  (avg bytes/token = {avg_bpt:.2f}) ──")

        if target_rec:
            n_tok = tokens_consumed(
                target_rec["step"], args.batch_size,
                args.seq_len, args.total_batch_size,
            )
            n_bytes = bytes_consumed(n_tok, avg_bpt)
            bpb     = target_rec["val_loss"] / np.log(2) / avg_bpt
            print(f"  Reached target loss {args.target_loss} at:")
            print(f"    step          = {target_rec['step']:,}")
            print(f"    val_loss      = {target_rec['val_loss']:.4f} nats")
            print(f"    tokens used   = {n_tok/1e6:.1f}M")
            print(f"    bytes used    = {n_bytes/1e9:.2f}GB")
            print(f"    BPB           = {bpb:.4f}")
            results[name] = {"step": target_rec["step"], "tokens": n_tok,
                             "bytes": n_bytes, "bpb": bpb,
                             "val_loss": target_rec["val_loss"]}
        else:
            if final_rec:
                n_tok  = tokens_consumed(
                    final_rec["step"], args.batch_size,
                    args.seq_len, args.total_batch_size,
                )
                n_bytes = bytes_consumed(n_tok, avg_bpt)
                bpb     = final_rec["val_loss"] / np.log(2) / avg_bpt
                print(f"  Did NOT reach target (best val_loss = {final_rec['val_loss']:.4f})")
                print(f"    final step    = {final_rec['step']:,}")
                print(f"    tokens used   = {n_tok/1e6:.1f}M")
                print(f"    bytes used    = {n_bytes/1e9:.2f}GB")
                print(f"    final BPB     = {bpb:.4f}")
                results[name] = {"step": None, "tokens": n_tok,
                                 "bytes": n_bytes, "bpb": bpb,
                                 "val_loss": final_rec["val_loss"]}
            else:
                print(f"  No val loss records found")

    # ── Head-to-head comparison ─────────────────────────────
    if "BPE" in results and "Clustering" in results:
        b = results["BPE"]
        c = results["Clustering"]

        print(f"\n{'─'*65}")
        print(" HEAD-TO-HEAD")
        print(f"{'─'*65}")

        # Token efficiency
        if b["step"] and c["step"]:
            tok_ratio = b["tokens"] / c["tokens"]
            byt_ratio = b["bytes"]  / c["bytes"]
            print(f"\n  To reach val_loss <= {args.target_loss}:")
            print(f"  {'metric':<30} {'BPE':>12} {'Clustering':>12} {'ratio':>8}")
            print(f"  {'-'*62}")
            print(f"  {'steps':<30} {b['step']:>12,} {c['step']:>12,} "
                  f"  {b['step']/c['step']:>6.2f}x")
            print(f"  {'tokens consumed (M)':<30} {b['tokens']/1e6:>12.1f} "
                  f"{c['tokens']/1e6:>12.1f}  {tok_ratio:>6.2f}x")
            print(f"  {'bytes consumed (GB)':<30} {b['bytes']/1e9:>12.2f} "
                  f"{c['bytes']/1e9:>12.2f}  {byt_ratio:>6.2f}x")
            print(f"  {'BPB at target':<30} {b['bpb']:>12.4f} {c['bpb']:>12.4f}")

            winner = "Clustering" if c["bytes"] < b["bytes"] else "BPE"
            savings = abs(b["bytes"] - c["bytes"]) / max(b["bytes"], c["bytes"]) * 100
            print(f"\n  WINNER: {winner} used {savings:.1f}% fewer bytes to reach the target")

        # Equal-compute BPB comparison (same token budget)
        # Find the minimum max step across both logs
        bpe_max = max((r["step"] for r in bpe_records if "val_loss" in r),
                      default=0)
        cls_max = max((r["step"] for r in cls_records if "val_loss" in r),
                      default=0)
        common_step = min(bpe_max, cls_max)

        bpe_bpb_eq = bpb_at_step(bpe_records, common_step, args.bpe_avg_bpt)
        cls_bpb_eq = bpb_at_step(cls_records, common_step, cls_avg_bpt)

        if bpe_bpb_eq and cls_bpb_eq:
            print(f"\n  Equal-token-compute comparison (step ≈ {common_step:,}):")
            print(f"    BPE BPB        = {bpe_bpb_eq:.4f}")
            print(f"    Clustering BPB = {cls_bpb_eq:.4f}")
            diff = cls_bpb_eq - bpe_bpb_eq
            print(f"    Δ BPB          = {diff:+.4f}  "
                  f"({'Clustering wins' if diff < 0 else 'BPE wins'})")

    # ── Visualize ──────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for ax_idx, (metric, ylabel, transform) in enumerate([
            ("val_loss", "Val Loss (nats, ↓ better)",
             lambda v, bpt: v),
            ("bpb",      "BPB (↓ better)",
             lambda v, bpt: v / np.log(2) / bpt),
        ]):
            for name, records, avg_bpt, color in [
                ("BPE",        bpe_records, args.bpe_avg_bpt, "steelblue"),
                ("Clustering", cls_records, cls_avg_bpt,       "darkorange"),
            ]:
                val_recs = [(r["step"], r["val_loss"])
                            for r in records if "val_loss" in r]
                if not val_recs:
                    continue
                steps, losses = zip(*val_recs)
                y = [transform(l, avg_bpt) for l in losses]
                axes[ax_idx].plot(steps, y, "o-", label=name, color=color,
                                  markersize=4)

            if metric == "val_loss":
                axes[ax_idx].axhline(y=args.target_loss, color="red",
                                     linestyle="--", alpha=0.5,
                                     label=f"target = {args.target_loss}")

            axes[ax_idx].set_xlabel("Step")
            axes[ax_idx].set_ylabel(ylabel)
            axes[ax_idx].set_title(
                "Val Loss Convergence" if metric == "val_loss"
                else "BPB Convergence (cross-tokenizer fair)"
            )
            axes[ax_idx].legend()
            axes[ax_idx].grid(alpha=0.3)

        plt.tight_layout()
        out_path = Path("experiments/out/convergence/convergence.png")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"\n  Plot saved to {out_path}")
        plt.show()

    except ImportError:
        print("  (matplotlib not installed, skipping plot)")


if __name__ == "__main__":
    main()
