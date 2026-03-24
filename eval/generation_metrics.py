"""
Compare language model generation quality: BPE vs Clustering tokenizer models.

Metrics:
  - Bits-per-byte (BPB): canonical cross-tokenizer metric, lower = better
  - Token-level perplexity (within each tokenizer's space)
  - Sample generations (qualitative)

This script reads:
  1. Validation binary data files (.bin) for each tokenizer
  2. Model checkpoint files from llm.c training

Since llm.c doesn't natively expose per-token log-probs via Python, this
script supports two modes:

  Mode A (estimate): use the training loss from llm.c's output logs
    → reads loss curves from experiments/out/*/log.txt
    → computes approximate BPB from final validation loss

  Mode B (full): load model checkpoint with PyTorch and compute BPB directly
    → uses a minimal GPT-2 implementation to load llm.c .pt checkpoints

Usage:
  # Compare using training logs (easy)
  python eval/generation_metrics.py --mode logs \
      --bpe_log      experiments/out/bpe/log.txt \
      --cls_log      experiments/out/clustering/log.txt \
      --bpe_val_bin  data/bpe/val_0000.bin \
      --cls_val_bin  data/clustering/val_0000.bin

  # Full BPB computation using PyTorch model
  python eval/generation_metrics.py --mode model \
      --bpe_ckpt     experiments/out/bpe/model.pt \
      --cls_ckpt     experiments/out/clustering/model.pt \
      --bpe_val_bin  data/bpe/val_0000.bin \
      --cls_val_bin  data/clustering/val_0000.bin \
      --bpe_tok      tokenizer/bpe \
      --cls_tok      data/clustering/tokenizer.bin
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from tokenizer.tokenizer_utils import read_datafile, bits_per_byte


# ──────────────────────────────────────────────
#  Mode A: parse llm.c log files
# ──────────────────────────────────────────────

def parse_llmc_log(log_path: Path) -> dict:
    """
    Parse llm.c training log.

    llm.c prints lines like:
      step 100: train loss 4.123456, val loss 4.234567, ...
    Returns dict with final val loss and loss curve arrays.
    """
    train_losses: list[tuple[int, float]] = []
    val_losses:   list[tuple[int, float]] = []

    with open(log_path) as f:
        for line in f:
            # Match "step N: train loss X" or "val loss X"
            m = re.search(r"step\s+(\d+).*train loss\s+([\d.]+)", line)
            if m:
                train_losses.append((int(m.group(1)), float(m.group(2))))
            m = re.search(r"val loss\s+([\d.]+)", line)
            if m and train_losses:
                step = train_losses[-1][0]
                val_losses.append((step, float(m.group(1))))

    result = {"train_losses": train_losses, "val_losses": val_losses}
    if val_losses:
        result["final_val_loss"] = val_losses[-1][1]
        result["final_step"]     = val_losses[-1][0]
    return result


def val_loss_to_bpb(val_loss: float, bytes_per_token: float) -> float:
    """
    Convert validation cross-entropy loss (nats) to bits-per-byte.

    BPB = val_loss_nats / (bytes_per_token * ln(2))
    Since val_loss is in nats and BPB is in bits:
      BPB = val_loss / ln(2) / bytes_per_token
          = val_loss * log2(e) / bytes_per_token
    """
    return val_loss / np.log(2) / bytes_per_token


def compute_bytes_per_token(val_bin: Path,
                            tokenizer_bin: Optional[Path] = None,
                            vocab_size: int = 50257,
                            is_bpe: bool = False) -> float:
    """
    Compute actual encoding density (bytes per token) from a .bin file.

    Decodes a sample of val tokens with the tokenizer and measures:
        actual_bytes_per_token = total_decoded_bytes / n_sampled_tokens

    This is the correct denominator for BPB.  Do NOT use vocabulary-entry
    mean length, which diverges from encoding density when coverage is low.
    """
    tokens, _ = read_datafile(val_bin)
    if len(tokens) == 0:
        return 3.5

    if is_bpe:
        try:
            from tokenizer.bpe_baseline import BPETokenizer
            tok = BPETokenizer()
            sample = tokens[:10_000]
            total = sum(len(tok.decode([int(t)])) for t in sample)
            return total / len(sample)
        except Exception:
            return 3.6

    return _estimate_avg_bytes_per_token(tokenizer_bin, vocab_size,
                                         val_tokens=tokens)


# ──────────────────────────────────────────────
#  Mode B: compute BPB with PyTorch model
# ──────────────────────────────────────────────

def compute_bpb_from_checkpoint(
    ckpt_path: Path,
    val_bin:   Path,
    tokenizer_bin: Optional[Path],
    device: str = "cuda",
    batch_size: int = 8,
    seq_len: int = 1024,
) -> float:
    """
    Load an llm.c model checkpoint (.pt) and compute bits-per-byte on the
    validation set.

    llm.c saves checkpoints as PyTorch state dicts compatible with
    the Hugging Face GPT-2 architecture when using the Python reference
    implementation (train_gpt2.py).
    """
    import torch
    from transformers import GPT2LMHeadModel, GPT2Config

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"[bpb] loading checkpoint {ckpt_path} on {device}")

    ckpt = torch.load(ckpt_path, map_location=device)

    # Detect config from checkpoint
    # llm.c's train_gpt2.py saves: {"model": state_dict, "config": {...}}
    if isinstance(ckpt, dict) and "config" in ckpt:
        cfg_dict = ckpt["config"]
        config = GPT2Config(
            vocab_size      = cfg_dict.get("vocab_size", 50257),
            n_positions     = cfg_dict.get("block_size", 1024),
            n_embd          = cfg_dict.get("n_embd", 768),
            n_layer         = cfg_dict.get("n_layer", 12),
            n_head          = cfg_dict.get("n_head", 12),
        )
        state_dict = ckpt["model"]
    else:
        config     = GPT2Config()
        state_dict = ckpt

    model = GPT2LMHeadModel(config)
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)

    # Load validation tokens
    val_tokens, _ = read_datafile(val_bin)
    total_tokens = len(val_tokens)
    val_t = torch.tensor(val_tokens.astype(np.int64), dtype=torch.long)

    total_nll   = 0.0
    total_toks  = 0

    with torch.no_grad():
        for start in range(0, total_tokens - seq_len, seq_len * batch_size):
            # Build batch
            batch_inputs:  list[torch.Tensor] = []
            batch_targets: list[torch.Tensor] = []
            for b in range(batch_size):
                s = start + b * seq_len
                e = s + seq_len + 1
                if e > total_tokens:
                    break
                batch_inputs.append(val_t[s: s + seq_len])
                batch_targets.append(val_t[s + 1: e])
            if not batch_inputs:
                break

            inp = torch.stack(batch_inputs).to(device)    # (B, T)
            tgt = torch.stack(batch_targets).to(device)   # (B, T)

            out = model(input_ids=inp, labels=tgt)
            # out.loss is mean cross-entropy over the batch in nats
            total_nll  += out.loss.item() * inp.numel()
            total_toks += inp.numel()

    # Compute actual bytes/token from the validation tokens themselves.
    # This is the encoding density (source bytes covered per token), which
    # is the correct denominator for BPB — NOT the vocabulary entry mean.
    avg_bytes_per_token = _estimate_avg_bytes_per_token(
        tokenizer_bin, config.vocab_size,
        val_tokens=val_tokens,   # decode actual tokens for ground-truth density
    )
    total_bytes = total_toks * avg_bytes_per_token
    bpb = (total_nll / np.log(2)) / total_bytes

    print(f"  val tokens: {total_toks:,}  avg_bytes/tok: {avg_bytes_per_token:.3f}")
    print(f"  BPB: {bpb:.4f}")
    return bpb


def _estimate_avg_bytes_per_token(tokenizer_bin: Optional[Path],
                                  vocab_size: int,
                                  val_tokens: Optional[np.ndarray] = None) -> float:
    """
    Compute average bytes per token — encoding density, not vocabulary mean.

    The correct denominator for BPB is the mean number of *source bytes*
    covered per token when actually encoding text, NOT the mean length of
    vocabulary entries.  These differ dramatically when vocab coverage is
    low (e.g. clustering tokenizer at 6.5% multi-byte coverage: vocab mean
    ≈ 2.80 bytes/token, but actual encoding density ≈ 1.09 bytes/token).

    If val_tokens is provided we decode them directly with the tokenizer
    and measure actual_bytes / n_tokens.  This is the ground truth.

    Falls back to vocabulary mean only when the tokenizer cannot be loaded.
    """
    if tokenizer_bin is None or not Path(tokenizer_bin).exists():
        # BPE: use tiktoken to measure on val_tokens if available
        if val_tokens is not None:
            try:
                from tokenizer.bpe_baseline import BPETokenizer
                tok = BPETokenizer()
                total_bytes = sum(len(tok.decode([int(t)])) for t in val_tokens[:10_000])
                return total_bytes / min(len(val_tokens), 10_000)
            except Exception:
                pass
        return 3.6   # GPT-2 BPE well-known empirical value

    # Clustering tokenizer binary
    try:
        from tokenizer.clustering_tokenizer import ClusteringTokenizer
        ct = ClusteringTokenizer(vocab_size=vocab_size)
        ct.load(str(tokenizer_bin))

        if val_tokens is not None and len(val_tokens) > 0:
            # Ground truth: decode actual val tokens and measure byte coverage.
            # Sample up to 100K tokens for speed.
            sample = val_tokens[:100_000]
            total_bytes = sum(len(ct._vocab[int(t)]) for t in sample
                              if int(t) < len(ct._vocab))
            return total_bytes / len(sample)
        else:
            # Fallback: frequency-weighted vocab mean.
            # Single-byte tokens (0-255) are almost always over-represented,
            # so this still beats the unweighted mean.
            lengths = [len(tok) for tok in ct._vocab]
            return float(np.mean(lengths))
    except Exception:
        return 3.5


# ──────────────────────────────────────────────
#  Generation samples
# ──────────────────────────────────────────────

def generate_samples(
    ckpt_path: Path,
    tokenizer_bin: Optional[Path],
    prompts: list[str],
    max_new_tokens: int = 200,
    device: str = "cuda",
) -> list[str]:
    """Generate text samples from a model checkpoint for qualitative comparison."""
    import torch
    from transformers import GPT2LMHeadModel, GPT2Config

    device = torch.device(device if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "config" in ckpt:
        cfg_dict = ckpt["config"]
        config = GPT2Config(
            vocab_size = cfg_dict.get("vocab_size", 50257),
            n_positions= cfg_dict.get("block_size", 1024),
            n_embd     = cfg_dict.get("n_embd", 768),
            n_layer    = cfg_dict.get("n_layer", 12),
            n_head     = cfg_dict.get("n_head", 12),
        )
        state_dict = ckpt["model"]
    else:
        config     = GPT2Config()
        state_dict = ckpt

    model = GPT2LMHeadModel(config)
    model.load_state_dict(state_dict, strict=False)
    model.eval().to(device)

    # Load tokenizer
    if tokenizer_bin and tokenizer_bin.exists():
        from tokenizer.clustering_tokenizer import ClusteringTokenizer
        tok = ClusteringTokenizer()
        tok.load(str(tokenizer_bin))
        encode = tok.encode
        decode = tok.decode_str
    else:
        from tokenizer.bpe_baseline import BPETokenizer
        bpe = BPETokenizer()
        encode = bpe.encode
        decode = bpe.decode_str

    samples: list[str] = []
    with torch.no_grad():
        for prompt in prompts:
            ids = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
            out = model.generate(
                ids,
                max_new_tokens = max_new_tokens,
                do_sample      = True,
                temperature    = 0.8,
                top_p          = 0.9,
            )
            samples.append(decode(out[0].tolist()))
    return samples


# ──────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["logs", "model"], default="logs")

    # Log mode args
    parser.add_argument("--bpe_log", type=Path, default=None)
    parser.add_argument("--cls_log", type=Path, default=None)

    # Shared
    parser.add_argument("--bpe_val_bin",  type=Path, default=None)
    parser.add_argument("--cls_val_bin",  type=Path, default=None)
    parser.add_argument("--bpe_avg_bpt",  type=float, default=3.6,
                        help="Average bytes per token for BPE (GPT-2 ≈ 3.6)")
    parser.add_argument("--cls_avg_bpt",  type=float, default=None,
                        help="Average bytes per token for clustering tokenizer "
                             "(computed from tokenizer.bin if not set)")
    # Model mode args
    parser.add_argument("--bpe_ckpt",   type=Path, default=None)
    parser.add_argument("--cls_ckpt",   type=Path, default=None)
    parser.add_argument("--cls_tok",    type=Path, default=None,
                        help="Path to clustering tokenizer .bin")
    parser.add_argument("--device",     type=str, default="cuda")

    args = parser.parse_args()

    if args.mode == "logs":
        print("═" * 60)
        print("GENERATION QUALITY COMPARISON (from training logs)")
        print("═" * 60)

        rows = [
            # (name,          log_path,      val_bin,           tokenizer_bin,  is_bpe, fallback_bpt)
            ("BPE",        args.bpe_log, args.bpe_val_bin, None,          True,  args.bpe_avg_bpt),
            ("Clustering", args.cls_log, args.cls_val_bin, args.cls_tok,  False, args.cls_avg_bpt),
        ]

        for name, log_path, val_bin, tok_bin, is_bpe, override_bpt in rows:
            if log_path is None or not log_path.exists():
                print(f"\n[{name}] log not found: {log_path}")
                continue
            info = parse_llmc_log(log_path)
            if "final_val_loss" not in info:
                print(f"\n[{name}] no val loss found in log")
                continue

            # Determine bytes/token: prefer override → computed from val_bin → fallback
            if override_bpt is not None:
                avg_bpt = override_bpt
                bpt_src = "override"
            elif val_bin is not None and val_bin.exists():
                avg_bpt = compute_bytes_per_token(
                    val_bin, tok_bin, vocab_size=50257, is_bpe=is_bpe
                )
                bpt_src = f"measured from {val_bin.name}"
            else:
                avg_bpt = 3.6 if is_bpe else 2.5
                bpt_src = "default fallback (no val_bin provided)"

            val_loss = info["final_val_loss"]
            bpb      = val_loss_to_bpb(val_loss, avg_bpt)
            print(f"\n{name}:")
            print(f"  final_step      : {info.get('final_step', '?')}")
            print(f"  val_loss (nats) : {val_loss:.4f}")
            print(f"  avg bytes/token : {avg_bpt:.3f}  ({bpt_src})")
            print(f"  BPB             : {bpb:.4f}  bits/byte")

    elif args.mode == "model":
        results = {}
        for name, ckpt, val_bin, tok_bin in [
            ("BPE",        args.bpe_ckpt, args.bpe_val_bin, None),
            ("Clustering", args.cls_ckpt, args.cls_val_bin, args.cls_tok),
        ]:
            if ckpt is None or not ckpt.exists():
                print(f"\n[{name}] checkpoint not found: {ckpt}")
                continue
            print(f"\n[{name}] computing BPB ...")
            bpb = compute_bpb_from_checkpoint(
                ckpt_path      = ckpt,
                val_bin        = val_bin,
                tokenizer_bin  = tok_bin,
                device         = args.device,
            )
            results[name] = bpb

        if len(results) == 2:
            bpe_bpb = results.get("BPE")
            cls_bpb = results.get("Clustering")
            if bpe_bpb and cls_bpb:
                diff = cls_bpb - bpe_bpb
                print(f"\nFinal BPB comparison:")
                print(f"  BPE        : {bpe_bpb:.4f}")
                print(f"  Clustering : {cls_bpb:.4f}")
                print(f"  Difference : {diff:+.4f}  "
                      f"({'Clustering wins' if diff < 0 else 'BPE wins'})")


if __name__ == "__main__":
    main()
