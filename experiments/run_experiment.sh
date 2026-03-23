#!/usr/bin/env bash
# ============================================================
#  End-to-end experiment: BPE vs Clustering tokenizer
#  on TinyStories with llm.c GPT-2 training
#
#  Usage:
#    ./experiments/run_experiment.sh [smoke]
#
#  With "smoke":
#    - 5MB corpus for tokenizer training
#    - 1 data shard
#    - 100 training iterations
#    - Quick sanity check (~5 minutes on 4090)
#
#  Full run (no args):
#    - 500MB corpus for tokenizer training
#    - All data shards (~2GB)
#    - 10,000 training iterations
#    - ~2-4 hours on 4090
#
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

SMOKE=${1:-""}

if [[ "$SMOKE" == "smoke" ]]; then
  TRAIN_MB=5
  VOCAB_SIZE=8192
  NUM_ITERS=100
  BATCH_SIZE=8
  SEQ_LEN=128
  echo "=== SMOKE MODE: quick sanity check ==="
else
  TRAIN_MB=500
  VOCAB_SIZE=50257
  NUM_ITERS=10000
  BATCH_SIZE=32
  SEQ_LEN=1024
  echo "=== FULL EXPERIMENT ==="
fi

TOTAL_BATCH=16384   # gradient accumulation target ~16K tokens
VAL_EVERY=250
LR=3e-4
WARMUP=100
DECAY_FRAC=0.1

# ── 0. Check Python & CUDA ──────────────────────────────────
echo ""
echo "── Environment Check ──"
python3 -c "import torch; print('PyTorch:', torch.__version__, '| CUDA:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# ── 1. Download data ────────────────────────────────────────
echo ""
echo "── Step 1: Download TinyStories ──"
python3 data/download.py --dataset tinystories

# ── 2. Prepare BPE data ─────────────────────────────────────
echo ""
echo "── Step 2: Prepare BPE data ──"
if [[ "$SMOKE" == "smoke" ]]; then
  python3 data/prepare_bpe.py --smoke
else
  python3 data/prepare_bpe.py
fi

# ── 3. Train clustering tokenizer & prepare data ────────────
echo ""
echo "── Step 3: Train clustering tokenizer & prepare data ──"
if [[ "$SMOKE" == "smoke" ]]; then
  python3 data/prepare_clustering.py \
    --train_mb $TRAIN_MB \
    --vocab_size $VOCAB_SIZE \
    --smoke
else
  python3 data/prepare_clustering.py \
    --train_mb $TRAIN_MB \
    --vocab_size $VOCAB_SIZE
fi

# ── 4. Train: BPE model ─────────────────────────────────────
echo ""
echo "── Step 4: Train BPE model ──"
mkdir -p experiments/out/bpe
python3 llm.c/train_gpt2.py \
  --input_bin          "data/bpe/train_*.bin" \
  --input_val_bin      "data/bpe/val_0000.bin" \
  --output_dir         "experiments/out/bpe" \
  --model              d12 \
  --batch_size         $BATCH_SIZE \
  --sequence_length    $SEQ_LEN \
  --total_batch_size   $TOTAL_BATCH \
  --num_iterations     $NUM_ITERS \
  --learning_rate      $LR \
  --warmup_iters       $WARMUP \
  --learning_rate_decay_frac $DECAY_FRAC \
  --weight_decay       0.1 \
  --grad_clip          1.0 \
  --val_loss_every     $VAL_EVERY \
  --val_max_steps      20 \
  --tensorcores        1 \
  --overfit_single_batch 0 \
  2>&1 | tee experiments/out/bpe/log.txt

# ── 5. Train: Clustering model ──────────────────────────────
echo ""
echo "── Step 5: Train Clustering model ──"
mkdir -p experiments/out/clustering
python3 llm.c/train_gpt2.py \
  --input_bin          "data/clustering/train_*.bin" \
  --input_val_bin      "data/clustering/val_0000.bin" \
  --output_dir         "experiments/out/clustering" \
  --model              d12 \
  --batch_size         $BATCH_SIZE \
  --sequence_length    $SEQ_LEN \
  --total_batch_size   $TOTAL_BATCH \
  --num_iterations     $NUM_ITERS \
  --learning_rate      $LR \
  --warmup_iters       $WARMUP \
  --learning_rate_decay_frac $DECAY_FRAC \
  --weight_decay       0.1 \
  --grad_clip          1.0 \
  --val_loss_every     $VAL_EVERY \
  --val_max_steps      20 \
  --tensorcores        1 \
  --overfit_single_batch 0 \
  2>&1 | tee experiments/out/clustering/log.txt

# ── 6. Evaluate tokenizer metrics ───────────────────────────
echo ""
echo "── Step 6: Tokenizer metrics ──"
python3 eval/tokenizer_metrics.py \
  --text_file              data/raw/tinystories/val_0000.txt \
  --clustering_tokenizer   data/clustering/tokenizer.bin \
  --out_json               experiments/out/tokenizer_metrics.json

# ── 7. Compare generation quality (BPB) ─────────────────────
echo ""
echo "── Step 7: Generation quality (BPB) ──"
python3 eval/generation_metrics.py \
  --mode       logs \
  --bpe_log    experiments/out/bpe/log.txt \
  --cls_log    experiments/out/clustering/log.txt \
  --bpe_val_bin  data/bpe/val_0000.bin \
  --cls_val_bin  data/clustering/val_0000.bin \
  --bpe_avg_bpt  3.6

echo ""
echo "════════════════════════════════════════"
echo "  Experiment complete!"
echo "  Results in experiments/out/"
echo "════════════════════════════════════════"
