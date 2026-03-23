#!/usr/bin/env bash
# ============================================================
#  收敛速度对比实验
#
#  目标：在相同 token 预算下，比较 BPE 和聚类分词器
#        哪个能更快达到 val loss <= TARGET_LOSS
#
#  设计思路：
#    - 两个模型用完全相同的超参数（架构 d12，lr，schedule）
#    - 每 CKPT_EVERY 步记录一次 val loss 和累计消耗 token 数
#    - 谁先达到 TARGET_LOSS，消耗 token 更少 → 分词效率更好
#
#  公平性：
#    - 步数相同，但"消耗 byte 数"可能不同（因为压缩率不同）
#    - 需同时记录 token 消耗 和 byte 消耗（BPB 才是公平指标）
#
#  用法：
#    ./experiments/convergence_test.sh [smoke]
# ============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SMOKE=${1:-""}
TARGET_LOSS=3.2     # 目标 val loss（nats）

if [[ "$SMOKE" == "smoke" ]]; then
  BATCH_SIZE=8
  SEQ_LEN=256
  TOTAL_BATCH=4096
  MAX_ITERS=500       # 封顶步数（避免无限跑）
  CKPT_EVERY=50
  echo "=== SMOKE: batch=$BATCH_SIZE seq=$SEQ_LEN max_iters=$MAX_ITERS ==="
else
  BATCH_SIZE=32
  SEQ_LEN=1024
  TOTAL_BATCH=16384
  MAX_ITERS=20000
  CKPT_EVERY=100
  echo "=== CONVERGENCE TEST: target_loss=$TARGET_LOSS max_iters=$MAX_ITERS ==="
fi

LR=3e-4
WARMUP=200

# ── 确保数据已准备 ──────────────────────────────────────────
if [ ! -d "data/bpe" ] || [ -z "$(ls data/bpe/train_*.bin 2>/dev/null)" ]; then
  echo "[prepare] BPE data not found, running prepare_bpe.py ..."
  python3 data/prepare_bpe.py ${SMOKE:+--smoke}
fi

if [ ! -d "data/clustering" ] || [ -z "$(ls data/clustering/train_*.bin 2>/dev/null)" ]; then
  echo "[prepare] Clustering data not found, running prepare_clustering.py ..."
  python3 data/prepare_clustering.py ${SMOKE:+--smoke}
fi

# ── 训练函数 ────────────────────────────────────────────────
run_training() {
  local name="$1"
  local data_glob="$2"
  local val_bin="$3"
  local out_dir="experiments/out/convergence/${name}"

  mkdir -p "$out_dir"

  echo ""
  echo "── Training: $name ──"
  python3 llm.c/train_gpt2.py \
    --input_bin           "$data_glob" \
    --input_val_bin       "$val_bin" \
    --output_dir          "$out_dir" \
    --model               d12 \
    --batch_size          $BATCH_SIZE \
    --sequence_length     $SEQ_LEN \
    --total_batch_size    $TOTAL_BATCH \
    --num_iterations      $MAX_ITERS \
    --learning_rate       $LR \
    --warmup_iters        $WARMUP \
    --learning_rate_decay_frac 0.1 \
    --weight_decay        0.1 \
    --grad_clip           1.0 \
    --val_loss_every      $CKPT_EVERY \
    --val_max_steps       20 \
    --tensorcores         1 \
    --overfit_single_batch 0 \
    2>&1 | tee "$out_dir/log.txt"
}

run_training "bpe"        "data/bpe/train_*.bin"        "data/bpe/val_0000.bin"
run_training "clustering" "data/clustering/train_*.bin" "data/clustering/val_0000.bin"

# ── 分析：谁先到达 target loss ───────────────────────────────
echo ""
echo "── Analysis ──"
python3 experiments/analyze_convergence.py \
  --bpe_log        "experiments/out/convergence/bpe/log.txt" \
  --cls_log        "experiments/out/convergence/clustering/log.txt" \
  --target_loss    $TARGET_LOSS \
  --batch_size     $BATCH_SIZE \
  --seq_len        $SEQ_LEN \
  --bpe_avg_bpt    3.6 \
  --cls_tokenizer  "data/clustering/tokenizer.bin"
