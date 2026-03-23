# lp_tokenizer: Clustering Tokenizer Experiment

验证假设：用 **MinHash + LSH** 近似聚类构建分词器，在保持 O(n) 编码速度和理论近似比保证的同时，能否在语言模型生成质量（Bits-per-Byte）上与 BPE 持平或更优？

底座：[karpathy/llm.c](https://github.com/karpathy/llm.c)（GPT-2 轻量 C/CUDA 实现）

---

## 相关工作

| 论文 | arXiv | 关键发现 |
|------|-------|---------|
| BlockBPE (2025) | 2507.11941 | 并行 GPU BPE，2x 吞吐，0.998 相似度；仍是 BPE 范式 |
| GPUTOK (2026) | 2603.02597 | CUDA BPE，7.6x faster，cuCollections GPU hash map |
| Speech k-means (2025) | 2509.05359 | embedding 空间聚类用于语音 LM，非 byte 级别文本 |

**本工作的新颖性**：首次将 LSH/MinHash 直接应用于 byte 级别文本词表构建，并与 BPE 在 BPB 上对比。

---

## 算法：MinHash + LSH 聚类

```
语料库
  ↓ 滑动窗口 (n=2..8)
所有 byte n-gram + Count-Min Sketch 频率估计
  ↓ top-K 高频候选 (K = vocab_size × 10)
每个候选 → 128-dim MinHash 签名（(pos, byte) 特征集合）
  ↓ LSH: b=16 bands, r=8 rows
同桶候选合并 → 选最高频者为代表 token
  ↓
词表 = {代表 tokens} + {256 byte fallback} + {EOT}
```

**近似比保证**（LSH S 曲线，b=16, r=8）：
- Jaccard ≥ 0.8 的 token 对：碰撞概率 ≥ 99%（false-negative < 1%）
- Jaccard ≤ 0.3 的 token 对：碰撞概率 ≤ 0.5%（false-positive < 0.5%）
- 推理时间复杂度：O(n) vs BPE O(n log n)

---

## 项目结构

```
lp_tokenizer/
├── llm.c/                      # karpathy/llm.c (git subtree)
├── tokenizer/
│   ├── clustering_tokenizer.py  # 核心：GPU MinHash+LSH 聚类分词器
│   ├── bpe_baseline.py         # tiktoken gpt2 包装
│   └── tokenizer_utils.py      # write_datafile(), BPB, metrics
├── data/
│   ├── download.py             # 下载 TinyStories / TinyShakespeare
│   ├── prepare_bpe.py          # BPE 编码 → data/bpe/*.bin
│   └── prepare_clustering.py   # 聚类编码 → data/clustering/*.bin
├── eval/
│   ├── tokenizer_metrics.py    # 压缩率、fertility、速度对比
│   └── generation_metrics.py   # BPB、perplexity、生成样本
├── experiments/
│   └── run_experiment.sh       # 端到端实验脚本
└── Makefile
```

---

## 快速开始

### 环境安装

```bash
pip install -r requirements.txt
```

### 烟雾测试（~5分钟，4090）

```bash
make smoke
# 或
./experiments/run_experiment.sh smoke
```

### 完整实验（~2-4小时，4090）

```bash
# 1. 下载数据
make data

# 2. 分步运行（推荐）
python data/prepare_bpe.py
python data/prepare_clustering.py --train_mb 500 --vocab_size 50257

# 3. 训练（两次，分别用 BPE 和聚类分词器的数据）
python llm.c/train_gpt2.py \
    --input_bin "data/bpe/train_*.bin" \
    --input_val_bin "data/bpe/val_0000.bin" \
    --output_dir "experiments/out/bpe" \
    --model d12 --batch_size 32 --sequence_length 1024 \
    --num_iterations 10000 --tensorcores 1 \
    2>&1 | tee experiments/out/bpe/log.txt

python llm.c/train_gpt2.py \
    --input_bin "data/clustering/train_*.bin" \
    --input_val_bin "data/clustering/val_0000.bin" \
    --output_dir "experiments/out/clustering" \
    --model d12 --batch_size 32 --sequence_length 1024 \
    --num_iterations 10000 --tensorcores 1 \
    2>&1 | tee experiments/out/clustering/log.txt

# 4. 评估
make eval-tokenizer
make eval-generation
```

---

## 评估指标

| 指标 | 说明 | 公式 |
|------|------|------|
| **BPB** | 核心对比指标，跨分词器可比 | `val_loss / ln(2) / avg_bytes_per_token` |
| compression_ratio | 平均 bytes/token | 越高越好 |
| fertility | 平均 tokens/word | 越低越好 |
| encode_speed | 编码吞吐 MB/s | 越高越好 |

---

## Hardware

- GPU: RTX 4090 48GB
- 聚类分词器训练全程 GPU（PyTorch）
- 模型训练：`train_gpt2.py --tensorcores 1`
