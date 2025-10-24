#!/usr/bin/env bash
# 生成 TabICL Stage-2 离线 prior（官方 genload.py 语义对齐）
# 用法：
#   bash scripts/make_prior_tabicl_official_stage2.sh [OUT_DIR] [NUM_BATCHES] [DEVICE] [GPU_IDS] [BATCH_SIZE]
# 例子（推荐CPU生成）：./make_prior_tabicl_official_stage2.sh ./tabicl_prior 80 cpu
# 例子（用3号GPU生成）： ./make_prior_stage3_3gpu_1to5min.sh ./tabicl_prior 30 cuda 3 512

set -euo pipefail

OUT_DIR="${1:-/data/tabicl_prior_stage2_official}"
NUM_BATCHES="${2:-80}"           # 30~120：训练过短就加，过长就减
DEVICE="${3:-cpu}"               # cpu | cuda
GPU_IDS="${4:-0}"                # DEVICE=cuda 时可指定，如 "1"
BATCH_SIZE="${5:-512}"           # Stage-2 对齐：512

# —— 与 Stage-2 对齐的范围 ——（见 train_stage2.sh）
MIN_FEATURES=2
MAX_FEATURES=100
MAX_CLASSES=10

# Stage-2：样本数按对数分布抽样 1K~40K
MIN_SEQ_LEN=1000
MAX_SEQ_LEN=40000
LOG_SEQ_LEN=True

# 训练划分比例（0~1 浮点），官方示例：0.5~0.9
MIN_TRAIN_SIZE=0.5
MAX_TRAIN_SIZE=0.9

# per-GPU 参数对齐官方示例
BATCH_SIZE_PER_GP=2
SEQ_LEN_PER_GP=True

# 并行
if [[ "$DEVICE" == "cuda" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_IDS"
  N_JOBS=1
  NUM_THREADS_PER_GENERATE=1
else
  N_JOBS=-1
  NUM_THREADS_PER_GENERATE=1
fi

NP_SEED=42
TORCH_SEED=42
PRIOR_TYPE="mix_scm"            # {mlp_scm, tree_scm, mix_scm}
REPLAY_SMALL=0

mkdir -p "$OUT_DIR"

python -m tabicl.prior.genload \
  --save_dir "$OUT_DIR" \
  --np_seed "$NP_SEED" \
  --torch_seed "$TORCH_SEED" \
  --num_batches "$NUM_BATCHES" \
  --batch_size "$BATCH_SIZE" \
  --batch_size_per_gp "$BATCH_SIZE_PER_GP" \
  --min_features "$MIN_FEATURES" \
  --max_features "$MAX_FEATURES" \
  --max_classes "$MAX_CLASSES" \
  --min_seq_len "$MIN_SEQ_LEN" \
  --max_seq_len "$MAX_SEQ_LEN" \
  --log_seq_len "$LOG_SEQ_LEN" \
  --seq_len_per_gp "$SEQ_LEN_PER_GP" \
  --min_train_size "$MIN_TRAIN_SIZE" \
  --max_train_size "$MAX_TRAIN_SIZE" \
  --replay_small "$REPLAY_SMALL" \
  --prior_type "$PRIOR_TYPE" \
  --n_jobs "$N_JOBS" \
  --num_threads_per_generate "$NUM_THREADS_PER_GENERATE" \
  --device "$DEVICE"

echo "[OK] Prior saved to: $OUT_DIR"
