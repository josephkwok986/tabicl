#!/usr/bin/env bash
# 极小数据生成，自测用（验证目录结构/可读性）
# 用法：
#   bash scripts/make_prior_tabicl_official_mini.sh [OUT_DIR]
# 例子：
#   bash scripts/make_prior_tabicl_official_mini.sh ./_mini_prior_official

set -euo pipefail

OUT_DIR="${1:-./_mini_prior_official}"

# 缩小到几百/千级别，几秒能跑完
MIN_FEATURES=16
MAX_FEATURES=32
MAX_CLASSES=5

MIN_SEQ_LEN=512
MAX_SEQ_LEN=1024
LOG_SEQ_LEN=1

MIN_TRAIN_SIZE=20
MAX_TRAIN_SIZE=50

BATCH_SIZE=64
NUM_BATCHES=2
PRIOR_TYPE="mix_scm"

# CPU 跑就足够
DEVICE="cpu"
N_JOBS="$(getconf _NPROCESSORS_ONLN 2>/dev/null || echo 4)"
NUM_THREADS_PER_GENERATE=2

NP_SEED=123
TORCH_SEED=123

mkdir -p "$OUT_DIR"

python -m tabicl.prior.genload \
  --save_dir "$OUT_DIR" \
  --np_seed "$NP_SEED" \
  --torch_seed "$TORCH_SEED" \
  --num_batches "$NUM_BATCHES" \
  --batch_size "$BATCH_SIZE" \
  --min_features "$MIN_FEATURES" \
  --max_features "$MAX_FEATURES" \
  --max_classes "$MAX_CLASSES" \
  --min_seq_len "$MIN_SEQ_LEN" \
  --max_seq_len "$MAX_SEQ_LEN" \
  --log_seq_len "$LOG_SEQ_LEN" \
  --min_train_size "$MIN_TRAIN_SIZE" \
  --max_train_size "$MAX_TRAIN_SIZE" \
  --prior_type "$PRIOR_TYPE" \
  --n_jobs "$N_JOBS" \
  --num_threads_per_generate "$NUM_THREADS_PER_GENERATE" \
  --device "$DEVICE"

echo "[OK] Mini prior saved to: $OUT_DIR"
