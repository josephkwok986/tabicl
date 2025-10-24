#!/usr/bin/env bash
# 串联：启动监控 -> 运行推理 -> 推理结束后继续监控一段时间 -> 停止监控
# 依赖：同目录下已存在 run_infer.sh、monitor_start.sh、monitor_stop.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------- 默认参数（都可被命令行覆盖） ----------
DATA_DIR_DEFAULT="data/synth"
TARGET_DEFAULT="label"
GPU_IDS_DEFAULT="3"
CKPT_PATH_DEFAULT="/home/amax/gjj_tabicl/checkpoint/tabicl-classifier-v1.1-0506.ckpt"
N_ESTIMATORS_DEFAULT=32
BATCH_SIZE_DEFAULT=8
AMP_DEFAULT=1
PROG_CHUNK_DEFAULT=20000
OUT_DIR_DEFAULT="runs/tabicl_infer_$(date +%Y%m%d_%H%M%S)"

# 监控相关
MONITOR_AFTER_SECS_DEFAULT=300          # 推理结束后继续监控的秒数（建议 3~5 分钟）
MONITOR_GPU_IDS_DEFAULT=""              # 不指定则跟随 --gpu_ids
LOGROOT_DEFAULT="logs"                  # monitor_start.sh 的日志根目录

# ---------- 解析参数 ----------
DATA_DIR="$DATA_DIR_DEFAULT"
TARGET="$TARGET_DEFAULT"
GPU_IDS="$GPU_IDS_DEFAULT"
CKPT_PATH="$CKPT_PATH_DEFAULT"
N_ESTIMATORS="$N_ESTIMATORS_DEFAULT"
BATCH_SIZE="$BATCH_SIZE_DEFAULT"
AMP="$AMP_DEFAULT"
PROG_CHUNK="$PROG_CHUNK_DEFAULT"
OUT_DIR="$OUT_DIR_DEFAULT"

MONITOR_AFTER_SECS="$MONITOR_AFTER_SECS_DEFAULT"
MONITOR_GPU_IDS="$MONITOR_GPU_IDS_DEFAULT"
LOGROOT="$LOGROOT_DEFAULT"

# 允许透传给 run_infer.sh 的“其它参数”
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_dir) DATA_DIR="$2"; shift 2 ;;
    --target) TARGET="$2"; shift 2 ;;
    --gpu_ids) GPU_IDS="$2"; shift 2 ;;
    --checkpoint_path) CKPT_PATH="$2"; shift 2 ;;
    --n_estimators) N_ESTIMATORS="$2"; shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --amp) AMP="$2"; shift 2 ;;
    --progress_chunk_rows) PROG_CHUNK="$2"; shift 2 ;;
    --out_dir) OUT_DIR="$2"; shift 2 ;;
    --monitor_after_secs) MONITOR_AFTER_SECS="$2"; shift 2 ;;
    --monitor_gpu_ids) MONITOR_GPU_IDS="$2"; shift 2 ;;
    --logroot) LOGROOT="$2"; shift 2 ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

# 如果未指定监控 GPU，则跟随推理的 GPU
if [[ -z "$MONITOR_GPU_IDS" ]]; then
  MONITOR_GPU_IDS="$GPU_IDS"
fi

mkdir -p "$OUT_DIR"
MAIN_LOG="$OUT_DIR/main.log"
echo "[infer_with_monitor] Start at $(date -Is)" | tee -a "$MAIN_LOG"

# ---------- 启动系统监控 ----------
if [[ ! -x "$SCRIPT_DIR/monitor_start.sh" || ! -x "$SCRIPT_DIR/monitor_stop.sh" ]]; then
  echo "ERR: 需要 $SCRIPT_DIR/monitor_start.sh 和 monitor_stop.sh" | tee -a "$MAIN_LOG"
  exit 1
fi

# 记录启动前已有的日志目录，便于识别新目录
BEFORE_LIST="$(ls -td "${LOGROOT}"/*/ 2>/dev/null || true)"

echo "[infer_with_monitor] Starting monitor for GPUs: ${MONITOR_GPU_IDS}" | tee -a "$MAIN_LOG"
# monitor_start 本身会在后台起多个采集进程，这里是一次性同步调用
"$SCRIPT_DIR/monitor_start.sh" "$MONITOR_GPU_IDS" >>"$MAIN_LOG" 2>&1 || true

# 找到刚刚创建的最新日志目录
sleep 1
MON_DIR="$(ls -td "${LOGROOT}"/*/ 2>/dev/null | head -n1 || true)"
if [[ -z "$MON_DIR" ]]; then
  echo "ERR: 未发现监控日志目录（${LOGROOT}/*/）。" | tee -a "$MAIN_LOG"
  exit 1
fi

# 若存在 BEFORE_LIST，尽量确保是“新目录”
if [[ -n "$BEFORE_LIST" && "$BEFORE_LIST" == *"$MON_DIR"* ]]; then
  # 可能在同一秒内重名；不致命，继续
  echo "[infer_with_monitor] monitor dir may coincide with previous: $MON_DIR" | tee -a "$MAIN_LOG"
else
  echo "[infer_with_monitor] Monitor dir: $MON_DIR" | tee -a "$MAIN_LOG"
fi

# 记录 & 软链接，便于查看
echo "$MON_DIR" > "$OUT_DIR/monitor_dir.txt"
ln -sfn "$MON_DIR" "$OUT_DIR/monitor_logs"

# ---------- 运行推理 ----------
if [[ ! -x "$SCRIPT_DIR/run_infer.sh" ]]; then
  echo "ERR: 需要 $SCRIPT_DIR/run_infer.sh" | tee -a "$MAIN_LOG"
  exit 1
fi

echo "[infer_with_monitor] Running inference ..." | tee -a "$MAIN_LOG"

# 组合传给 run_infer.sh 的参数（显式暴露你的默认项）
RUN_ARGS=(
  --data_dir "$DATA_DIR"
  --target "$TARGET"
  --gpu_ids "$GPU_IDS"
  --out_dir "$OUT_DIR"
  --checkpoint_path "$CKPT_PATH"
  --n_estimators "$N_ESTIMATORS"
  --batch_size "$BATCH_SIZE"
  --amp "$AMP"
  --progress_chunk_rows "$PROG_CHUNK"
  "${EXTRA_ARGS[@]}"
)

# 把 run_infer 的日志也并入 main.log（run_infer 内部还会写自己的 run.log）
"$SCRIPT_DIR/run_infer.sh" "${RUN_ARGS[@]}" 2>&1 | tee -a "$MAIN_LOG"

echo "[infer_with_monitor] Inference finished at $(date -Is)" | tee -a "$MAIN_LOG"

# ---------- 推理结束后，继续监控一段时间 ----------
if [[ "$MONITOR_AFTER_SECS" -gt 0 ]]; then
  echo "[infer_with_monitor] Cooling down (monitoring extra ${MONITOR_AFTER_SECS}s) ..." | tee -a "$MAIN_LOG"
  sleep "$MONITOR_AFTER_SECS"
fi

# ---------- 停止监控 ----------
echo "[infer_with_monitor] Stopping monitor in: $MON_DIR" | tee -a "$MAIN_LOG"
"$SCRIPT_DIR/monitor_stop.sh" "$MON_DIR" >>"$MAIN_LOG" 2>&1 || true

echo "[infer_with_monitor] All done at $(date -Is)" | tee -a "$MAIN_LOG"
