#!/usr/bin/env bash
# 后台启动 infer_with_monitor.sh（nohup + setsid），掉线不断。
# 日志: <OUT_DIR>/nohup.out    PID: <OUT_DIR>/job.pid

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OUT_DIR="runs/tabicl_infer_$(date +%Y%m%d_%H%M%S)"
ARGS=()

# 解析参数；同时捕获 --out_dir 的值
while [[ $# -gt 0 ]]; do
  case "$1" in
    --out_dir)
      if [[ $# -lt 2 ]]; then echo "ERR: --out_dir 需要一个值" >&2; exit 1; fi
      OUT_DIR="$2"
      ARGS+=("$1" "$2")
      shift 2
      ;;
    *)
      # 其它参数原样透传（含 --data_dir/--target/--gpu_ids/...）
      if [[ $# -ge 2 && "$2" != --* ]]; then
        ARGS+=("$1" "$2"); shift 2
      else
        ARGS+=("$1"); shift 1
      fi
      ;;
  esac
done

mkdir -p "$OUT_DIR"
LOG="$OUT_DIR/nohup.out"
echo "[starter] $(date -Is) launching infer_with_monitor.sh" | tee -a "$LOG"
echo "[starter] OUT_DIR=$OUT_DIR" | tee -a "$LOG"
echo "[starter] CMD: $SCRIPT_DIR/infer_with_monitor.sh ${ARGS[*]}" | tee -a "$LOG"

CMD=( "$SCRIPT_DIR/infer_with_monitor.sh" "${ARGS[@]}" )

if command -v setsid >/dev/null 2>&1; then
  nohup setsid "${CMD[@]}" >>"$LOG" 2>&1 &
else
  nohup "${CMD[@]}" >>"$LOG" 2>&1 &
fi

echo $! > "$OUT_DIR/job.pid"
echo "[starter] PID $(cat "$OUT_DIR"/job.pid) ; logs -> $LOG" | tee -a "$LOG"
