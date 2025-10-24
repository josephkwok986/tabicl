#!/usr/bin/env bash
# stop_infer_only_all.sh
# 作用：遍历 runs 根目录下的所有任务 OUT_DIR，仅停止推理进程（infer_tabicl.py），不停止监控。
# 默认 runs 根目录：/home/amax/gjj_tabicl/scripts/runs
#
# 用法：
#   ./stop_infer_only_all.sh                           # 用默认根目录
#   ./stop_infer_only_all.sh --runs_root <路径>        # 自定义根目录
#   ./stop_infer_only_all.sh --dry_run                 # 只打印将要停止的进程，不执行
#
# 停止策略（每个任务）：
#   1) 优先依据 OUT_DIR/job.pid 找到该作业的进程组 PGID；
#      在该组内仅挑出命令行包含 infer_tabicl.py 的进程进行 SIGINT->SIGTERM->SIGKILL 逐级停止。
#   2) 若 job.pid/PGID 不可用，则回退为按命令行匹配：
#      匹配包含 "infer_tabicl.py" 且包含 "--out_dir <该 OUT_DIR>" 的进程，仅停止这些 PID。

set -euo pipefail

RUNS_ROOT="/home/amax/gjj_tabicl/scripts/runs"
DRY_RUN=0

# -------- 解析参数 --------
while [[ $# -gt 0 ]]; do
  case "$1" in
    --runs_root) RUNS_ROOT="$2"; shift 2 ;;
    --dry_run)   DRY_RUN=1; shift ;;
    *) echo "未知参数: $1" >&2; exit 1 ;;
  esac
done

if [[ ! -d "$RUNS_ROOT" ]]; then
  echo "ERR: runs 根目录不存在: $RUNS_ROOT" >&2
  exit 1
fi

echo "[info] runs 根目录: $RUNS_ROOT"
[[ $DRY_RUN -eq 1 ]] && echo "[info] DRY-RUN 模式：只展示将要停止的进程，不实际发送信号"

stop_infer_one_outdir() {
  local OUT_DIR="$1"
  local did_any=0

  echo "----------------------------------------"
  echo "[task] OUT_DIR = $OUT_DIR"

  # A. 优先用 job.pid + 进程组过滤
  if [[ -f "$OUT_DIR/job.pid" ]]; then
    local PID PGID
    PID="$(cat "$OUT_DIR/job.pid" 2>/dev/null || true)"
    if [[ -n "${PID:-}" ]] && ps -p "$PID" >/dev/null 2>&1; then
      PGID="$(ps -o pgid= -p "$PID" | tr -d ' ')"
      if [[ -n "${PGID:-}" ]]; then
        echo "[info] 进程组 PGID=$PGID（来自 job.pid=$PID）"
        # 只选择该组内的 infer_tabicl.py
        mapfile -t INF_PIDS < <(ps -o pid,cmd --group "$PGID" | awk '/infer_tabicl\.py/ {print $1}')
        if [[ ${#INF_PIDS[@]} -gt 0 ]]; then
          echo "[info] 发现推理进程（进程组内）: ${INF_PIDS[*]}"
          if [[ $DRY_RUN -eq 0 ]]; then
            kill -INT "${INF_PIDS[@]}" 2>/dev/null || true
            sleep 10
            # 仍在的再 TERM
            mapfile -t STILL < <(printf '%s\n' "${INF_PIDS[@]}" | xargs -r ps -p | awk 'NR>1{print $1}')
            if [[ ${#STILL[@]} -gt 0 ]]; then
              kill -TERM "${STILL[@]}" 2>/dev/null || true
              sleep 5
              # 仍在的再 KILL
              mapfile -t STILL2 < <(printf '%s\n' "${STILL[@]}" | xargs -r ps -p | awk 'NR>1{print $1}')
              [[ ${#STILL2[@]} -gt 0 ]] && kill -KILL "${STILL2[@]}" 2>/dev/null || true
            fi
          fi
          did_any=1
        else
          echo "[info] 该进程组内未发现 infer_tabicl.py，尝试回退匹配..."
        fi
      fi
    else
      echo "[warn] job.pid 指向的 PID 不存在，尝试回退匹配..."
    fi
  else
    echo "[info] 未找到 $OUT_DIR/job.pid，尝试回退匹配..."
  fi

  # B. 回退：按命令行匹配 "--out_dir <OUT_DIR>" 的 infer_tabicl.py
  if [[ $did_any -eq 0 ]]; then
    # 用 ps aux 逐行查找，避免 pgrep 的正则转义问题
    # 只匹配包含 infer_tabicl.py 且包含 "--out_dir <OUT_DIR>" 或 "--out_dir=<OUT_DIR>"
    local escaped_outdir
    escaped_outdir="$(printf '%s' "$OUT_DIR" | sed 's/[.[\*^$()+?{}|]/\\&/g')"
    mapfile -t INF_PIDS2 < <(
      ps axo pid=,command= \
      | grep -F "infer_tabicl.py" \
      | grep -E -- "--out_dir(=| )${escaped_outdir}(/| |$)" \
      | awk '{print $1}'
    )
    if [[ ${#INF_PIDS2[@]} -gt 0 ]]; then
      echo "[info] 发现推理进程（命令行匹配）: ${INF_PIDS2[*]}"
      if [[ $DRY_RUN -eq 0 ]]; then
        kill -INT "${INF_PIDS2[@]}" 2>/dev/null || true
        sleep 10
        mapfile -t STILL < <(printf '%s\n' "${INF_PIDS2[@]}" | xargs -r ps -p | awk 'NR>1{print $1}')
        if [[ ${#STILL[@]} -gt 0 ]]; then
          kill -TERM "${STILL[@]}" 2>/dev/null || true
          sleep 5
          mapfile -t STILL2 < <(printf '%s\n' "${STILL[@]}" | xargs -r ps -p | awk 'NR>1{print $1}')
          [[ ${#STILL2[@]} -gt 0 ]] && kill -KILL "${STILL2[@]}" 2>/dev/null || true
        fi
      fi
      did_any=1
    else
      echo "[info] 未发现需要停止的推理进程。"
    fi
  fi
}

# -------- 遍历 runs 根目录一级子目录（视为各个 OUT_DIR）--------
shopt -s nullglob
found_any=0
for d in "$RUNS_ROOT"/*; do
  [[ -d "$d" ]] || continue
  found_any=1
  stop_infer_one_outdir "$d"
done

if [[ $found_any -eq 0 ]]; then
  echo "[info] 在 $RUNS_ROOT 下未发现子目录。"
fi

echo "=== 完成：已按需停止各 OUT_DIR 的推理进程（监控未停止） ==="
