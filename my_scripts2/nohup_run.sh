#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "用法: $0 <脚本路径> [脚本参数...]" >&2
  exit 1
fi

SCRIPT="$1"
shift                         # 剩余的全部参数都透传给脚本
ARGS=("$@")                   # 保留原样（含空格/中文）

if [[ ! -f "$SCRIPT" ]]; then
  echo "找不到脚本: $SCRIPT" >&2
  exit 2
fi

BASENAME="$(basename "$SCRIPT")"
LOG_FILE="${BASENAME}_log"    # 日志名形如: xxx.sh_log
PID_FILE="${BASENAME}.pid"

# 以 nohup 执行，并透传参数
nohup bash "$SCRIPT" "${ARGS[@]}" > "$LOG_FILE" 2>&1 &
PID=$!

# 输出并保存 PID
echo "$PID" | tee "$PID_FILE" >/dev/null

# 友好提示
echo "已启动: $SCRIPT ${ARGS[*]:-}"
echo "PID: $PID"
echo "日志: $(pwd)/$LOG_FILE"
echo "PID文件: $(pwd)/$PID_FILE"
echo "结束进程可执行: kill $PID"
echo "或使用: kill \$(cat \"$PID_FILE\")"
