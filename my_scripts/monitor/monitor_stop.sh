#!/usr/bin/env bash
# 用法：
#   ./monitor_stop.sh                # 停止最近一次监控
#   ./monitor_stop.sh logs/20250101_123000/   # 指定日志目录
set -euo pipefail

LOGROOT="logs"
LATEST="$(ls -td ${LOGROOT}/*/ 2>/dev/null | head -n1 || true)"
TARGET="${1:-$LATEST}"

if [[ -z "${TARGET}" ]]; then
  echo "no log dir found"
  exit 0
fi

echo "Stopping monitors in: $TARGET"
for f in "$TARGET"/*.pid; do
  [[ -e "$f" ]] || continue
  PID="$(cat "$f" || true)"
  if [[ -n "${PID:-}" ]]; then
    kill "$PID" 2>/dev/null || true
  fi
done
echo "Done."
