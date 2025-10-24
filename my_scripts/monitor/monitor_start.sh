#!/usr/bin/env bash
# monitor_start.sh (v3) — 1s 采样 + 可选项降级 + 指定 GPU + PCIe/NVLink 采集
# 用法：
#   ./monitor_start.sh [GPU_IDS]
#   例：./monitor_start.sh 0,3   或   GPU_IDS="1,2" ./monitor_start.sh

set -euo pipefail

GPU_IDS="${1:-${GPU_IDS:-0,1,2,3}}"
if ! [[ "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "ERR: GPU_IDS 格式不正确：'$GPU_IDS'（应为逗号分隔的数字，如 0,2,3）" >&2
  exit 1
fi

command -v nvidia-smi >/dev/null 2>&1 || { echo "ERR: 未找到 nvidia-smi"; exit 1; }

LOGROOT="logs"
TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="$LOGROOT/$TS"
mkdir -p "$LOGDIR"

{
  echo "Monitor started at $(date -Is)"
  echo "GPU_IDS=$GPU_IDS"
  echo "Hostname=$(hostname)"
  echo "User=$(id -un)"
  nvidia-smi -i "$GPU_IDS" --query-gpu=index,uuid,name,driver_version --format=csv,noheader || true
} | tee "$LOGDIR/START.txt"

# ---------- GPU ----------
nvidia-smi dmon -i "$GPU_IDS" -s pucm -d 1 > "$LOGDIR/gpu_dmon.log" 2>&1 & echo $! > "$LOGDIR/gpu_dmon.pid"
nvidia-smi pmon -i "$GPU_IDS" -s um -d 1 > "$LOGDIR/gpu_pmon.log" 2>&1 & echo $! > "$LOGDIR/gpu_pmon.pid"
# 其中 -s um 的 'm' 一般会让 pmon 增加 'fb' 列（不同驱动可能略有差异）

# ---------- CPU / 内存 ----------
if command -v pidstat >/dev/null 2>&1; then
  pidstat -durh 1 > "$LOGDIR/pidstat.log" 2>&1 & echo $! > "$LOGDIR/pidstat.pid"
else
  echo "warn: 未找到 pidstat；使用 vmstat/ps 代替" | tee -a "$LOGDIR/START.txt"
  if command -v vmstat >/dev/null 2>&1; then
    vmstat 1 > "$LOGDIR/vmstat.log" 2>&1 & echo $! > "$LOGDIR/vmstat.pid"
  else
    ( while true; do echo "===== $(date -Is) ====="; free -m; sleep 1; done ) \
      > "$LOGDIR/free_mem.log" 2>&1 & echo $! > "$LOGDIR/free_mem.pid"
  fi
  ( while true; do echo "===== $(date -Is) ====="; ps -eo pid,user,comm,%cpu,%mem --sort=-%cpu | head -n 25; sleep 2; done ) \
    > "$LOGDIR/ps_top.log" 2>&1 & echo $! > "$LOGDIR/ps_top.pid"
fi

# ---------- 磁盘 ----------
if command -v iostat >/dev/null 2>&1; then
  iostat -x 1 > "$LOGDIR/iostat.log" 2>&1 & echo $! > "$LOGDIR/iostat.pid"
else
  echo "warn: 未找到 iostat；跳过磁盘指标" | tee -a "$LOGDIR/START.txt"
fi

# ---------- 网络 ----------
if command -v ifstat >/dev/null 2>&1; then
  ifstat 1 > "$LOGDIR/ifstat.log" 2>&1 & echo $! > "$LOGDIR/ifstat.pid"
elif command -v sar >/dev/null 2>&1; then
  sar -n DEV 1 > "$LOGDIR/sar_net.log" 2>&1 & echo $! > "$LOGDIR/sar_net.pid"
else
  echo "warn: 未找到 ifstat/sar；跳过网络带宽" | tee -a "$LOGDIR/START.txt"
fi

# ---------- PCIe / NVLink ----------
if python -c "import pynvml" >/dev/null 2>&1; then
  python "$(dirname "$0")/pcie_nvlink_monitor.py" --gpu_ids "$GPU_IDS" --interval 1.0 \
    --out_csv "$LOGDIR/pcie_nvlink.csv" > "$LOGDIR/pcie_nvlink_monitor.log" 2>&1 \
    & echo $! > "$LOGDIR/pcie_nvlink_monitor.pid"
else
  echo "warn: 未装 pynvml；跳过 PCIe/NVLink（pip install nvidia-ml-py3）" | tee -a "$LOGDIR/START.txt"
fi

echo "PIDs written under $LOGDIR:"
ls "$LOGDIR"/*.pid 2>/dev/null || true
