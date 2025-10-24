#!/usr/bin/env bash
set -euo pipefail

echo "== GPU / Driver =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | sed 's/^/  /'
  echo "  nvlink status cmd:"; nvidia-smi nvlink -h >/dev/null 2>&1 && echo "  OK (has 'nvidia-smi nvlink')" || echo "  MISSING (no 'nvlink' subcmd)"
else
  echo "  MISSING: nvidia-smi"
fi

echo
echo "== CPU / Memory tools =="
command -v pidstat >/dev/null 2>&1 && pidstat -V 2>/dev/null | head -n1 || echo "  pidstat: MISSING"
command -v vmstat  >/dev/null 2>&1 && vmstat -V  2>/dev/null | head -n1 || echo "  vmstat : MISSING"

echo
echo "== Disk IO =="
command -v iostat  >/dev/null 2>&1 && iostat -V  2>/dev/null | head -n1 || echo "  iostat : MISSING"

echo
echo "== Network =="
if command -v ifstat >/dev/null 2>&1; then
  echo "  ifstat : OK"
else
  echo "  ifstat : MISSING"
fi
command -v sar >/dev/null 2>&1 && sar -V 2>/dev/null | head -n1 || echo "  sar    : MISSING"

echo
echo "== Python & pynvml =="
command -v python3 >/dev/null 2>&1 && python3 -V || echo "  python3: MISSING"
if command -v python3 >/dev/null 2>&1; then
  python3 - <<'PY'
import sys
print("  Python exe:", sys.executable)
try:
  import pynvml as N
  N.nvmlInit()
  print("  pynvml   : OK, driver version:", N.nvmlSystemGetDriverVersion().decode() if hasattr(N.nvmlSystemGetDriverVersion(), 'decode') else N.nvmlSystemGetDriverVersion())
except ModuleNotFoundError:
  print("  pynvml   : MISSING (ModuleNotFoundError)")
except Exception as e:
  print("  pynvml   : Present but init failed:", e)
PY
fi

echo
echo "== Optional (nohup/setsid for background) =="
command -v nohup >/dev/null 2>&1 && echo "  nohup   : OK" || echo "  nohup   : MISSING"
command -v setsid >/dev/null 2>&1 && echo "  setsid  : OK" || echo "  setsid  : MISSING"
