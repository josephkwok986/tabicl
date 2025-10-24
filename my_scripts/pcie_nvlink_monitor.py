#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pcie_nvlink_monitor.py — 采集每张 GPU 的 PCIe 与（尽力）NVLink 吞吐，写入 CSV。
# 输出字段：
#   ts_iso, ts_epoch, gpu, pcie_rx_kBs, pcie_tx_kBs, nvlink_rx_kBs, nvlink_tx_kBs
#
# 依赖：pynvml（pip 包名 nvidia-ml-py3），nvidia-smi（用于尝试解析 NVLink）。
# 说明：
# - PCIe 吞吐来自 NVML：单位 kB/s，稳定可靠。
# - NVLink：不同驱动/设备输出差异很大，这里用 `nvidia-smi nvlink --status -i <idx>` 做“尽力解析”，
#   如果无法解析，就记空（留空字符串）。
#
# 用法示例：
#   python pcie_nvlink_monitor.py --gpu_ids 0,3 --interval 1.0 --out_csv logs/20250101_000000/pcie_nvlink.csv

import argparse, time, os, sys, csv, subprocess, re
from datetime import datetime

try:
    import pynvml as N
except Exception:
    N = None

def parse_gpu_ids(s: str):
    return [int(x) for x in s.split(",") if x.strip() != ""]

def now_iso():
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"

# ===== NVLink 解析（尽力而为） =====
_prev_nvlink_bytes = {}  # (gpu, 'tx'/'rx') -> (bytes, ts)
_unit_scale = {"b":1, "kb":1024, "mb":1024**2, "gb":1024**3, "bytes":1}

def _to_bytes(val: str, unit: str) -> float:
    unit = unit.strip().lower()
    if unit in _unit_scale:
        return float(val) * _unit_scale[unit]
    # 有些版本可能输出 "MB/s" 之类的速率（不是累计字节），尽量按每秒理解
    if unit.endswith("/s"):
        base = unit[:-2]
        return float(val) * _unit_scale.get(base, 1.0)
    return float(val)

# 例： "... Tx: 123.4 MB   Rx: 111.1 MB ..."
_nv_pat = re.compile(r'\b(Tx|TX|tx|Rx|RX|rx)\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)\s*([A-Za-z/]+)')

def read_nvlink_kBs_via_nvsmi(gpu_idx: int):
    """
    使用 `nvidia-smi nvlink --status -i <idx>` 尝试解析 Tx/Rx 数值并累加后做差分求速率。
    返回 (rx_kBs, tx_kBs)，解析失败时返回 (None, None)。
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "nvlink", "--status", "-i", str(gpu_idx)],
            stderr=subprocess.STDOUT, text=True
        )
    except Exception:
        return (None, None)

    total_bytes = {"tx": 0.0, "rx": 0.0}
    for m in _nv_pat.finditer(out):
        kind = m.group(1).lower()  # tx / rx
        val = m.group(2)
        unit = m.group(3)
        try:
            b = _to_bytes(val, unit)
        except Exception:
            continue
        if kind.startswith("t"):
            total_bytes["tx"] += b
        else:
            total_bytes["rx"] += b

    if total_bytes["tx"] == 0.0 and total_bytes["rx"] == 0.0:
        return (None, None)

    now = time.time()
    rx_prev = _prev_nvlink_bytes.get((gpu_idx, "rx"))
    tx_prev = _prev_nvlink_bytes.get((gpu_idx, "tx"))
    rx_kBs = tx_kBs = None

    if rx_prev:
        diff_b = max(0.0, total_bytes["rx"] - rx_prev[0])
        dt = max(1e-6, now - rx_prev[1])
        rx_kBs = diff_b / dt / 1024.0
    if tx_prev:
        diff_b = max(0.0, total_bytes["tx"] - tx_prev[0])
        dt = max(1e-6, now - tx_prev[1])
        tx_kBs = diff_b / dt / 1024.0

    _prev_nvlink_bytes[(gpu_idx, "rx")] = (total_bytes["rx"], now)
    _prev_nvlink_bytes[(gpu_idx, "tx")] = (total_bytes["tx"], now)
    return (rx_kBs, tx_kBs)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu_ids", type=str, default="0,1,2,3", help="逗号分隔的 GPU id，例如 0,3")
    ap.add_argument("--interval", type=float, default=1.0, help="采样间隔（秒）")
    ap.add_argument("--out_csv", type=str, required=True, help="输出 CSV 路径")
    args = ap.parse_args()

    if N is None:
        print("ERR: pynvml not available. 请先安装：pip install nvidia-ml-py3", file=sys.stderr)
        sys.exit(1)

    try:
        N.nvmlInit()
    except Exception as e:
        print(f"ERR: NVML init failed: {e}", file=sys.stderr)
        sys.exit(1)

    gpu_ids = parse_gpu_ids(args.gpu_ids)
    handles = {}
    for i in gpu_ids:
        try:
            handles[i] = N.nvmlDeviceGetHandleByIndex(i)
        except Exception as e:
            print(f"warn: 获取 GPU {i} 句柄失败：{e}", file=sys.stderr)

    out_path = args.out_csv
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    newf = not os.path.exists(out_path)

    with open(out_path, "a", newline="") as fp:
        wr = csv.writer(fp)
        if newf:
            wr.writerow(["ts_iso","ts_epoch","gpu","pcie_rx_kBs","pcie_tx_kBs","nvlink_rx_kBs","nvlink_tx_kBs"])
        while True:
            ts = time.time()
            ts_iso = now_iso()
            for gid, h in handles.items():
                # PCIe：NVML 直接返回 kB/s
                try:
                    rx_kB = float(N.nvmlDeviceGetPcieThroughput(h, N.NVML_PCIE_UTIL_RX_BYTES))
                    tx_kB = float(N.nvmlDeviceGetPcieThroughput(h, N.NVML_PCIE_UTIL_TX_BYTES))
                except Exception:
                    rx_kB = tx_kB = None

                # NVLink：尽力解析（可能不可用/返回空）
                nv_rx_kB, nv_tx_kB = read_nvlink_kBs_via_nvsmi(gid)

                wr.writerow([
                    ts_iso, f"{ts:.3f}", gid,
                    f"{rx_kB:.3f}" if rx_kB is not None else "",
                    f"{tx_kB:.3f}" if tx_kB is not None else "",
                    f"{nv_rx_kB:.3f}" if nv_rx_kB is not None else "",
                    f"{nv_tx_kB:.3f}" if nv_tx_kB is not None else ""
                ])
            fp.flush()
            time.sleep(args.interval)

if __name__ == "__main__":
    main()
