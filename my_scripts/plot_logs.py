#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
python plot_logs.py --mon_dir "/workspace/Gjj Local/下载/logs/logs/20250917_200939"
"""

import argparse, os, sys, re, math, json, datetime
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def read_start_time(mon_dir: Path):
    start_path = mon_dir / "START.txt"
    if not start_path.exists():
        return None
    try:
        txt = start_path.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"Monitor started at\s+([0-9T:\-\.]+(?:Z|[+\-][0-9:]+)?)", txt)
        if m:
            s = m.group(1)
            try:
                ts = datetime.datetime.fromisoformat(s.replace("Z","+00:00"))
                return ts
            except Exception:
                pass
    except Exception:
        pass
    return None

def robust_split_ws(line):
    return [tok for tok in re.split(r"\s+", line.strip()) if tok]

def parse_dmon(mon_dir: Path, sample_interval: float = 1.0, start_ts=None):
    path = mon_dir / "gpu_dmon.log"
    if not path.exists():
        return None
    rows = []
    header_cols = None
    first_gpu_id = None
    tick_idx = -1
    current_time_s = 0.0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            if line.lstrip().startswith("#"):
                hdr = line.strip("# \n")
                cols = [tok for tok in re.split(r"\s+", hdr.lower()) if tok]
                if "gpu" in cols:
                    header_cols = cols
                continue
            if header_cols is None:
                continue
            toks = [tok for tok in re.split(r"\s+", line.strip()) if tok]
            if len(toks) < len(header_cols):
                continue
            try:
                gpu_id = int(toks[header_cols.index("gpu")])
            except Exception:
                continue
            if first_gpu_id is None:
                first_gpu_id = gpu_id
                tick_idx = 0
                current_time_s = 0.0
            elif gpu_id == first_gpu_id:
                tick_idx += 1
                current_time_s = tick_idx * sample_interval
            rec = {"gpu": gpu_id, "time_s": current_time_s}
            mapping = {"pwr":"pwr_W","sm":"sm_%","mem":"mem_%","mclk":"mclk_MHz","pclk":"pclk_MHz","gtemp":"temp_C","temp":"temp_C"}
            for i, name in enumerate(header_cols):
                if name == "gpu":
                    continue
                try:
                    v = float(toks[i])
                except Exception:
                    continue
                std = mapping.get(name)
                if std:
                    rec[std] = v
            rows.append(rec)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    if start_ts is not None:
        df["timestamp"] = pd.to_datetime(start_ts) + pd.to_timedelta(df["time_s"], unit="s")
    for col in ["pwr_W","sm_%","mem_%","mclk_MHz","pclk_MHz","temp_C"]:
        if col not in df.columns:
            df[col] = np.nan
    return df

def parse_pmon(mon_dir: Path, sample_interval: float = 1.0, start_ts=None):
    path = mon_dir / "gpu_pmon.log"
    if not path.exists(): return None
    rows=[]; header_cols=None
    first_gpu_id=None; tick_idx=-1; current_time_s=0.0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s=line.strip()
            if not s: continue
            if s.startswith("#"):
                # 只记录列名，不在这里推进时间
                cols=[tok for tok in re.split(r"\s+", s.strip("# ").lower()) if tok]
                if "gpu" in cols and "pid" in cols: header_cols=cols
                continue
            if header_cols is None: continue
            toks=[tok for tok in re.split(r"\s+", s) if tok]
            if len(toks) < len(header_cols): continue
            try:
                g=int(toks[header_cols.index("gpu")])
            except: continue

            # 时间推进：每次再次遇到“首个 GPU”的行，就进入下一秒
            if first_gpu_id is None:
                first_gpu_id=g; tick_idx=0; current_time_s=0.0
            elif g == first_gpu_id:
                tick_idx += 1
                current_time_s = tick_idx * sample_interval

            rec={"gpu": g, "time_s": current_time_s}
            # 读 fb（MiB）
            try:
                rec["fb_proc_MB"] = float(toks[header_cols.index("fb")])
            except: pass
            # 可选：保留 sm%
            try:
                rec["sm_proc"] = float(toks[header_cols.index("sm")])
            except: pass
            rows.append(rec)

    if not rows: return None
    df = pd.DataFrame(rows)
    agg = df.groupby(["time_s","gpu"]).agg(
        fb_mem_MB=("fb_proc_MB","sum"),
        sm_pct=("sm_proc","sum"),
        proc_count=("fb_proc_MB","count"),
    ).reset_index()
    if start_ts is not None:
        agg["timestamp"] = pd.to_datetime(start_ts) + pd.to_timedelta(agg["time_s"], unit="s")
    return agg

def parse_vmstat(mon_dir: Path):
    path = mon_dir / "vmstat.log"
    if not path.exists():
        path = mon_dir / "free_mem.log"
        if not path.exists():
            return None
        rows=[]; t=0; stamp=None
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if line.startswith("===== "):
                    stamp = True; continue
                if stamp:
                    rows.append({"time_s": t, "note":"free_mem"})
                    t += 1; stamp=None
        return pd.DataFrame(rows) if rows else None
    rows=[]; header=None; sample_idx=0
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            if header is None and re.search(r"\br\b.*\bb\b", line) and ("si" in line or "swpd" in line):
                header = robust_split_ws(line.lower())
                continue
            if header is None:
                continue
            toks = robust_split_ws(line)
            if len(toks) != len(header):
                continue
            try:
                vals = list(map(float, toks))
            except:
                continue
            rec = dict(zip(header, vals)); rec["time_s"]=sample_idx; sample_idx+=1; rows.append(rec)
    return pd.DataFrame(rows) if rows else None

def parse_iostat(mon_dir: Path):
    path = mon_dir / "iostat.log"
    if not path.exists():
        return None
    rows = []; current_ts=0; header=None
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.lower().startswith("device") and "%util" in s.lower():
                header = robust_split_ws(s.lower()); continue
            if header is None:
                continue
            toks = robust_split_ws(s)
            if len(toks) != len(header):
                if re.match(r"^[A-Za-z]{3}\s+\d", s) or re.match(r"^\d{2}:\d{2}:", s):
                    current_ts += 1
                continue
            rec = dict(zip(header, toks))
            try:
                util = float(rec.get("%util", rec.get("util", "nan")))
            except: util=float('nan')
            try:
                await_ms_val = float(rec.get("await", "nan")) if rec.get("await") is not None else float('nan')
            except: await_ms_val=float('nan')
            rows.append({"time_s": current_ts, "util_pct": util, "await_ms": await_ms_val})
    if not rows: return None
    df = pd.DataFrame(rows).groupby("time_s").mean(numeric_only=True).reset_index()
    return df

def parse_net(mon_dir: Path):
    path_if = mon_dir / "ifstat.log"
    path_sar = mon_dir / "sar_net.log"
    if path_if.exists():
        rows=[]; 
        with path_if.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s=line.strip(); 
                if not s: continue
                toks=robust_split_ws(s); num=0; total=0.0
                for tok in toks:
                    try:
                        v=float(tok); num+=1; total+=v
                    except: pass
                if num>=2:
                    rows.append({"time_s": len(rows), "sum_kBs": total})
        return pd.DataFrame(rows) if rows else None
    elif path_sar.exists():
        rows=[]; header=None
        with path_sar.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                s=line.strip(); toks=robust_split_ws(s)
                if not toks: continue
                if "IFACE" in toks and "rxkB/s" in toks and "txkB/s" in toks:
                    header=toks; continue
                if header is None: continue
                if re.match(r"^\d{1,2}:\d{2}:\d{2}", toks[0]) and len(toks)>=4:
                    try: rx=float(toks[2]); tx=float(toks[3])
                    except: continue
                    rows.append({"time_idx": len(rows), "rx_kBs": rx, "tx_kBs": tx})
        if rows:
            df=pd.DataFrame(rows); df["time_s"]=df["time_idx"]; df["sum_kBs"]=df["rx_kBs"]+df["tx_kBs"]; 
            return df[["time_s","sum_kBs"]]
    return None

def ensure_out(out_dir: Path): out_dir.mkdir(parents=True, exist_ok=True)

def plot_series(df, x, y, title, out_path):
    if df is None or df.empty: return False
    fig = plt.figure(); plt.plot(df[x], df[y]); plt.xlabel(x); plt.ylabel(y); plt.title(title)
    fig.savefig(out_path, bbox_inches="tight"); plt.close(fig); return True

def plot_gpu_metrics(df_dmon, df_pmon, out_dir: Path):
    made=[]
    if df_dmon is not None and not df_dmon.empty:
        for col,label in [("sm_%","GPU SM Util (%)"),("mem_%","GPU MemCtrl Util (%)"),("pwr_W","GPU Power (W)"),("temp_C","GPU Temp (C)")]: 
            fig=plt.figure()
            for gid,sub in df_dmon.groupby("gpu"):
                sub=sub.sort_values("time_s"); plt.plot(sub["time_s"], sub[col], label=f"gpu{gid}")
            plt.xlabel("time_s"); plt.ylabel(col); plt.title(label); plt.legend()
            out=out_dir/f"dmon_{col}.png"; fig.savefig(out, bbox_inches="tight"); plt.close(fig); made.append(out)
    if df_pmon is not None and not df_pmon.empty and "fb_mem_MB" in df_pmon.columns:
        fig=plt.figure()
        for gid,sub in df_pmon.groupby("gpu"):
            sub=sub.sort_values("time_s"); plt.plot(sub["time_s"], sub["fb_mem_MB"], label=f"gpu{gid}")
        plt.xlabel("time_s"); plt.ylabel("fb_mem_MB"); plt.title("GPU FB Memory Usage by processes (MB)"); plt.legend()
        out=out_dir/"pmon_fb_mem_MB.png"; fig.savefig(out, bbox_inches="tight"); plt.close(fig); made.append(out)
    return made

def parse_pcie_nvlink(mon_dir: Path):
    path = mon_dir / "pcie_nvlink.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    try:
        df["ts_epoch"] = pd.to_numeric(df["ts_epoch"], errors="coerce")
        df["gpu"] = pd.to_numeric(df["gpu"], errors="coerce").astype("Int64")
    except Exception:
        return None
    df = df.dropna(subset=["ts_epoch","gpu"])
    df["time_s"] = df["ts_epoch"] - df["ts_epoch"].min()
    for c in ["pcie_rx_kBs","pcie_tx_kBs","nvlink_rx_kBs","nvlink_tx_kBs"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def plot_pcie_nvlink(df, out_dir: Path):
    if df is None or df.empty:
        return []
    made=[]
    fig=plt.figure()
    for gid, sub in df.groupby("gpu"):
        sub=sub.sort_values("time_s")
        total_MBps=(sub["pcie_rx_kBs"].fillna(0)+sub["pcie_tx_kBs"].fillna(0))/1024.0
        plt.plot(sub["time_s"], total_MBps, label=f"gpu{int(gid)}")
    plt.xlabel("time_s"); plt.ylabel("MB/s"); plt.title("PCIe total (RX+TX) per GPU"); plt.legend()
    out=out_dir/"pcie_total_MBps.png"; fig.savefig(out, bbox_inches="tight"); plt.close(fig); made.append(out)

    if "nvlink_rx_kBs" in df.columns and df["nvlink_rx_kBs"].notna().any():
        fig=plt.figure()
        for gid, sub in df.groupby("gpu"):
            sub=sub.sort_values("time_s")
            total_MBps=(sub["nvlink_rx_kBs"].fillna(0)+sub["nvlink_tx_kBs"].fillna(0))/1024.0
            plt.plot(sub["time_s"], total_MBps, label=f"gpu{int(gid)}")
        plt.xlabel("time_s"); plt.ylabel("MB/s"); plt.title("NVLink total (RX+TX) per GPU"); plt.legend()
        out=out_dir/"nvlink_total_MBps.png"; fig.savefig(out, bbox_inches="tight"); plt.close(fig); made.append(out)
    return made

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mon_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--sample_interval", type=float, default=1.0)
    args = ap.parse_args()

    mon_dir = Path(args.mon_dir); out_dir = Path(args.out_dir) if args.out_dir else (mon_dir / "plots")
    out_dir.mkdir(parents=True, exist_ok=True)
    start_ts = read_start_time(mon_dir)

    df_dmon = parse_dmon(mon_dir, sample_interval=args.sample_interval, start_ts=start_ts)
    df_pmon = parse_pmon(mon_dir, sample_interval=args.sample_interval, start_ts=start_ts)
    df_vm = parse_vmstat(mon_dir)
    df_io = parse_iostat(mon_dir)
    df_net = parse_net(mon_dir)
    df_pcie = parse_pcie_nvlink(mon_dir)

    if df_dmon is not None: df_dmon.to_csv(out_dir/"parsed_gpu_dmon.csv", index=False)
    if df_pmon is not None: df_pmon.to_csv(out_dir/"parsed_gpu_pmon.csv", index=False)
    if df_vm is not None: df_vm.to_csv(out_dir/"parsed_vmstat.csv", index=False)
    if df_io is not None: df_io.to_csv(out_dir/"parsed_iostat.csv", index=False)
    if df_net is not None: df_net.to_csv(out_dir/"parsed_net.csv", index=False)
    if df_pcie is not None: df_pcie.to_csv(out_dir/"parsed_pcie_nvlink.csv", index=False)

    made = []
    made += plot_pcie_nvlink(df_pcie, out_dir)
    made += plot_gpu_metrics(df_dmon, df_pmon, out_dir)
    if df_vm is not None and "id" in df_vm.columns:
        cpu_df = df_vm[["time_s","id"]].copy(); cpu_df["cpu_usage_pct"] = 100.0 - cpu_df["id"]
        plot_series(cpu_df, "time_s", "cpu_usage_pct", "CPU Usage (%)", out_dir/"cpu_usage_pct.png")
        made.append(out_dir/"cpu_usage_pct.png")
    if df_io is not None and not df_io.empty:
        plot_series(df_io, "time_s", "util_pct", "Disk %util (avg)", out_dir/"disk_util_pct.png"); made.append(out_dir/"disk_util_pct.png")
        if "await_ms" in df_io.columns:
            plot_series(df_io, "time_s", "await_ms", "Disk await (ms, avg)", out_dir/"disk_await_ms.png"); made.append(out_dir/"disk_await_ms.png")
    if df_net is not None and not df_net.empty:
        plot_series(df_net, "time_s", "sum_kBs", "Network sum (kB/s)", out_dir/"net_sum_kBs.png"); made.append(out_dir/"net_sum_kBs.png")

    idx = out_dir/"index.html"
    with idx.open("w", encoding="utf-8") as f:
        f.write("<html><head><meta charset='utf-8'><title>Monitor Plots</title></head><body>\n")
        f.write(f"<h1>Plots for {mon_dir}</h1>\n")
        for p in made:
            f.write(f"<div><h3>{p.name}</h3><img src='{p.name}' style='max-width:100%'></div>\n")
        f.write("</body></html>\n")
    print(f"[OK] Wrote {len(made)} plots to {out_dir}")
    print(f"[OK] Open {idx}")

if __name__ == "__main__":
    main()
