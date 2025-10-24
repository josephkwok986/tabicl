#!/usr/bin/env python
# -*- coding: utf-8 -*-
# gen_prior_data.py  —— 合成多套（train/test）表格数据，并持续输出进度
#
# 用法示例：
#   python gen_prior_data.py --out_dir data/synth --n_datasets 3 \
#     --rows_min 20000 --rows_max 80000 --cols_min 20 --cols_max 120 \
#     --n_classes_max 10 --prior_type mix_scm --seed 42 \
#     --hb_interval 1 --save_chunk_rows 200000
#
# 进度输出包含两类：
# 1) 心跳（Heartbeat）：默认每 1 秒输出一行，显示当前数据集编号、阶段、耗时等。
# 2) 阶段细节：列缩放（逐列）与 CSV 写入（按行块）会打印更细的进度。
#
# 关闭心跳：加 --quiet     调整心跳频率：--hb_interval 0.5
# 写入 CSV 的块大小：--save_chunk_rows（默认 200000 行）

import os, json, argparse, importlib, threading, time, sys
import numpy as np
import pandas as pd

# ----------------------------
# 实用工具：尝试导入 TabICL 的 prior 生成模块
# ----------------------------
def try_import_tabicl_prior():
    try:
        return importlib.import_module("tabicl.prior.dataset")
    except Exception:
        return None

# ----------------------------
# 全局进度状态（由各阶段更新；Heartbeat 线程定期打印）
# ----------------------------
class ProgressState:
    def __init__(self, total_sets: int, hb_interval: float = 1.0, quiet: bool = False):
        self.lock = threading.Lock()
        self.start_ts = time.perf_counter()
        self.hb_interval = max(0.1, float(hb_interval)) if not quiet else None
        self.quiet = quiet

        # 动态字段
        self.cur_set = -1           # 当前数据集索引（0-based）
        self.total_sets = total_sets
        self.phase = "init"         # init/generate/scale/save/meta/done
        self.rows = 0
        self.cols = 0
        self.cols_done = 0
        self.cols_total = 0
        self.save_kind = ""         # "train"/"test"
        self.save_done = 0
        self.save_total = 0
        self.using_prior = False
        self.subdir = ""

        self._stop = threading.Event()
        self._thr = None

    def update(self, **kw):
        with self.lock:
            for k, v in kw.items():
                setattr(self, k, v)

    def fmt_status(self):
        with self.lock:
            elapsed = time.perf_counter() - self.start_ts
            base = f"[HB {elapsed:7.1f}s] set={self.cur_set+1}/{self.total_sets} phase={self.phase}"
            extra = []
            if self.phase == "generate":
                extra.append(f"prior={int(self.using_prior)} rows={self.rows} cols={self.cols}")
            elif self.phase == "scale":
                if self.cols_total:
                    pct = 100.0 * self.cols_done / max(1, self.cols_total)
                    extra.append(f"cols={self.cols_done}/{self.cols_total} ({pct:.1f}%)")
            elif self.phase == "save":
                if self.save_total:
                    pct = 100.0 * self.save_done / max(1, self.save_total)
                    extra.append(f"{self.save_kind}: {self.save_done}/{self.save_total} ({pct:.1f}%)")
            elif self.phase == "meta":
                extra.append("writing meta.json")
            elif self.phase == "done":
                extra.append("all done")
            if self.subdir:
                extra.append(f"path={self.subdir}")
            return base + (" | " + " ".join(extra) if extra else "")

    def start_heartbeat(self):
        if self.quiet:
            return
        def _run():
            while not self._stop.is_set():
                print(self.fmt_status(), flush=True)
                time.sleep(self.hb_interval)
        self._thr = threading.Thread(target=_run, daemon=True)
        self._thr.start()

    def stop_heartbeat(self):
        if self._thr is None:
            return
        self._stop.set()
        self._thr.join(timeout=1.0)
        # 结束前再打印一次最终状态
        print(self.fmt_status(), flush=True)

# ----------------------------
# 数值列缩放到 [0, 999]，带逐列进度
# ----------------------------
def minmax_scale_0_999(df: pd.DataFrame, ignore_cols, prog: ProgressState = None):
    df2 = df.copy()
    num_cols = [c for c in df2.columns if c not in ignore_cols and pd.api.types.is_numeric_dtype(df2[c])]
    total = len(num_cols)
    if prog is not None:
        prog.update(phase="scale", cols_total=total, cols_done=0)
    for i, c in enumerate(num_cols, 1):
        col = df2[c].astype(float)
        lo, hi = np.nanmin(col.values), np.nanmax(col.values)
        if not np.isfinite(lo) or not np.isfinite(hi) or hi == lo:
            df2[c] = 0.0
        else:
            df2[c] = (col - lo) * 999.0 / (hi - lo)
        # 进度输出（逐列）
        if prog is not None:
            prog.update(cols_done=i)
            print(f"[scale] col {i}/{total}: {c}", flush=True)
    return df2

# ----------------------------
# sklearn 合成器（打印开始/结束）
# ----------------------------
def gen_with_sklearn(n_rows, n_cols, n_classes, cat_frac=0.2, random_state=0):
    print(f"[generate] sklearn make_classification rows={n_rows} cols={n_cols} classes={n_classes}", flush=True)
    from sklearn.datasets import make_classification
    n_informative = max(2, int(0.6 * n_cols))
    X, y = make_classification(
        n_samples=n_rows, n_features=n_cols, n_informative=n_informative,
        n_redundant=max(0, int(0.2*n_cols)), n_repeated=0, n_classes=n_classes,
        random_state=random_state, shuffle=True
    )
    df = pd.DataFrame(X, columns=[f"num_{i}" for i in range(n_cols)])
    # 注入一批类别特征
    n_cat = max(0, int(cat_frac * n_cols))
    rng = np.random.RandomState(random_state)
    for j in range(n_cat):
        src = df[f"num_{j}"]
        bins = np.unique(np.quantile(src, q=np.linspace(0,1,num=6)))
        cats = pd.cut(src, bins=bins, include_lowest=True, labels=False)
        df[f"cat_{j}"] = ("C" + (cats.fillna(0).astype(int) % 10).astype(str))
        miss_idx = rng.choice(len(df), size=int(0.01*len(df)), replace=False)
        df.loc[miss_idx, f"cat_{j}"] = np.nan
    # 日期字符串列
    df["date_str"] = pd.Timestamp("2021-01-01") + pd.to_timedelta(rng.randint(0, 365*2, size=len(df)), unit="D")
    df["date_str"] = df["date_str"].dt.strftime("%Y-%m-%d")
    print("[generate] sklearn done", flush=True)
    return df, pd.Series(y, name="label")

# ----------------------------
# 带进度的 CSV 写入（按块）
# ----------------------------
def save_csv_with_progress(df: pd.DataFrame, path: str, prog: ProgressState, kind: str, chunk_rows: int = 200000):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    total = len(df)
    prog.update(phase="save", save_kind=kind, save_total=total, save_done=0)
    print(f"[save] writing {kind}.csv -> {path} (rows={total}, chunk={chunk_rows})", flush=True)
    if total == 0:
        # 空表也写 header
        df.head(0).to_csv(path, index=False)
        return
    if chunk_rows <= 0 or total <= chunk_rows:
        df.to_csv(path, index=False)
        prog.update(save_done=total)
        print(f"[save] {kind}.csv done ({total}/{total})", flush=True)
        return
    # 分块写入（首块含表头，后续 append）
    written = 0
    # 先写 header
    df.iloc[0:0].to_csv(path, index=False)
    for i in range(0, total, chunk_rows):
        j = min(i + chunk_rows, total)
        df.iloc[i:j].to_csv(path, index=False, mode="a", header=False)
        written = j
        prog.update(save_done=written)
        pct = 100.0 * written / max(1, total)
        print(f"[save] {kind}.csv {written}/{total} ({pct:.1f}%)", flush=True)

# ----------------------------
# 主流程
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--n_datasets", type=int, default=3)
    ap.add_argument("--rows_min", type=int, default=10000)
    ap.add_argument("--rows_max", type=int, default=60000)
    ap.add_argument("--cols_min", type=int, default=20)
    ap.add_argument("--cols_max", type=int, default=100)
    ap.add_argument("--n_classes_max", type=int, default=10)
    ap.add_argument("--prior_type", type=str, default="mix_scm", choices=["mix_scm","mlp_scm","tree_scm"])
    ap.add_argument("--train_ratio", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)

    # 新增参数：心跳与写盘块大小
    ap.add_argument("--hb_interval", type=float, default=1.0, help="心跳输出间隔（秒）；<=0 等效 1 秒")
    ap.add_argument("--quiet", action="store_true", help="关闭心跳输出（阶段进度仍会打印）")
    ap.add_argument("--save_chunk_rows", type=int, default=200000, help="写 CSV 的分块行数；<=0 表示一次性写入")

    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    rng = np.random.RandomState(args.seed)

    # 尝试加载 TabICL 的 prior 生成器
    prior_mod = try_import_tabicl_prior()
    using_tabicl_prior = prior_mod is not None
    if using_tabicl_prior:
        print("[prior] tabicl.prior.dataset 导入成功，将尝试使用其生成器。", flush=True)
    else:
        print("[prior] 未找到 tabicl.prior.dataset，退回 sklearn 合成器。", flush=True)

    prog = ProgressState(total_sets=args.n_datasets, hb_interval=args.hb_interval, quiet=args.quiet)
    prog.start_heartbeat()

    meta = {"using_tabicl_prior": using_tabicl_prior, "datasets": []}

    try:
        for k in range(args.n_datasets):
            rows = int(rng.randint(args.rows_min, args.rows_max+1))
            cols = int(rng.randint(args.cols_min, args.cols_max+1))
            n_classes = int(rng.randint(2, args.n_classes_max+1))

            subdir = os.path.join(args.out_dir, f"set_{k:02d}")
            os.makedirs(subdir, exist_ok=True)

            prog.update(cur_set=k, phase="generate", rows=rows, cols=cols,
                        using_prior=using_tabicl_prior, subdir=subdir)

            # 路径 A：使用 tabicl.prior.dataset
            if using_tabicl_prior:
                cfg = dict(
                    min_features=cols, max_features=cols,
                    max_classes=n_classes,
                    min_seq_len=rows, max_seq_len=rows+1,
                    batch_size=1, batch_size_per_gp=1,
                    log_seq_len=False, seq_len_per_gp=False, replay_small=False,
                    min_train_size=int(args.train_ratio*rows),
                    max_train_size=int(args.train_ratio*rows),
                    prior_type=args.prior_type,
                    fixed_hp={}, sampled_hp={},
                    n_jobs=1, num_threads_per_generate=0, device="cpu"
                )
                ds = None
                for fn_name in ["generate_datasets","generate_batch","make_batch","sample_batch","generate"]:
                    if hasattr(prior_mod, fn_name):
                        fn = getattr(prior_mod, fn_name)
                        try:
                            print(f"[prior] calling {fn_name}(**cfg)", flush=True)
                            ds = fn(**cfg)
                            break
                        except TypeError:
                            try:
                                print(f"[prior] calling {fn_name}(cfg)  # alt", flush=True)
                                ds = fn(cfg)
                                break
                            except Exception as e:
                                print(f"[prior] {fn_name} failed: {e}", flush=True)
                        except Exception as e:
                            print(f"[prior] {fn_name} failed: {e}", flush=True)
                if ds is None:
                    print("[prior] 未能成功调用 prior 生成器，切换到 sklearn 合成器。", flush=True)
                    using_tabicl_prior = False  # 后续也退回 sklearn
                    prog.update(using_prior=False)
                    df, y = gen_with_sklearn(rows, cols, n_classes, random_state=args.seed+k)
                else:
                    item = ds[0] if isinstance(ds, (list, tuple)) else ds
                    X = None; y = None
                    if isinstance(item, dict):
                        X = item.get("X") or item.get("features") or item.get("data")
                        y = item.get("y") or item.get("labels") or item.get("target")
                    if X is None:
                        raise RuntimeError("prior 生成器返回的数据无法解析出 X/y")
                    if isinstance(X, np.ndarray):
                        cols_names = [f"f_{i}" for i in range(X.shape[1])]
                        df = pd.DataFrame(X, columns=cols_names)
                    else:
                        df = pd.DataFrame(X)
                    if not isinstance(y, pd.Series):
                        y = pd.Series(np.asarray(y).reshape(-1), name="label")
                    print("[prior] 生成完成。", flush=True)
            else:
                # 路径 B：sklearn
                df, y = gen_with_sklearn(rows, cols, n_classes, random_state=args.seed+k)

            # 数值列缩放到 0-999（带逐列进度）
            non_num_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
            df_scaled = minmax_scale_0_999(df, ignore_cols=non_num_cols, prog=prog)

            # 切分 & 保存（带写盘进度）
            n_total = len(df_scaled)
            n_train = int(round(args.train_ratio * n_total))
            idx = np.arange(n_total)
            np.random.shuffle(idx)

            train_idx = idx[:n_train]
            test_idx  = idx[n_train:]

            df_train = df_scaled.iloc[train_idx].copy()
            df_train["label"] = y.iloc[train_idx].values

            df_test = df_scaled.iloc[test_idx].copy()
            df_test["label"] = y.iloc[test_idx].values

            save_csv_with_progress(df_train, os.path.join(subdir, "train.csv"),
                                   prog, kind="train", chunk_rows=args.save_chunk_rows)
            save_csv_with_progress(df_test,  os.path.join(subdir, "test.csv"),
                                   prog, kind="test",  chunk_rows=args.save_chunk_rows)

            # 记录 meta
            meta["datasets"].append({
                "path": subdir, "n_rows": int(n_total), "n_cols": int(df_scaled.shape[1]),
                "n_train": int(len(df_train)), "n_test": int(len(df_test)),
                "n_classes": int(pd.Series(y).nunique()),
                "using_prior": bool(using_tabicl_prior)
            })

        # 写 meta.json
        prog.update(phase="meta", subdir=args.out_dir)
        meta_path = os.path.join(args.out_dir, "meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"[meta] wrote {meta_path}", flush=True)

        prog.update(phase="done")
        print(json.dumps(meta, indent=2, ensure_ascii=False))

    finally:
        prog.stop_heartbeat()

if __name__ == "__main__":
    main()
