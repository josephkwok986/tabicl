#!/usr/bin/env python
# infer_tabicl.py  （进度 & ETA + 本地 checkpoint 版）
import argparse, os, time, json, sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, classification_report
from tabicl import TabICLClassifier

try:
    import torch
except Exception:
    torch = None

def parse_args():
    p = argparse.ArgumentParser(description="TabICL inference with local checkpoint + progress & ETA")
    p.add_argument("--train_csv", type=str, required=True)
    p.add_argument("--test_csv", type=str, default=None)
    p.add_argument("--target", type=str, required=True)
    p.add_argument("--test_size", type=float, default=0.2)

    # ==== 变化 1：增加本地 checkpoint 参数 ====
    # NEW: 明确本地文件路径；优先使用此路径加载
    p.add_argument("--checkpoint_path", type=str, default=None, help="本地 .ckpt 路径，优先使用")
    # CHANGED: 原先的 --checkpoint 表示“版本名”，保留作回退（不提供路径时才会用）
    p.add_argument("--checkpoint", type=str, default="tabicl-classifier-v1.1-0506.ckpt",
                   help="checkpoint 版本名（仅在未提供 --checkpoint_path 时生效）")

    p.add_argument("--n_estimators", type=int, default=32)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--amp", type=int, default=1)
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--n_jobs", type=int, default=None)
    p.add_argument("--save_proba", action="store_true")
    p.add_argument("--out_dir", type=str, default="runs/tabicl_infer")
    p.add_argument("--verbose", action="true", help=argparse.SUPPRESS) if False else p.add_argument("--verbose", action="store_true")
    p.add_argument("--progress_chunk_rows", type=int, default=20000, help="每次预测多少行后汇报一次进度；0=整表一次性")
    return p.parse_args()

def save_progress_row(fp, ts, done, total, thr, eta_s):
    fp.write(f"{ts},{done},{total},{thr:.6f},{eta_s:.3f}\n")
    fp.flush()

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # 环境记录
    env = {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
           "torch_cuda_available": bool(torch and torch.cuda.is_available())}
    if torch and torch.cuda.is_available():
        env.update({"cuda_device_count": torch.cuda.device_count(),
                    "cuda_device_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]})
    with open(os.path.join(args.out_dir, "env.json"), "w") as f: json.dump(env, f, indent=2, ensure_ascii=False)

    # 读取数据
    t0 = time.perf_counter()
    df_train = pd.read_csv(args.train_csv)
    assert args.target in df_train.columns, f"target 列 {args.target} 不在训练集里"
    X_train = df_train.drop(columns=[args.target]); y_train = df_train[args.target]

    if args.test_csv:
        df_test = pd.read_csv(args.test_csv)
        if args.target in df_test.columns:
            X_test = df_test.drop(columns=[args.target]); y_test = df_test[args.target]
        else:
            X_test = df_test; y_test = None
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X_train, y_train, test_size=args.test_size,
            stratify=y_train if y_train.nunique() > 1 else None, random_state=42
        )
    read_secs = time.perf_counter() - t0

    # ==== 变化 2：构建 TabICLClassifier 时的本地加载逻辑 ====
    use_local_ckpt = False
    model_path = None
    allow_auto_download = False  # NEW: 默认禁止自动下载，除非未给路径

    if args.checkpoint_path:
        if not os.path.isfile(args.checkpoint_path):
            raise FileNotFoundError(f"未找到本地 checkpoint: {args.checkpoint_path}")
        model_path = os.path.abspath(args.checkpoint_path)
        use_local_ckpt = True
    else:
        # 没给路径：退回按“版本名”从 Hub 加载，并允许自动下载
        allow_auto_download = True

    clf = TabICLClassifier(
        n_estimators=args.n_estimators,
        batch_size=args.batch_size,
        use_amp=bool(args.amp),
        # NEW/CHANGED:
        model_path=model_path,                    # <—— 若给了本地路径，将从这里读取
        allow_auto_download=allow_auto_download,  # <—— 禁止/允许自动下载
        checkpoint_version=args.checkpoint,       # <—— 未给路径时用于指定 Hub 上的版本
        device=args.device,
        n_jobs=args.n_jobs,
        verbose=args.verbose,
    )

    # 轻训练（fit）
    t0 = time.perf_counter()
    clf.fit(X_train, y_train)
    fit_secs = time.perf_counter() - t0

    # 推理（带进度）
    n_total = len(X_test)
    preds = []
    prob_list = [] if args.save_proba else None
    prog_path = os.path.join(args.out_dir, "progress.csv")
    with open(prog_path, "w") as fp:
        fp.write("ts,done,total,rows_per_sec,eta_sec\n")
        t_start = time.perf_counter()
        if args.progress_chunk_rows and args.progress_chunk_rows > 0:
            chunk = args.progress_chunk_rows
            done = 0
            print(f"[infer] start: total={n_total} rows, chunk={chunk}")
            for i in range(0, n_total, chunk):
                X_chunk = X_test.iloc[i:min(i+chunk, n_total)]
                y_pred_chunk = clf.predict(X_chunk)
                if prob_list is not None:
                    try:
                        y_prob_chunk = clf.predict_proba(X_chunk)
                    except Exception as e:
                        y_prob_chunk = None
                        if i == 0:
                            print(f"[warn] predict_proba 失败：{e}", file=sys.stderr)
                    if y_prob_chunk is not None:
                        prob_list.append(y_prob_chunk)
                preds.append(pd.Series(y_pred_chunk, index=X_chunk.index))
                done += len(X_chunk)
                # 进度
                elapsed = time.perf_counter() - t_start
                thr = done / max(elapsed, 1e-9)
                eta_s = (n_total - done) / max(thr, 1e-9)
                ts = time.strftime("%Y-%m-%dT%H:%M:%S")
                save_progress_row(fp, ts, done, n_total, thr, eta_s)
                print(f"[infer] {done}/{n_total} done | thr={thr:.1f} rows/s | ETA={eta_s/60:.1f} min")
            y_pred = pd.concat(preds).sort_index().values
            y_prob = (np.concatenate(prob_list, axis=0) if (prob_list and len(prob_list)>0) else None)
        else:
            print(f"[infer] start (single-shot): total={n_total} rows")
            y_pred = clf.predict(X_test)
            y_prob = None
            if args.save_proba:
                try:
                    y_prob = clf.predict_proba(X_test)
                except Exception as e:
                    print(f"[warn] predict_proba 失败：{e}", file=sys.stderr)
            done = n_total
            elapsed = time.perf_counter() - t_start
            thr = done / max(elapsed, 1e-9)
            ts = time.strftime("%Y-%m-%dT%H:%M:%S")
            save_progress_row(fp, ts, done, n_total, thr, 0.0)
            print(f"[infer] finished: {done}/{n_total} | thr={thr:.1f} rows/s")

    infer_secs = (time.perf_counter() - t_start)
    thr = n_total / max(infer_secs, 1e-9)

    # 指标
    metrics = {
        "n_train": int(len(X_train)), "n_test": int(n_total),
        "n_features": int(X_train.shape[1]), "n_classes": int(y_train.nunique()),
        "read_secs": read_secs, "fit_secs": fit_secs,
        "infer_secs": infer_secs, "throughput_rows_per_sec": thr,
        # NEW: 记录 checkpoint 来源
        "checkpoint_source": "local" if use_local_ckpt else "hub",
        "checkpoint_path": model_path if use_local_ckpt else "",
        "checkpoint_version": args.checkpoint,
        "n_estimators": args.n_estimators, "batch_size": args.batch_size,
        "use_amp": bool(args.amp), "device": args.device or "auto",
        "progress_chunk_rows": args.progress_chunk_rows
    }
    if y_train.nunique() > 1 and isinstance(y_test, pd.Series):
        try:
            metrics.update({
                "acc": float(accuracy_score(y_test, y_pred)),
                "balanced_acc": float(balanced_accuracy_score(y_test, y_pred)),
                "f1_macro": float(f1_score(y_test, y_pred, average="macro"))
            })
        except Exception as e:
            print(f"[warn] 指标计算失败：{e}", file=sys.stderr)

    # 保存
    out_pred = pd.DataFrame({"pred": y_pred})
    if isinstance(y_test, pd.Series): out_pred.insert(0, "y_true", y_test.values)
    out_pred.to_csv(os.path.join(args.out_dir, "predictions.csv"), index=False)
    if 'y_prob' in locals() and y_prob is not None:
        np.save(os.path.join(args.out_dir, "proba.npy"), y_prob)
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print("\n==== Inference Summary ====")
    for k,v in metrics.items(): print(f"{k}: {v}")
    if isinstance(y_test, pd.Series):
        try:
            print("\nClassification report (head):")
            print(classification_report(y_test, y_pred)[:1000])
        except Exception:
            pass

if __name__ == "__main__":
    main()
