#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# gen_prior.py — Offline prior generator for TabICL Stage-3-like training
# Reference ranges: up to 60K rows, up to 100 features, up to 10 classes; synthetic prior. (TabICL paper)
# https://arxiv.org/abs/2502.05564

import os, json, time, argparse
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np

@dataclass
class PriorTableSpec:
    n_rows: int
    n_features: int
    n_classes: int
    frac_categorical: float

@dataclass
class PriorGenConfig:
    output_dir: str
    num_tables: int
    min_rows: int = 40000
    max_rows: int = 60000
    min_features: int = 20
    max_features: int = 100
    min_classes: int = 2
    max_classes: int = 10
    frac_categorical: float = 0.15
    dtype: str = "float16"  # "float16" or "float32"
    seed: int = 42

def _make_latents(n_rows: int, latent_dim: int, rng: np.random.Generator) -> np.ndarray:
    return rng.normal(size=(n_rows, latent_dim)).astype(np.float32)

def _nonlinear_mix(Z: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    if n <= 0:
        return np.empty((Z.shape[0], 0), dtype=np.float32)
    m = Z.shape[1]
    W = rng.normal(size=(m, n)).astype(np.float32)
    B = rng.normal(scale=0.5, size=(n,)).astype(np.float32)
    A = Z @ W + B
    n1 = n // 3
    X1 = np.tanh(A[:, :n1])
    X2 = np.sin(A[:, n1:2*n1])
    X3 = np.maximum(A[:, 2*n1:], 0.0)
    out = np.concatenate([X1, X2, X3], axis=1)
    out += rng.normal(scale=0.03, size=out.shape).astype(out.dtype)
    return out

def _make_categorical(Z: np.ndarray, n_cat_cols: int, n_classes_per_col: Tuple[int, int], rng: np.random.Generator):
    if n_cat_cols == 0:
        return np.empty((Z.shape[0], 0), dtype=np.int32), []
    m = Z.shape[1]
    W = rng.normal(size=(m, n_cat_cols)).astype(np.float32)
    A = Z @ W
    cats_meta = []
    cat_cols = []
    for j in range(n_cat_cols):
        k = int(rng.integers(n_classes_per_col[0], n_classes_per_col[1] + 1))
        edges = np.quantile(A[:, j], np.linspace(0, 1, k + 1)[1:-1])
        col = np.digitize(A[:, j], edges).astype(np.int32)
        vocab = [f"C{j}_{i}" for i in range(k)]
        cats_meta.append(vocab)
        cat_cols.append(col.reshape(-1, 1))
    return np.concatenate(cat_cols, axis=1), cats_meta

def _assign_labels(Z: np.ndarray, n_classes: int, rng: np.random.Generator) -> np.ndarray:
    centers = rng.normal(size=(n_classes, Z.shape[1])).astype(np.float32)
    Z_nl = np.tanh(Z + 0.15 * np.sin(Z))
    Z_norm = Z_nl / (np.linalg.norm(Z_nl, axis=1, keepdims=True) + 1e-8)
    C_norm = centers / (np.linalg.norm(centers, axis=1, keepdims=True) + 1e-8)
    sims = Z_norm @ C_norm.T
    logits = sims / 0.3
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    y = np.array([rng.choice(np.arange(n_classes), p=p) for p in probs], dtype=np.int64)
    return y

def generate_one(spec: PriorTableSpec, dtype: str, rng: np.random.Generator):
    latent_dim = int(np.clip(spec.n_features // 4, 8, 16))
    Z = _make_latents(spec.n_rows, latent_dim, rng)
    n_cat = int(round(spec.n_features * spec.frac_categorical))
    n_num = spec.n_features - n_cat
    X_num = _nonlinear_mix(Z, n_num, rng)
    X_cat, cats_meta = _make_categorical(Z, n_cat, (3, 12), rng)
    X_cat_f = X_cat.astype(np.float32) if X_cat.size else np.empty((spec.n_rows, 0), dtype=np.float32)
    X = np.concatenate([X_num, X_cat_f], axis=1)
    if X_num.size:
        mu = X_num.mean(axis=0, keepdims=True)
        std = X_num.std(axis=0, keepdims=True) + 1e-6
        X[:, :X_num.shape[1]] = (X[:, :X_num.shape[1]] - mu) / std
    y = _assign_labels(Z, spec.n_classes, rng)
    feat_types = (["num"] * n_num) + (["cat"] * n_cat)
    if dtype == "float16":
        X = X.astype(np.float16, copy=False)
    elif dtype == "float32":
        X = X.astype(np.float32, copy=False)
    else:
        raise ValueError("dtype must be float16 or float32")
    return X, y, feat_types, cats_meta

def _save_npz_table(path_npz: str, X: np.ndarray, y: np.ndarray, feat_types: List[str], cats_meta, meta: dict):
    os.makedirs(os.path.dirname(path_npz), exist_ok=True)
    np.savez_compressed(path_npz, X=X, y=y)
    meta_path = path_npz.replace(".npz", ".json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump({**meta, "feature_types": feat_types, "categorical_vocabs": cats_meta}, f, indent=2)

def generate_prior(cfg: PriorGenConfig) -> dict:
    os.makedirs(cfg.output_dir, exist_ok=True)
    rng = np.random.default_rng(cfg.seed)
    manifest = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "seed": cfg.seed,
        "num_tables": cfg.num_tables,
        "dtype": cfg.dtype,
        "min_rows": cfg.min_rows, "max_rows": cfg.max_rows,
        "min_features": cfg.min_features, "max_features": cfg.max_features,
        "min_classes": cfg.min_classes, "max_classes": cfg.max_classes,
        "frac_categorical": cfg.frac_categorical,
        "tables": []
    }
    for i in range(cfg.num_tables):
        n_rows = int(rng.integers(cfg.min_rows, cfg.max_rows + 1))
        n_features = int(rng.integers(cfg.min_features, cfg.max_features + 1))
        n_classes = int(rng.integers(cfg.min_classes, cfg.max_classes + 1))
        spec = PriorTableSpec(n_rows, n_features, n_classes, cfg.frac_categorical)
        X, y, feat_types, cats_meta = generate_one(spec, cfg.dtype, rng)
        fname = f"table_{i:05d}.npz"
        p = os.path.join(cfg.output_dir, fname)
        _save_npz_table(p, X, y, feat_types, cats_meta, meta={
            "n_rows": n_rows, "n_features": n_features, "n_classes": n_classes,
            "frac_categorical": cfg.frac_categorical
        })
        manifest["tables"].append({"file": fname, "n_rows": n_rows, "n_features": n_features, "n_classes": n_classes})
    with open(os.path.join(cfg.output_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest

def main():
    ap = argparse.ArgumentParser(description="Generate offline prior tables for TabICL Stage-3-like training")
    ap.add_argument("--output-dir", type=str, required=True)
    ap.add_argument("--num-tables", type=int, default=60)
    ap.add_argument("--min-rows", type=int, default=40000)
    ap.add_argument("--max-rows", type=int, default=60000)
    ap.add_argument("--min-features", type=int, default=20)
    ap.add_argument("--max-features", type=int, default=100)
    ap.add_argument("--min-classes", type=int, default=2)
    ap.add_argument("--max-classes", type=int, default=10)
    ap.add_argument("--frac-categorical", type=float, default=0.15)
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "float32"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = PriorGenConfig(
        output_dir=args.output_dir,
        num_tables=args.num_tables,
        min_rows=args.min_rows, max_rows=args.max_rows,
        min_features=args.min_features, max_features=args.max_features,
        min_classes=args.min_classes, max_classes=args.max_classes,
        frac_categorical=args.frac_categorical,
        dtype=args.dtype, seed=args.seed
    )
    m = generate_prior(cfg)
    print(json.dumps({"ok": True, "output_dir": args.output_dir, "num_tables": m["num_tables"]}, indent=2))

if __name__ == "__main__":
    main()
