# pip install -U tabicl tqdm scikit-learn pandas pyyaml

import os
import sys
import math
import yaml
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from tabicl import TabICLClassifier


DEFAULT_CFG = {
    "train_path": "train.csv",
    "test_path": "test.csv",
    "ckpt_path": "tabicl-classifier-v1.1-0506.ckpt",
    "ICL_MAX_SAMPLES": 4000,      # None/0 使用全部训练样本作为ICL支持样本
    "ICL_SAMPLE_RANDOM": True,    # True 随机抽样；False 取前N
    "SEED": 42,
    "submission_path": "submission.csv",
}


def _expand(p):
    return None if p is None else os.path.abspath(os.path.expanduser(p))


def load_config(yaml_path):
    # 允许空YAML或缺失字段，使用默认值补全
    with open(yaml_path, "r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    cfg = DEFAULT_CFG.copy()
    cfg.update(user_cfg)

    # 规范化路径
    cfg["train_path"] = _expand(cfg["train_path"])
    cfg["test_path"] = _expand(cfg["test_path"])
    cfg["ckpt_path"] = _expand(cfg["ckpt_path"])
    cfg["submission_path"] = _expand(cfg["submission_path"])

    return cfg


def select_support(X, y, n_max, random_pick=True, seed=42):
    """从给定训练子集中选 ICL 支持样本，返回 X_sel, y_sel"""
    if not n_max or n_max <= 0 or n_max >= len(X):
        return X, y
    if random_pick:
        idx = X.sample(n=n_max, random_state=seed, replace=False).index
    else:
        idx = X.index[:n_max]
    return X.loc[idx], y.loc[idx]


def get_acc(X, y, train_df, ICL_MAX_SAMPLES, ICL_SAMPLE_RANDOM, SEED, ckpt_path):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold_acc, oof_pred = [], pd.Series(index=train_df.index, dtype=object)

    outer = tqdm(total=5, desc="CV folds", ncols=100)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        X_tr, y_tr = X.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = X.iloc[va_idx], y.iloc[va_idx]

        # 从本折训练子集抽取 ICL 支持样本
        X_sup, y_sup = select_support(X_tr, y_tr, ICL_MAX_SAMPLES, ICL_SAMPLE_RANDOM, SEED + fold)

        clf = TabICLClassifier(
            model_path=ckpt_path,
            allow_auto_download=False,
            use_amp=True,
            random_state=SEED,
        )
        clf.fit(X_sup, y_sup)

        # 进度条：≥1% 粒度
        n = len(X_va)
        step = max(1, math.ceil(n / 100))
        preds = []
        inner = tqdm(total=n, desc=f"Fold {fold} infer", leave=False, ncols=100)
        for s in range(0, n, step):
            e = min(s + step, n)
            preds.extend(clf.predict(X_va.iloc[s:e]))
            inner.update(e - s)
        inner.close()

        preds = pd.Series(preds, index=va_idx)
        oof_pred.loc[va_idx] = preds
        fold_acc.append(accuracy_score(y_va, preds))
        outer.update(1)
    outer.close()

    print("Fold Accuracies:", [round(a, 6) for a in fold_acc])
    print(f"CV Accuracy (mean±std): {pd.Series(fold_acc).mean():.6f} ± {pd.Series(fold_acc).std():.6f}")
    print(f"OOF Accuracy: {accuracy_score(y, oof_pred):.6f}")


def get_submission(X, y, X_test, test_df, id_col, label_col, ICL_MAX_SAMPLES, ICL_SAMPLE_RANDOM, SEED, ckpt_path, submission_path):
    X_sup_full, y_sup_full = select_support(X, y, ICL_MAX_SAMPLES, ICL_SAMPLE_RANDOM, SEED)

    print('X_sup_full.shape', X_sup_full.shape)
    print('y_sup_full.shape', y_sup_full.shape)

    final_clf = TabICLClassifier(
        model_path=ckpt_path,
        allow_auto_download=False,
        use_amp=True,
        random_state=SEED,
    )
    final_clf.fit(X_sup_full, y_sup_full)

    n_test = len(X_test)
    step = max(1, math.ceil(n_test / 100))
    test_preds = []
    pbar = tqdm(total=n_test, desc="Test infer", ncols=100)
    for s in range(0, n_test, step):
        e = min(s + step, n_test)
        test_preds.extend(final_clf.predict(X_test.iloc[s:e]))
        pbar.update(e - s)
    pbar.close()

    sub = pd.DataFrame({id_col: test_df[id_col], label_col: test_preds})
    os.makedirs(os.path.dirname(submission_path) or ".", exist_ok=True)
    sub.to_csv(submission_path, index=False)
    print(f"Saved: {submission_path}")


def main():
    # 仅接受一个命令行参数：YAML 配置路径
    if len(sys.argv) != 2:
        print("用法: python main.py <config.yaml>")
        sys.exit(2)
    cfg_path = sys.argv[1]
    if not os.path.exists(cfg_path):
        print(f"未找到配置文件: {cfg_path}")
        sys.exit(2)

    cfg = load_config(cfg_path)

    train_path = cfg["train_path"]
    test_path = cfg["test_path"]
    ckpt_path = cfg["ckpt_path"]
    ICL_MAX_SAMPLES = cfg["ICL_MAX_SAMPLES"]
    ICL_SAMPLE_RANDOM = cfg["ICL_SAMPLE_RANDOM"]
    SEED = cfg["SEED"]
    NEED_GET_ACC = cfg["NEED_GET_ACC"]
    NEED_GET_SUBMISSION = cfg["NEED_GET_SUBMISSION"]
    submission_path = cfg["submission_path"]
    LABEL_COL = cfg["LABEL_COL"]

    assert os.path.exists(train_path), "缺少 train.csv"
    assert os.path.exists(test_path), "缺少 test.csv"
    assert os.path.exists(ckpt_path), "ckpt 路径无效"

    np.random.seed(SEED)

    # ===== 读取数据 =====
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    label_col = LABEL_COL
    id_candidates = [c for c in test.columns if c.lower() in ("id", "row_id")]
    id_col = id_candidates[0] if id_candidates else "id"
    if id_col not in test.columns:
        test[id_col] = range(len(test))

    drop_cols = [label_col] + [c for c in train.columns if c.lower() in ("id", "row_id")]
    X = train.drop(columns=drop_cols, errors="ignore")
    y = train[label_col]
    X_test = test.drop(columns=[id_col], errors="ignore")

    if NEED_GET_ACC:
        get_acc(
            X, y, train,
            ICL_MAX_SAMPLES=ICL_MAX_SAMPLES,
            ICL_SAMPLE_RANDOM=ICL_SAMPLE_RANDOM,
            SEED=SEED,
            ckpt_path=ckpt_path,
        )

    if NEED_GET_SUBMISSION:
        get_submission(
            X, y, X_test, test,
            id_col=id_col, label_col=label_col,
            ICL_MAX_SAMPLES=ICL_MAX_SAMPLES,
            ICL_SAMPLE_RANDOM=ICL_SAMPLE_RANDOM,
            SEED=SEED,
            ckpt_path=ckpt_path,
            submission_path=submission_path,
        )


if __name__ == "__main__":
    main()
