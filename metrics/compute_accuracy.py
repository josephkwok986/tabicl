#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute micro-averaged Accuracy across all CSV files in a directory.
Usage example:
python metrics/compute_accuracy.py --dir data/ --pred-col y_pred --true-col y_true --dropna


python metrics/compute_accuracy.py --dir "/workspace/Gjj Doc/Code/BRepNet/mycmd/classification_server/infer_server/runtime/results" --pred-col predicted_label --true-col true_label --dropna

Overall N=77070 Correct=70937   Acc=0.920423

python metrics/compute_accuracy.py --dir "/workspace/Gjj Local/data/fusion-360-gallery-dataset-s2.0.0/processed/tabicl_predict" --pred-col prediction --true-col col_0 --dropna

1000上下文
Overall N=77067 Correct=70762   Acc=0.918188
2000上下文
Overall N=6525  Correct=6128    Acc=0.939157
Overall N=8401  Correct=7841    Acc=0.933341
Overall N=8420  Correct=7860    Acc=0.933492
Overall N=8732  Correct=8170    Acc=0.935639
Overall N=9050  Correct=8455    Acc=0.934254
Overall N=10270 Correct=9606    Acc=0.935346
Overall N=10728 Correct=10038   Acc=0.935682
Overall N=14498 Correct=13506   Acc=0.931577
Overall N=15406 Correct=14313   Acc=0.929054
Overall N=18776 Correct=17355   Acc=0.924318
Overall N=20689 Correct=19097   Acc=0.923051
Overall N=22421 Correct=20717   Acc=0.924000
Overall N=29321 Correct=27080   Acc=0.923570

"""
import argparse
import glob
import os
import sys
import pandas as pd

def normalize_series(s: pd.Series, mode: str) -> pd.Series:
    s = s.astype(str)
    if mode in ("strip", "lower_strip"):
        s = s.str.strip()
    if mode in ("lower", "lower_strip"):
        s = s.str.lower()
    return s

def main():
    ap = argparse.ArgumentParser(description="Compute Accuracy over CSVs in a directory")
    ap.add_argument("--dir", required=True, help="Directory containing CSV files")
    ap.add_argument("--pred-col", required=True, help="Predicted label column name")
    ap.add_argument("--true-col", required=True, help="Ground-truth label column name")
    ap.add_argument("--pattern", default="*.csv", help="File pattern. Default: *.csv")
    ap.add_argument("--sep", default=",", help="CSV delimiter. Default: ','")
    ap.add_argument("--encoding", default="utf-8", help="CSV text encoding. Default: utf-8")
    ap.add_argument("--dropna", action="store_true", help="Drop rows with NA in either label column")
    ap.add_argument("--per-file", action="store_true", help="Print per-file Accuracy")
    ap.add_argument("--normalize", choices=["none","strip","lower","lower_strip"], default="none",
                    help="Optional normalization before comparison")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, args.pattern)))
    if not files:
        print(f"No files matched: {os.path.join(args.dir, args.pattern)}", file=sys.stderr)
        sys.exit(2)

    total = 0
    correct = 0
    for f in files:
        try:
            df = pd.read_csv(f, sep=args.sep, encoding=args.encoding)
        except Exception as e:
            print(f"Failed to read {f}: {e}", file=sys.stderr)
            continue

        if args.pred_col not in df.columns or args.true_col not in df.columns:
            print(f"Missing required columns in {f}. "
                  f"Found columns: {list(df.columns)}", file=sys.stderr)
            sys.exit(3)

        cols = df[[args.pred_col, args.true_col]]
        if args.dropna:
            cols = cols.dropna(subset=[args.pred_col, args.true_col])

        n = len(cols)
        if n == 0:
            if args.per_file:
                print(f"{os.path.basename(f)}\tN=0\tAcc=nan")
            continue

        if args.normalize != "none":
            p = normalize_series(cols[args.pred_col], args.normalize)
            t = normalize_series(cols[args.true_col], args.normalize)
        else:
            # Compare as-is but coerce to string to avoid dtype mismatch pitfalls.
            p = cols[args.pred_col].astype(str)
            t = cols[args.true_col].astype(str)

        matches = p.eq(t)
        c = int(matches.sum())

        if args.per_file:
            acc_f = c / n
            print(f"{os.path.basename(f)}\tN={n}\tCorrect={c}\tAcc={acc_f:.6f}")

        total += n
        correct += c

    if total == 0:
        print("No rows across all files after filtering.", file=sys.stderr)
        sys.exit(4)

    acc = correct / total
    print(f"Overall\tN={total}\tCorrect={correct}\tAcc={acc:.6f}")

if __name__ == "__main__":
    main()
