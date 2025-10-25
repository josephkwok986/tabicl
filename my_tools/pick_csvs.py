#!/usr/bin/env python3

'''
随机从指定输入目录里，挑选N个（这是参数） csv 文件，复制到指定输出目录中

python my_tools/pick_csvs.py /path/to/in /path/to/out 10 --recursive --seed 42


python my_tools/pick_csvs.py '/workspace/Gjj Local/data/fusion-360-gallery-dataset-s2.0.0/processed/embeddings_brepnet_csv' '/workspace/Gjj Local/data/fusion-360-gallery-dataset-s2.0.0/processed/tabicl_context' 1000 --recursive --seed 42

'''

#!/usr/bin/env python3
import argparse
import os
import random
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import count
from pathlib import Path


def iter_csv_paths(root: str, recursive: bool):
    stack = [root]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for e in it:
                    if e.is_dir(follow_symlinks=False):
                        if recursive:
                            stack.append(e.path)
                    elif e.is_file(follow_symlinks=False):
                        name = e.name
                        if len(name) >= 4 and name[-4:].lower() == ".csv":
                            yield e.path
        except PermissionError:
            continue


def reservoir_sample_paths(paths_iter, k: int, seed: int | None = None):
    rng = random.Random(seed)
    sample = []
    seen = 0
    for seen, p in enumerate(paths_iter, 1):
        if seen <= k:
            sample.append(p)
        else:
            j = rng.randint(1, seen)
            if j <= k:
                sample[j - 1] = p
    return sample, seen


def unique_dst(out_dir: Path, src_path: str) -> Path:
    name = os.path.basename(src_path)
    stem, ext = os.path.splitext(name)
    dst = out_dir / name
    if not dst.exists():
        return dst
    for i in count(1):
        cand = out_dir / f"{stem}.{i}{ext}"
        if not cand.exists():
            return cand


def copy_one(src: str, dst: Path) -> tuple[str, bool, str | None]:
    try:
        shutil.copyfile(src, dst)
        return (src, True, None)
    except Exception as e:
        return (src, False, str(e))


def fast_clear_dir(path: Path, workers: int = 8):
    path.mkdir(parents=True, exist_ok=True)
    tasks = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        with os.scandir(path) as it:
            for e in it:
                if e.is_dir(follow_symlinks=False):
                    tasks.append(ex.submit(shutil.rmtree, e.path))
                else:
                    tasks.append(ex.submit(os.unlink, e.path))
        for f in as_completed(tasks):
            f.result()


def pick_and_copy(in_dir: Path, out_dir: Path, n: int, recursive: bool,
                  seed: int | None, workers: int):
    if not in_dir.is_dir():
        raise SystemExit(f"输入目录不存在: {in_dir}")

    # 安全检查
    if in_dir.resolve() == out_dir.resolve():
        raise SystemExit("输出目录与输入目录相同，拒绝清空。")

    # 先清空目标目录
    fast_clear_dir(out_dir, workers=max(2, min(32, workers)))

    # 单次遍历+水库抽样
    sample, seen = reservoir_sample_paths(iter_csv_paths(str(in_dir), recursive), n, seed)
    if seen < n:
        raise SystemExit(f"仅找到 {seen} 个 CSV，少于请求 {n}")

    # 并发复制
    results = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = []
        for src in sample:
            dst = unique_dst(out_dir, src)
            futs.append(ex.submit(copy_one, src, dst))
        for f in as_completed(futs):
            results.append(f.result())

    ok = sum(1 for _, s, _ in results if s)
    errs = [(src, err) for src, s, err in results if not s]
    print(f"扫描到 {seen} 个 CSV，已随机复制 {ok}/{n} 个到 {out_dir}")
    if errs:
        print(f"有 {len(errs)} 个失败，例如：{errs[:3]}")


def main():
    ap = argparse.ArgumentParser(description="从大目录中随机挑选 N 个 CSV 并复制（清空目标+水库抽样+并发复制）")
    ap.add_argument("input_dir", type=Path)
    ap.add_argument("output_dir", type=Path)
    ap.add_argument("n", type=int)
    ap.add_argument("--recursive", action="store_true", help="递归子目录")
    ap.add_argument("--seed", type=int, default=None, help="随机种子")
    ap.add_argument("--workers", type=int, default=8, help="并发线程数，I/O 受限建议 4-16")
    args = ap.parse_args()
    pick_and_copy(args.input_dir, args.output_dir, args.n, args.recursive, args.seed, args.workers)


if __name__ == "__main__":
    main()
