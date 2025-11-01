'''
python mycmd/slipt_csv_set/check/check_split.py --data-root mycmd/slipt_csv_set/test_out

python mycmd/slipt_csv_set/check/check_split.py --data-root "/workspace/Gjj Local/data/fusion-360-gallery-dataset-s2.0.0/processed/embeddings_brepnet_csv_step_1"
'''

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def iter_csv_rows(csv_path: Path) -> Iterable[Tuple[str, str, str]]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 5:
                continue
            label = row[0]
            source_path = row[1]
            face_index = row[2]
            yield label, source_path, face_index


def find_leaks(root: Path) -> Dict[str, List[Path]]:
    datasets = ["training_set", "validation_set", "test_set"]
    part_map: Dict[str, str] = {}
    leaks: Dict[str, List[Path]] = defaultdict(list)

    for dataset in datasets:
        dataset_dir = root / dataset
        if not dataset_dir.is_dir():
            continue
        for csv_path in dataset_dir.glob("*.csv"):
            for _, source_path, _ in iter_csv_rows(csv_path):
                key = source_path
                if key in part_map and part_map[key] != dataset:
                    leaks[key].append(csv_path)
                else:
                    part_map[key] = dataset
    return leaks


def main() -> None:
    parser = argparse.ArgumentParser(description="检测数据集划分是否泄露。")
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="包含 training_set/validation_set/test_set 的目录。",
    )
    args = parser.parse_args()
    leaks = find_leaks(args.data_root)
    if not leaks:
        print("未发现跨数据集的零件或设计泄露。")
        return
    print("发现以下泄露：")
    for key, paths in leaks.items():
        datasets = {path.parent.name for path in paths}
        print(f"- {key} 同时出现在: {', '.join(sorted(datasets))}")
        for path in paths:
            print(f"  * {path}")


if __name__ == "__main__":
    main()

