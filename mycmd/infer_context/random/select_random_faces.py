"""
python mycmd/infer_context/random/select_random_faces.py \
    --input-root mycmd/slipt_csv_set/test_out/training_set \
    --output-dir mycmd/infer_context/random/test_out \
    --sample-size 5 \
    --seed 42

python mycmd/infer_context/random/select_random_faces.py \
    --input-root "/workspace/Gjj Local/data/fusion-360-gallery-dataset-s2.0.0/processed/embeddings_brepnet_csv_step_1/training_set" \
    --output-dir "/workspace/Gjj Local/data/fusion-360-gallery-dataset-s2.0.0/processed/tabicl_context/random" \
    --sample-size 2000 \
    --seed 42

脚本会递归扫描 ``--input-root`` 指定目录下的全部 ``*.csv`` 文件并随机抽样。
"""

from __future__ import annotations

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional


@dataclass(frozen=True)
class FaceRecord:
    source_file: Path
    row: List[str]


def iter_faces(csv_path: Path) -> Iterable[FaceRecord]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            yield FaceRecord(source_file=csv_path, row=row)


def gather_faces(root: Path) -> List[FaceRecord]:
    records: List[FaceRecord] = []
    for csv_path in sorted(root.rglob("*.csv")):
        if not csv_path.is_file():
            continue
        records.extend(iter_faces(csv_path))
    return records


def sample_faces(records: List[FaceRecord], k: int, seed: Optional[int]) -> List[FaceRecord]:
    rng = random.Random(seed)
    if k >= len(records):
        return records
    return rng.sample(records, k)


def write_sample(output_dir: Path, sampled: List[FaceRecord]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "sampled_faces.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        for record in sampled:
            row = record.row
            prefix = row[:5]  # label, source_rel_path, face_index, design_id, part_uid
            embeddings = row[5:]
            writer.writerow(prefix + embeddings)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="随机抽取指定数量的面级嵌入样本。")
    parser.add_argument(
        "--input-root",
        type=Path,
        required=True,
        help="包含 training_set/validation_set/test_set 的根目录。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="抽样结果输出目录。",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=32,
        help="抽样数量 K，默认 32。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="随机种子，便于复现，缺省为 true 随机。",
    )
    args = parser.parse_args()

    input_root = args.input_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    records = gather_faces(input_root)
    if not records:
        raise RuntimeError(f"在 {input_root} 下未发现任何 CSV 面数据")

    sampled = sample_faces(records, args.sample_size, args.seed)
    output_path = write_sample(output_dir, sampled)
    print(f"抽样完成，共 {len(sampled)} 行 -> {output_path}")


if __name__ == "__main__":
    main()
