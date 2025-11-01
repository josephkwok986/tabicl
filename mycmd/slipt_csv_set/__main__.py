from __future__ import annotations

import argparse
import os
from pathlib import Path

from .job import CsvSplitService


DEFAULT_INPUT_DIR = Path(
    "/workspace/Gjj Local/data/fusion-360-gallery-dataset-s2.0.0/processed/embeddings_brepnet_csv"
)
DEFAULT_DATASET_JSON = Path(
    "/workspace/Gjj Local/data/fusion-360-gallery-dataset-s2.0.0/processed/dataset.json"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/workspace/Gjj Local/data/fusion-360-gallery-dataset-s2.0.0/processed/embeddings_brepnet_csv_step_1"
)
DEFAULT_QUEUE_ROOT = Path(__file__).with_name("runtime") / "queue"
DEFAULT_CONFIG = Path(__file__).with_name("task_config.yaml")


def _ensure_config() -> None:
    os.environ.setdefault("CAD_TASK_CONFIG", str(DEFAULT_CONFIG))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按数据集划分 BRepNet 嵌入 CSV。")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="原始 CSV 所在目录。",
    )
    parser.add_argument(
        "--dataset-json",
        type=Path,
        default=DEFAULT_DATASET_JSON,
        help="包含 train/val/test 划分的 dataset.json。",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="输出根目录，子目录为 training_set/validation_set/test_set。",
    )
    parser.add_argument(
        "--queue-root",
        type=Path,
        default=DEFAULT_QUEUE_ROOT,
        help="持久化任务队列目录。",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="入队 TaskRecord 时的批大小。",
    )
    return parser.parse_args()


def main() -> None:
    _ensure_config()
    args = parse_args()
    service = CsvSplitService(
        input_dir=args.input_dir,
        dataset_json=args.dataset_json,
        output_root=args.output_root,
        queue_root=args.queue_root,
        batch_size=args.batch_size,
    )
    raise SystemExit(service.run())


if __name__ == "__main__":
    main()
