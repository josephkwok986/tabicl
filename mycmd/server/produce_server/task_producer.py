#!/usr/bin/env python3
"""批量扫描 CSV 数据并写入任务队列。

python mycmd/server/produce_server/task_producer.py --config mycmd/server/produce_server/producer_config.yaml

该脚本使用 ``base_components`` 的 ``TaskPool`` 与 ``TaskRecord``，支持多进程
扫描 `mycmd/server/sample_data`（可覆盖）目录下的全部 ``.csv`` 数据集，并为每个文件
构造一个队列任务。每个任务的 ``extras`` 字段携带预测所需的路径、格式以及
样本数量提示，供单 GPU worker 直接消费。

示例：

    python mycmd/server/produce_server/task_producer.py \\
        --input-dir mycmd/server/sample_data \\
        --queue-dir mycmd/server/infer_server/runtime/queue \\
        --job-id tabicl-sample
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import os
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

import yaml

from base_components.logger import StructuredLogger
from base_components.task_partitioner import TaskRecord
from base_components.task_pool import TaskPool
from base_components.config import Config


logger = StructuredLogger.get_logger("tabicl.task_producer")


@dataclass(frozen=True)
class FileTaskSpec:
    """封装 CSV 文件的任务描述元信息。"""

    path: Path
    rel_path: str
    rows: int
    size_bytes: int

    @property
    def weight(self) -> float:
        return float(max(1, self.rows))


def _count_csv_rows(path: Path) -> int:
    """返回 CSV 文件中数据行数（忽略首行表头）。"""
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        lines = sum(1 for _ in handle)
    return max(0, lines - 1)


def _prepare_spec(args: Tuple[Path, Path]) -> FileTaskSpec:
    """子进程读取单个文件，返回任务元信息。"""
    file_path, input_root = args
    rows = _count_csv_rows(file_path)
    stat = file_path.stat()
    rel_path = str(file_path.relative_to(input_root)).replace(os.sep, "/")
    return FileTaskSpec(
        path=file_path,
        rel_path=rel_path,
        rows=rows,
        size_bytes=stat.st_size,
    )


def _iter_chunks(items: Sequence[TaskRecord], chunk: int = 128) -> Iterator[Sequence[TaskRecord]]:
    it = iter(items)
    while True:
        batch = list(itertools.islice(it, chunk))
        if not batch:
            break
        yield batch


def build_task(
    spec: FileTaskSpec,
    job_id: str,
    input_root: Path,
    *,
    feature_indices: Optional[Sequence[int]] = None,
    feature_exclude: Optional[Sequence[int]] = None,
    id_column_index: Optional[int] = None,
    pass_through_indices: Optional[Sequence[int]] = None,
    attempt: int = 0,
) -> TaskRecord:
    """根据文件元信息构造 TaskRecord。"""
    digest = hashlib.blake2b(
        f"{spec.rel_path}:{spec.rows}:{spec.size_bytes}".encode("utf-8"),
        digest_size=10,
    ).hexdigest()
    task_id = f"{job_id}-{digest}"
    extras = {
        "predict_path": str((input_root / spec.rel_path).as_posix()),
        "predict_format": "csv",
        "row_hint": spec.rows,
    }
    if feature_indices:
        extras["feature_indices"] = [int(idx) for idx in feature_indices]
    if feature_exclude:
        extras["feature_exclude_indices"] = [int(idx) for idx in feature_exclude]
    if id_column_index is not None:
        extras["id_column_index"] = int(id_column_index)
    if pass_through_indices:
        extras["pass_through_indices"] = [int(idx) for idx in pass_through_indices]
    payload_ref = (spec.rel_path,)
    return TaskRecord(
        task_id=task_id,
        job_id=job_id,
        attempt=attempt,
        payload_ref=payload_ref,
        weight=spec.weight,
        group_keys=("csv",),
        checksum=digest,
        extras=extras,
    )


def produce_tasks(
    specs: Iterable[FileTaskSpec],
    job_id: str,
    input_root: Path,
    *,
    feature_indices: Optional[Sequence[int]] = None,
    feature_exclude: Optional[Sequence[int]] = None,
    id_column_index: Optional[int] = None,
    pass_through_indices: Optional[Sequence[int]] = None,
) -> List[TaskRecord]:
    return [
        build_task(
            spec,
            job_id,
            input_root,
            feature_indices=feature_indices,
            feature_exclude=feature_exclude,
            id_column_index=id_column_index,
            pass_through_indices=pass_through_indices,
        )
        for spec in specs
    ]


DEFAULT_CONFIG_PATH = Path("mycmd/server/produce_server/producer_config.yaml")


def _load_config(path: Optional[Path]) -> Mapping[str, object]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        raise ValueError("配置文件的根节点必须是映射类型")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="扫描 CSV 数据并写入 TabICL 队列")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help=f"可选的配置文件路径（默认查找 {DEFAULT_CONFIG_PATH}）",
    )
    parser.add_argument(
        "--task-config",
        type=Path,
        default=None,
        help="task_system 配置文件路径（默认使用 mycmd/server/infer_server/worker_config.yaml）",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help="待扫描的输入目录（默认：配置文件或 mycmd/server/sample_data）",
    )
    parser.add_argument(
        "--queue-dir",
        type=Path,
        default=None,
        help="任务队列所在目录（默认：配置文件或 mycmd/server/infer_server/runtime/queue）",
    )
    parser.add_argument(
        "--job-id",
        type=str,
        default=None,
        help="任务所属 job_id（默认：配置文件或 tabicl-sample）",
    )
    parser.add_argument(
        "--queue-name",
        type=str,
        default=None,
        help="任务队列名称（默认：配置文件或 tabicl）",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="随机打乱任务顺序时使用的随机种子（覆盖配置文件）",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=None,
        help="并行扫描进程数（默认：配置文件或 CPU 核心数）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅打印将要写入的任务信息，不实际写入队列",
    )
    parser.add_argument(
        "--feature-indices",
        type=str,
        default=None,
        help="逗号分隔的特征列索引（0 基），覆盖配置文件",
    )
    parser.add_argument(
        "--feature-exclude-indices",
        type=str,
        default=None,
        help="逗号分隔的需要排除的列索引，覆盖配置文件",
    )
    parser.add_argument(
        "--id-column-index",
        type=str,
        default=None,
        help="预测数据中的样本 ID 列索引（覆盖配置文件）",
    )
    parser.add_argument(
        "--pass-through-indices",
        type=str,
        default=None,
        help="逗号分隔的预测输出需要保留的列索引",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config_path = args.config or (DEFAULT_CONFIG_PATH if DEFAULT_CONFIG_PATH.exists() else None)
    cfg = _load_config(config_path)

    task_cfg_path = args.task_config or cfg.get("task_config")
    if task_cfg_path is None:
        task_cfg_path = Path("mycmd/server/infer_server/worker_config.yaml")
    elif not isinstance(task_cfg_path, Path):
        task_cfg_path = Path(str(task_cfg_path))
    task_cfg_path = task_cfg_path.expanduser().resolve()
    if not task_cfg_path.exists():
        raise FileNotFoundError(f"task_system 配置文件不存在: {task_cfg_path}")
    Config.load_singleton(task_cfg_path)

    def _cfg_path(key: str, default: str) -> Path:
        raw = cfg.get(key, default)
        if raw is None:
            raw = default
        if isinstance(raw, Path):
            return raw
        return Path(str(raw))

    if args.processes is not None:
        processes = max(1, int(args.processes))
    else:
        cfg_processes = cfg.get("processes", None)
        if cfg_processes in (None, "auto"):
            processes = max(1, os.cpu_count() or 1)
        else:
            processes = max(1, int(cfg_processes))

    def _parse_indices(config_value: Optional[object], override: Optional[str], *, name: str) -> Optional[List[int]]:
        tokens: Optional[List[str]] = None
        if override:
            tokens = [item.strip() for item in override.split(",") if item.strip()]
        elif config_value is not None:
            if isinstance(config_value, (list, tuple)):
                tokens = [str(item) for item in config_value if item is not None]
            else:
                tokens = [str(config_value)]
        if not tokens:
            return None
        indices: List[int] = []
        for token in tokens:
            try:
                indices.append(int(token))
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} 必须为整数列表") from exc
        return indices or None

    def _parse_bool(value: object, *, name: str, default: bool = False) -> bool:
        if value is None:
            return default
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lower = value.strip().lower()
            if lower in {"1", "true", "yes", "on"}:
                return True
            if lower in {"0", "false", "no", "off"}:
                return False
        try:
            return bool(value)
        except Exception as exc:  # pragma: no cover - 防御性兜底
            raise ValueError(f"{name} 必须为布尔类型") from exc

    def _parse_optional_int(value: object, *, name: str) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, bool):
            raise ValueError(f"{name} 不可为布尔类型")
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            stripped = value.strip()
            if stripped == "":
                return None
            try:
                return int(stripped)
            except ValueError as exc:
                raise ValueError(f"{name} 必须为整数或留空") from exc
        try:
            return int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} 必须为整数或留空") from exc

    input_dir = args.input_dir or _cfg_path("input_dir", "mycmd/server/sample_data")
    queue_dir = args.queue_dir or _cfg_path("queue_dir", "mycmd/server/infer_server/runtime/queue")
    job_id = args.job_id or str(cfg.get("job_id", "tabicl-sample"))
    queue_name = args.queue_name or str(cfg.get("queue_name", "tabicl"))
    randomize_task_order = _parse_bool(cfg.get("randomize_task_order", False), name="randomize_task_order")
    random_seed = _parse_optional_int(
        args.random_seed if args.random_seed is not None else cfg.get("random_seed"),
        name="random_seed",
    )

    feature_indices = _parse_indices(cfg.get("feature_indices", cfg.get("feature_columns")), args.feature_indices, name="feature_indices")
    feature_exclude = _parse_indices(cfg.get("feature_exclude_indices", cfg.get("feature_exclude_columns")), args.feature_exclude_indices, name="feature_exclude_indices")
    if feature_indices and feature_exclude:
        raise ValueError("不能同时指定 feature_indices 与 feature_exclude_indices")

    id_column_index_raw = args.id_column_index or cfg.get("id_column_index", cfg.get("id_column"))
    id_column_index: Optional[int]
    if id_column_index_raw is None or id_column_index_raw == "":
        id_column_index = None
    else:
        try:
            id_column_index = int(id_column_index_raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("id_column_index 必须为整数") from exc
    pass_through_indices = _parse_indices(cfg.get("pass_through_indices", cfg.get("pass_through_columns")), args.pass_through_indices, name="pass_through_indices")

    input_root = input_dir.expanduser().resolve()
    queue_root = queue_dir.expanduser().resolve()
    queue_root.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_root.rglob("*.csv"))
    if not csv_files:
        logger.warn("producer.no_files", directory=str(input_root))
        return 0

    logger.info(
        "producer.scan_start",
        files=len(csv_files),
        input=str(input_root),
        queue=str(queue_root),
        processes=processes,
        job_id=job_id,
        queue_name=queue_name,
    )

    specs: List[FileTaskSpec] = []
    try:
        with ProcessPoolExecutor(max_workers=processes) as executor:
            futures = executor.map(_prepare_spec, ((path, input_root) for path in csv_files))
            for spec in futures:
                specs.append(spec)
    except (PermissionError, OSError) as exc:
        logger.warn(
            "producer.process_pool_unavailable",
            error=str(exc),
            fallback="sequential",
        )
        for path in csv_files:
            specs.append(_prepare_spec((path, input_root)))

    tasks = produce_tasks(
        specs,
        job_id,
        input_root,
        feature_indices=feature_indices,
        feature_exclude=feature_exclude,
        id_column_index=id_column_index,
        pass_through_indices=pass_through_indices,
    )
    if randomize_task_order:
        if random_seed is not None:
            random.Random(random_seed).shuffle(tasks)
        else:
            random.shuffle(tasks)

    if args.dry_run:
        for task in tasks:
            logger.info(
                "producer.preview",
                task_id=task.task_id,
                payload=list(task.payload_ref),
                rows=int(task.weight),
                extras=task.extras,
            )
        logger.info("producer.dry_run_complete", total=len(tasks))
        return 0

    pool = TaskPool(queue_root=queue_root, queue_name=queue_name)
    for batch in _iter_chunks(tasks, chunk=128):
        pool.put(batch)

    logger.info(
        "producer.done",
        total=len(tasks),
        queue=str(queue_root),
        job_id=job_id,
        queue_name=queue_name,
        input=str(input_root),
    )
    return 0


if __name__ == "__main__":
    StructuredLogger.configure(
        sinks=[{"type": "console", "stream": "stdout", "min_level": "INFO"}],
        level="INFO",
        namespace="tabicl.task_producer",
    )
    raise SystemExit(main())
