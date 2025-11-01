from __future__ import annotations

import csv
import io
import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from base_components.logger import StructuredLogger
from base_components.parallel_executor import (
    ExecutorEvents,
    ExecutorPolicy,
    ParallelExecutor,
    TaskResult,
)
from base_components.progress import ProgressController
from base_components.task_partitioner import TaskRecord
from base_components.task_pool import LeasedTask, TaskPool
from base_components.task_system_config import ensure_task_config


_LOGGER = StructuredLogger.get_logger("csv_split.service")


@dataclass(frozen=True)
class _SplitEntry:
    split: str
    filename: str
    design_id: str
    part_uid: str


@dataclass(frozen=True)
class HandlerContext:
    input_dir: Path
    output_root: Path


def _parse_identifiers(filename: str) -> Tuple[str, str]:
    stem = Path(filename).stem
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(f"文件名不符合约定格式: {filename}")
    design_id = "_".join(parts[:2])
    part_uid = stem
    return design_id, part_uid


def _collect_entries(
    dataset_json: Path,
    input_dir: Path,
    required_splits: Iterable[str],
) -> Tuple[List[_SplitEntry], Dict[str, int], List[str]]:
    with dataset_json.open("r", encoding="utf-8") as handle:
        dataset_spec = json.load(handle)

    entries: List[_SplitEntry] = []
    counts: Dict[str, int] = {}
    missing: List[str] = []

    for split in required_splits:
        raw_items = dataset_spec.get(split, [])
        if not isinstance(raw_items, list):
            raise ValueError(f"dataset.json 中 {split} 字段不是列表")
        counts[split] = 0
        for item in raw_items:
            filename = f"{item}.csv"
            csv_path = input_dir / filename
            if not csv_path.exists():
                missing.append(filename)
                continue
            design_id, part_uid = _parse_identifiers(filename)
            entries.append(
                _SplitEntry(
                    split=split,
                    filename=filename,
                    design_id=design_id,
                    part_uid=part_uid,
                )
            )
            counts[split] += 1
    return entries, counts, missing


def _inject_identifier_columns(
    source: Path,
    design_id: str,
    part_uid: str,
) -> Tuple[str, int]:
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    row_count = 0
    with source.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                writer.writerow([])
                continue
            expanded = list(row)
            if len(expanded) < 3:
                raise ValueError(f"CSV 数据列数不足三列，无法插入设计编号: {source}")
            expanded[3:3] = [design_id, part_uid]
            embeddings = [float(value) for value in expanded[5:]]
            norm = math.sqrt(sum(value * value for value in embeddings))
            if norm > 0.0:
                embeddings = [value / norm for value in embeddings]
            expanded[5:] = [f"{value:.18e}" for value in embeddings]
            writer.writerow(expanded)
            row_count += 1
    content = buffer.getvalue()
    buffer.close()
    return content, row_count


def handle_csv_task(leased: LeasedTask, context: HandlerContext) -> TaskResult:
    extras = leased.task.extras
    split = extras["split"]
    filename = extras["filename"]
    design_id = extras["design_id"]
    part_uid = extras["part_uid"]

    source_path = context.input_dir / filename
    if not source_path.exists():
        raise FileNotFoundError(f"源 CSV 不存在: {source_path}")

    content, row_count = _inject_identifier_columns(source_path, design_id, part_uid)
    output_filename = str(Path(split) / filename)
    metadata = {
        "split": split,
        "design_id": design_id,
        "part_uid": part_uid,
        "rows": row_count,
    }
    return TaskResult(
        payload=content,
        processed=row_count,
        metadata=metadata,
        output_directory=str(context.output_root),
        output_filename=output_filename,
        is_final_output=True,
    )


class CsvSplitService:
    """手动管理 ParallelExecutor，进行 CSV 多进程划分。"""

    def __init__(
        self,
        *,
        input_dir: Path,
        dataset_json: Path,
        output_root: Path,
        queue_root: Path,
        queue_name: str = "csv_split",
        batch_size: int = 128,
    ) -> None:
        self.input_dir = input_dir.expanduser().resolve()
        self.dataset_json = dataset_json.expanduser().resolve()
        self.output_root = output_root.expanduser().resolve()
        self.queue_root = queue_root.expanduser().resolve()
        self.queue_name = queue_name
        self.batch_size = max(1, batch_size)

    def run(self) -> int:
        ensure_task_config()
        if not self.input_dir.is_dir():
            _LOGGER.error("input.missing", path=str(self.input_dir))
            return 1
        if not self.dataset_json.is_file():
            _LOGGER.error("dataset_json.missing", path=str(self.dataset_json))
            return 1

        entries, counts, missing = _collect_entries(
            self.dataset_json,
            self.input_dir,
            ("training_set", "validation_set", "test_set"),
        )

        if missing:
            _LOGGER.warn(
                "dataset.missing_files",
                count=len(missing),
                samples=missing[:5],
            )

        total_files = len(entries)
        _LOGGER.info(
            "dataset.collected",
            splits=counts,
            total=total_files,
            input_dir=str(self.input_dir),
            output_root=str(self.output_root),
        )
        if total_files == 0:
            _LOGGER.error("dataset.empty")
            return 1

        self._prepare_output_dirs()
        self._reset_queue_dir()

        pool = TaskPool(queue_root=self.queue_root, queue_name=self.queue_name)

        policy = ExecutorPolicy()
        events = ExecutorEvents(
            on_lease=lambda ex, leased: None,
            on_success=lambda ex, leased, latency, result: None,
            on_retry=lambda ex, leased, attempt, exc: _LOGGER.warn(
                "task.retry",
                task_id=leased.task.task_id,
                attempt=attempt,
                error=str(exc),
            ),
            on_dead=lambda ex, leased, exc: _LOGGER.error(
                "task.dead",
                task_id=leased.task.task_id,
                error=str(exc),
            ),
            on_stop=lambda ex: _LOGGER.info("executor.stop"),
        )

        handler_ctx = HandlerContext(
            input_dir=self.input_dir,
            output_root=self.output_root,
        )

        progress = ProgressController(
            total_units=total_files,
            description="处理嵌入 CSV",
            unit_name="文件",
        )
        progress.start()

        executor = ParallelExecutor(
            handler=handle_csv_task,
            pool=pool,
            policy=policy,
            events=events,
            handler_context=handler_ctx,
            result_handler=self._build_result_handler(progress),
            console_min_level="INFO",
        )

        try:
            executor.start()
        except (PermissionError, OSError) as exc:
            progress.close()
            _LOGGER.warn("executor.parallel_unavailable", error=str(exc))
            return self._run_sequential(entries)

        try:
            self._enqueue_tasks(pool, entries, progress)
            pool.seal()
            pool.wait_until_drained()
        except Exception:
            _LOGGER.exception("service.error")
            pool.seal()
            executor.stop()
            try:
                executor.wait()
            except Exception:
                pass
            progress.close()
            return 1

        executor.stop()
        executor.wait()
        progress.close()
        return 0

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------

    def _prepare_output_dirs(self) -> None:
        for split in ("training_set", "validation_set", "test_set"):
            target_dir = self.output_root / split
            target_dir.mkdir(parents=True, exist_ok=True)

    def _reset_queue_dir(self) -> None:
        queue_dir = self.queue_root / self.queue_name
        if queue_dir.exists():
            shutil.rmtree(queue_dir, ignore_errors=True)
        queue_dir.mkdir(parents=True, exist_ok=True)

    def _enqueue_tasks(
        self,
        pool: TaskPool,
        entries: List[_SplitEntry],
        progress: ProgressController,
    ) -> None:
        buffer: List[TaskRecord] = []
        for idx, entry in enumerate(entries):
            task_id = f"csv-split-{idx:06d}"
            record = TaskRecord(
                task_id=task_id,
                job_id="csv-split",
                attempt=0,
                payload_ref=(entry.split, entry.filename),
                weight=1.0,
                group_keys=(entry.split,),
                checksum=None,
                extras={
                    "split": entry.split,
                    "filename": entry.filename,
                    "design_id": entry.design_id,
                    "part_uid": entry.part_uid,
                },
            )
            buffer.append(record)
            if len(buffer) >= self.batch_size:
                batch = list(buffer)
                pool.put(batch)
                progress.discovered(len(batch))
                buffer.clear()
        if buffer:
            batch = list(buffer)
            pool.put(batch)
            progress.discovered(len(batch))
            buffer.clear()

    def _build_result_handler(self, progress: ProgressController):
        def _handler(leased: LeasedTask, result: TaskResult) -> None:
            _ = leased, result
            progress.advance(1)
        return _handler

    def _run_sequential(
        self,
        entries: List[_SplitEntry],
    ) -> int:
        progress = ProgressController(
            total_units=len(entries),
            description="处理嵌入 CSV (顺序模式)",
            unit_name="文件",
        )
        progress.start()
        if entries:
            progress.discovered(len(entries))
        handler_ctx = HandlerContext(
            input_dir=self.input_dir,
            output_root=self.output_root,
        )
        start = time.time()
        try:
            for entry in entries:
                extras = {
                    "split": entry.split,
                    "filename": entry.filename,
                    "design_id": entry.design_id,
                    "part_uid": entry.part_uid,
                }
                task = TaskRecord(
                    task_id="sequential",
                    job_id="csv-split",
                    attempt=0,
                    payload_ref=(entry.split, entry.filename),
                    weight=1.0,
                    group_keys=(entry.split,),
                    checksum=None,
                    extras=extras,
                )
                leased = LeasedTask(
                    task=task,
                    lease_id="sequential",
                    lease_deadline=time.time() + 3600,
                    attempt=0,
                )
                result = handle_csv_task(leased, handler_ctx)
                self._write_output(result)
                progress.advance(1)
        finally:
            progress.close()
        elapsed = time.time() - start
        _LOGGER.info("service.sequential_done", elapsed_s=elapsed)
        return 0

    @staticmethod
    def _write_output(result: TaskResult) -> None:
        if not result.output_directory or not result.output_filename:
            raise ValueError("TaskResult 输出路径缺失")
        root = Path(result.output_directory).expanduser()
        destination = (root / result.output_filename).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = result.payload
        if isinstance(data, str):
            destination.write_text(data, encoding="utf-8")
        elif isinstance(data, bytes):
            destination.write_bytes(data)
        else:
            raise TypeError("不支持的 payload 类型")
