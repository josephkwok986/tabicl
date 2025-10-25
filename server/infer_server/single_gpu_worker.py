#!/usr/bin/env python3
"""TabICL 单 GPU 推理常驻服务。

python server/infer_server/single_gpu_worker.py --config server/infer_server/worker_config.yaml

该脚本基于 ``base_components`` 提供的执行框架，实现一个“单 GPU、永不自停、
人工信号终止”的任务消费进程。主进程负责：

1. 读取配置（默认位于 ``server/infer_server/worker_config.yaml``，可通过 ``--config`` 覆盖）
2. 初始化本地文件队列 ``TaskPool`` 与单 GPU ``ParallelExecutor``
3. 注册进度控制器，持续打印 “已扫描/产生 | 已处理” 计数
4. 常驻轮询任务，空闲时输出 idle 日志，不会因队列耗尽而退出
5. 捕获 ``SIGINT/SIGTERM``，仅设置停止标志并调用 ``executor.stop()``

工作流说明：

* 队列使用文件后端，提供 put/lease/ack/nack/heartbeat 语义，支持租约过期回收。
* 单个 GPU 工作者进程会按上下文配置在首次运行时加载训练样本，并在每个任务中
  仅读取预测数据，通过 ``TabICLClassifier`` 完成推理后由 ``TaskResult`` 将结果写回磁盘。
* 结果写入使用 ``ParallelExecutor`` 内置的 ``_persist_task_result``，保证
  立即落盘，避免堆积在内存。
* Worker 发生异常时会按配置的 ``max_retries`` 指数回退重试；超过重试次数会
  标记为 dead 并把堆栈写入日志。
* 心跳由 worker 自动上报；主进程收到后调用 ``pool.heartbeat`` 续租，若终止或
  无心跳则任务会在 TTL 过期后回到可见集合。

上下文训练数据由 `single_gpu_worker.context_config_path` 指定的文件统一管理。
任务仅需提供待推理数据及输出偏好。例如：

```yaml
extras:
  predict_path: "/path/to/predict.csv"      # 待推理数据路径
  predict_format: "csv"                     # 可选：csv/parquet/jsonl
  id_column: "id"                           # 可选，覆盖默认上下文 id 列
  output_directory: "customer_a"            # 可选，若为相对路径则相对 output_root
  output_filename: "run-001.csv"            # 可选，未提供时按模板生成
  row_hint: 2048                            # 可选，供进度控制器估算“已扫描/产生”
```

环境准备：

1. 确保 HuggingFace checkpoint 已预先下载到配置中指定的 ``model_path``；
   在默认受限网络环境下不要依赖在线下载。
2. 外部生产者可持续向配置指定的队列目录写入任务，本服务仅消费。
3. 以守护进程方式运行本脚本，退出时直接发送 ``SIGINT`` 或 ``SIGTERM``。
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, List

import pandas as pd
import yaml

from base_components.logger import StructuredLogger
from base_components.parallel_executor import (
    ExecutorEvents,
    ExecutorPolicy,
    ParallelExecutor,
    TaskResult,
)
from base_components.progress import ProgressController
from base_components.task_pool import LeasedTask, TaskPool
from base_components.task_system_config import ensure_task_config

from tabicl.sklearn.classifier import TabICLClassifier

# ---------------------------------------------------------------------------
# 数据类定义
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ServiceSettings:
    """主进程所需的配置快照。"""

    output_root: Path
    default_output_filename: str
    progress_description: str
    default_data_format: str
    include_probabilities: bool
    job_id: str
    data_root: Optional[Path]
    model_params: Dict[str, Any]
    context_config_path: Path
    model_ckpt_path: Path
    task_queue_path: Path
    idle_log_interval: float


@dataclass(frozen=True)
class HandlerContext:
    """传入 worker 进程的只读配置。"""

    settings: ServiceSettings


# ---------------------------------------------------------------------------
# Worker 内部缓存（单进程，safe）
# ---------------------------------------------------------------------------

CLASSIFIER_INSTANCE: Optional[TabICLClassifier] = None
CLASSIFIER_FEATURE_INDICES: Tuple[int, ...] = ()
CLASSIFIER_ID_INDEX: Optional[int] = None
CLASSIFIER_PASSTHROUGH_INDICES: Tuple[int, ...] = ()
CLASSIFIER_LOCK = threading.Lock()
CLASSIFIER_TRAIN_PATH: Optional[str] = None


def _load_context_spec(path: Path) -> Mapping[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"上下文配置文件不存在: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"上下文配置文件格式应为映射: {path}")
    return data


def _parse_int_list(value: Optional[Iterable[Any]], *, name: str) -> Optional[List[int]]:
    if value is None:
        return None
    items: List[int] = []
    for item in value:
        if item is None or item == "":
            continue
        try:
            items.append(int(item))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} 必须为整数列表") from exc
    return items or None


def _parse_optional_index(value: Any, *, name: str) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须为整数") from exc


def _initialise_classifier_once(settings: ServiceSettings) -> Tuple[TabICLClassifier, Tuple[int, ...], Optional[int], Tuple[int, ...]]:
    global CLASSIFIER_INSTANCE, CLASSIFIER_FEATURE_INDICES, CLASSIFIER_ID_INDEX, CLASSIFIER_PASSTHROUGH_INDICES, CLASSIFIER_TRAIN_PATH
    if CLASSIFIER_INSTANCE is not None:
        return CLASSIFIER_INSTANCE, CLASSIFIER_FEATURE_INDICES, CLASSIFIER_ID_INDEX, CLASSIFIER_PASSTHROUGH_INDICES

    with CLASSIFIER_LOCK:
        if CLASSIFIER_INSTANCE is not None:
            return CLASSIFIER_INSTANCE, CLASSIFIER_FEATURE_INDICES, CLASSIFIER_ID_INDEX, CLASSIFIER_PASSTHROUGH_INDICES

        spec = _load_context_spec(settings.context_config_path)

        train_path_raw = spec.get("context_path")
        if not train_path_raw:
            raise ValueError("上下文配置缺少 context_path")

        label_index = _parse_optional_index(spec.get("label_column_index", spec.get("label_column")), name="label_column_index")
        if label_index is None:
            raise ValueError("label_column_index 不能为空")

        feature_indices_raw = spec.get("feature_indices", spec.get("feature_columns"))
        feature_exclude_raw = spec.get("feature_exclude_indices", spec.get("feature_exclude_columns"))
        id_index_raw = spec.get("id_column_index", spec.get("id_column"))

        train_fmt = spec.get("train_format", settings.default_data_format)

        base_dir = settings.data_root
        cwd = Path.cwd()
        train_path = _resolve_path(str(train_path_raw), base=base_dir, cwd=cwd)

        if Path(train_path).is_dir():
            dfs: List[pd.DataFrame] = []
            for file in sorted(Path(train_path).glob("*.csv")):
                dfs.append(_load_table(file, train_fmt, header=None))
            if not dfs:
                raise ValueError(f"上下文目录 {train_path} 下未找到 CSV 文件")
            train_df = pd.concat(dfs, ignore_index=True)
        else:
            train_df = _load_table(train_path, train_fmt, header=None)
        column_count = train_df.shape[1]
        if label_index < 0 or label_index >= column_count:
            raise ValueError("label_column_index 超出训练数据列范围")

        if feature_indices_raw and feature_exclude_raw:
            raise ValueError("上下文配置不能同时指定 feature_indices 和 feature_exclude_indices")

        feature_indices = _parse_int_list(feature_indices_raw, name="feature_indices")
        feature_exclude = _parse_int_list(feature_exclude_raw, name="feature_exclude_indices")

        if feature_indices:
            selected = []
            for idx in feature_indices:
                if idx == label_index:
                    raise ValueError("feature_indices 中包含 label_column_index")
                if idx < 0 or idx >= column_count:
                    raise ValueError(f"feature_indices 中的列 {idx} 超出范围")
                selected.append(idx)
        else:
            selected = [idx for idx in range(column_count) if idx != label_index]
            if feature_exclude:
                for idx in feature_exclude:
                    if idx < 0 or idx >= column_count:
                        raise ValueError(f"feature_exclude_indices 中的列 {idx} 超出范围")
                selected = [idx for idx in selected if idx not in feature_exclude]

        if not selected:
            raise ValueError("上下文配置未能确定特征列")

        id_index = _parse_optional_index(id_index_raw, name="id_column_index")
        if id_index is not None:
            if id_index < 0 or id_index >= column_count:
                raise ValueError("id_column_index 超出训练数据列范围")
            if id_index == label_index:
                raise ValueError("id_column_index 不应与 label_column_index 相同")
            if id_index in selected:
                raise ValueError("id_column_index 不应作为特征列")

        passthrough_raw = spec.get("output_passthrough_indices", [])
        passthrough_indices = _parse_int_list(passthrough_raw, name="output_passthrough_indices") or []
        for idx in passthrough_indices:
            if idx < 0 or idx >= column_count:
                raise ValueError(f"output_passthrough_indices 中的列 {idx} 超出范围")

        model_params = dict(settings.model_params)
        model_params["model_path"] = str(settings.model_ckpt_path)
        classifier = TabICLClassifier(**model_params)

        X_train = train_df.iloc[:, selected]
        y_train = train_df.iloc[:, label_index]
        classifier.fit(X_train, y_train)

        CLASSIFIER_INSTANCE = classifier
        CLASSIFIER_FEATURE_INDICES = tuple(selected)
        CLASSIFIER_ID_INDEX = id_index
        CLASSIFIER_PASSTHROUGH_INDICES = tuple(sorted(set(passthrough_indices)))
        CLASSIFIER_TRAIN_PATH = str(train_path)

        return CLASSIFIER_INSTANCE, CLASSIFIER_FEATURE_INDICES, CLASSIFIER_ID_INDEX, CLASSIFIER_PASSTHROUGH_INDICES


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------


def _load_table(path: Path, fmt: str, header: Optional[int | None] = "infer") -> pd.DataFrame:
    fmt_l = fmt.lower()
    if fmt_l == "csv":
        return pd.read_csv(path, header=header)
    if fmt_l in {"parquet", "pq"}:
        return pd.read_parquet(path)
    if fmt_l in {"jsonl", "ndjson"}:
        return pd.read_json(path, lines=True)
    raise ValueError(f"不支持的数据格式: {fmt}")


def _ensure_relative_filename(filename: str) -> str:
    candidate = Path(filename)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"output_filename 必须为相对路径且不能包含 ..: {filename!r}")
    return filename.replace("\\", "/")


def _resolve_path(raw: str, *, base: Optional[Path], cwd: Path) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    if base is not None:
        return (base / path).resolve()
    return (cwd / path).resolve()


def _format_default_filename(template: str, *, task_id: str, job_id: str) -> str:
    try:
        return template.format(task_id=task_id, job_id=job_id)
    except Exception as exc:  # pragma: no cover - 防御
        raise ValueError(f"无法根据模板 {template!r} 生成文件名") from exc


def _extract_row_hint(extras: Mapping[str, Any]) -> int:
    for key in ("row_hint", "predict_rows", "expected_rows", "item_count"):
        value = extras.get(key)
        if isinstance(value, (int, float)):
            return max(0, int(value))
    return 1


# ---------------------------------------------------------------------------
# Worker 处理函数
# ---------------------------------------------------------------------------


def handle_task(leased: LeasedTask, context: HandlerContext) -> TaskResult:
    """Worker 内部执行函数。"""
    start_ns = time.perf_counter_ns()
    extras: Mapping[str, Any] = leased.task.extras or {}
    settings = context.settings

    classifier, feature_indices, default_id_index, default_passthrough = _initialise_classifier_once(settings)

    predict_path_raw = extras.get("predict_path") or extras.get("inference_path")
    if not predict_path_raw:
        raise ValueError("任务缺少 predict_path")

    task_feature_indices_raw = extras.get("feature_indices", extras.get("feature_columns"))
    task_feature_exclude_raw = extras.get("feature_exclude_indices", extras.get("feature_exclude_columns"))
    task_feature_indices = _parse_int_list(task_feature_indices_raw, name="feature_indices")
    task_feature_exclude = _parse_int_list(task_feature_exclude_raw, name="feature_exclude_indices")
    if task_feature_indices and task_feature_exclude:
        raise ValueError("任务 extras 中不能同时指定 feature_indices 和 feature_exclude_indices")
    if task_feature_indices:
        expected = tuple(sorted(feature_indices))
        provided = tuple(sorted(task_feature_indices))
        if provided != expected:
            raise ValueError(
                "任务指定的 feature_indices 与上下文配置不一致，"
                f"上下文: {expected}, 任务: {provided}"
            )
    if task_feature_exclude:
        StructuredLogger.get_logger("tabicl.single_gpu_worker").debug(
            "task.feature_exclude_ignored",
            task_id=leased.task.task_id,
            provided=task_feature_exclude,
        )

    base_dir = settings.data_root
    cwd = Path.cwd()
    predict_path = _resolve_path(str(predict_path_raw), base=base_dir, cwd=cwd)
    predict_fmt = extras.get("predict_format", settings.default_data_format)

    predict_df = _load_table(predict_path, predict_fmt, header=None)
    max_index = predict_df.shape[1] - 1
    if max(feature_indices) > max_index:
        raise ValueError("预测数据特征列数量不足")

    X_predict = predict_df.iloc[:, list(feature_indices)]
    id_index_raw = extras.get("id_column_index", extras.get("id_column"))
    id_index = _parse_optional_index(id_index_raw if id_index_raw is not None else default_id_index, name="id_column_index")
    if id_index is not None and (id_index < 0 or id_index > max_index):
        raise ValueError(f"预测数据缺少 id_column_index: {id_index}")

    predictions = classifier.predict(X_predict)
    probabilities = classifier.predict_proba(X_predict) if settings.include_probabilities else None
    classes = classifier.classes_

    passthrough_raw = extras.get("pass_through_indices", extras.get("pass_through_columns"))
    passthrough_indices = _parse_int_list(passthrough_raw, name="pass_through_indices")
    if passthrough_indices is None:
        passthrough_indices = list(default_passthrough)
    passthrough_indices = list(dict.fromkeys(passthrough_indices))

    include_prob_column = settings.include_probabilities and probabilities is not None

    rows: List[Dict[str, Any]] = []
    for idx, label in enumerate(predictions):
        row: Dict[str, Any] = {"prediction": label}
        if include_prob_column:
            prob_map = {
                str(classes[j]): float(probabilities[idx][j]) for j in range(len(classes))
            }
            row["probabilities"] = json.dumps(prob_map, ensure_ascii=False)
        if id_index is not None:
            row["id"] = predict_df.iloc[idx, id_index]
        for col_index in passthrough_indices:
            if 0 <= col_index <= max_index:
                row[f"col_{col_index}"] = predict_df.iloc[idx, col_index]
        rows.append(row)

    desired_columns: List[str] = ["prediction"]
    if include_prob_column:
        desired_columns.append("probabilities")
    if id_index is not None:
        desired_columns.append("id")
    for col_index in passthrough_indices:
        name = f"col_{col_index}"
        if name not in desired_columns:
            desired_columns.append(name)

    if rows:
        result_df = pd.DataFrame(rows)
    else:
        result_df = pd.DataFrame(columns=desired_columns)
    result_df = result_df.reindex(columns=desired_columns)
    csv_content = result_df.to_csv(index=False)
    output_columns = [str(column) for column in result_df.columns]

    column_descriptions: Dict[str, str] = {}
    if "prediction" in output_columns:
        column_descriptions["prediction"] = "模型预测结果"
    if include_prob_column and "probabilities" in output_columns:
        column_descriptions["probabilities"] = "类别概率分布（JSON 字符串）"
    if id_index is not None and "id" in output_columns:
        column_descriptions["id"] = f"样本唯一 ID，来源于输入列索引 {id_index}"
    for col_index in passthrough_indices:
        name = f"col_{col_index}"
        if name in output_columns:
            column_descriptions[name] = f"输入数据原始列索引 {col_index}"

    output_dir_raw = extras.get("output_directory")
    output_filename = extras.get("output_filename")

    if output_dir_raw:
        out_dir = _resolve_path(str(output_dir_raw), base=settings.output_root, cwd=settings.output_root)
    else:
        out_dir = settings.output_root

    if output_filename:
        output_name = _ensure_relative_filename(output_filename)
    else:
        output_name = _ensure_relative_filename(
            _format_default_filename(settings.default_output_filename, task_id=leased.task.task_id, job_id=settings.job_id)
        )
    output_path_rel = Path(output_name)
    if output_path_rel.suffix.lower() != ".csv":
        output_path_rel = output_path_rel.with_suffix(".csv")
    output_name = output_path_rel.as_posix()

    out_dir.mkdir(parents=True, exist_ok=True)
    columns_relative = output_path_rel.with_suffix(".columns.json")
    columns_path = out_dir / columns_relative
    columns_path.parent.mkdir(parents=True, exist_ok=True)
    columns_payload = {
        "columns": output_columns,
        "descriptions": column_descriptions,
    }
    columns_path.write_text(json.dumps(columns_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    duration_ms = (time.perf_counter_ns() - start_ns) / 1_000_000
    metadata = {
        "task_id": leased.task.task_id,
        "job_id": leased.task.job_id,
        "row_count": len(rows),
        "feature_indices": list(feature_indices),
        "predict_path": str(predict_path),
        "duration_ms": round(duration_ms, 3),
        "attempt": leased.attempt,
        "context_config": str(settings.context_config_path),
    }
    if task_feature_indices:
        metadata["task_feature_indices"] = list(task_feature_indices)
    if task_feature_exclude:
        metadata["task_feature_exclude_indices"] = list(task_feature_exclude)
    if id_index is not None:
        metadata["id_column_index"] = id_index
    if passthrough_indices:
        metadata["pass_through_indices"] = list(passthrough_indices)
    metadata["output_columns"] = output_columns
    metadata["columns_description_file"] = str(columns_path)
    metadata["classes"] = [str(cls) for cls in classes]
    metadata["probabilities_included"] = bool(include_prob_column)
    if CLASSIFIER_TRAIN_PATH:
        metadata["context_path"] = CLASSIFIER_TRAIN_PATH

    return TaskResult(
        payload=csv_content,
        processed=len(rows) if rows else 1,
        metadata=metadata,
        output_directory=str(out_dir),
        output_filename=output_name,
        is_final_output=True,
    )


# ---------------------------------------------------------------------------
# 主服务
# ---------------------------------------------------------------------------


class SingleGPUWorkerService:
    def __init__(self, settings: ServiceSettings) -> None:
        self._settings = settings
        self._logger = StructuredLogger.get_logger("tabicl.single_gpu_worker")
        self._progress = ProgressController(
            total_units=None,
            description=settings.progress_description,
            stream=True,
            unit_name="task",
        )
        self._pool = TaskPool()
        self._executor: Optional[ParallelExecutor] = None
        self._stop_event = threading.Event()
        self._idle_log_interval = max(1.0, float(settings.idle_log_interval))
        self._last_idle_log = 0.0

    def run(self) -> int:
        self._logger.info(
            "service.start",
            job_id=self._settings.job_id,
            context_config=str(self._settings.context_config_path),
            model_ckpt=str(self._settings.model_ckpt_path),
            task_queue=str(self._settings.task_queue_path),
        )
        self._settings.output_root.mkdir(parents=True, exist_ok=True)
        (self._settings.output_root / "cache").mkdir(parents=True, exist_ok=True)

        policy = ExecutorPolicy()
        if policy.max_concurrency != 1:
            raise RuntimeError("配置必须将 task_system.executor.max_concurrency 设置为 1")

        events = ExecutorEvents(
            on_lease=self._on_lease,
            on_success=self._on_success,
            on_retry=self._on_retry,
            on_dead=self._on_dead,
            on_stop=lambda ex: self._logger.info("executor.stop"),
        )

        self._progress.start()

        handler_ctx = HandlerContext(settings=self._settings)

        self._executor = ParallelExecutor(
            handler=handle_task,
            pool=self._pool,
            policy=policy,
            events=events,
            handler_context=handler_ctx,
            result_handler=self._result_handler,
            console_min_level="INFO",
        )
        try:
            self._executor.start()
        except Exception:
            self._progress.close()
            raise

        self._install_signal_handlers()

        try:
            while not self._stop_event.wait(timeout=1.0):
                stats = self._pool.stats()
                visible = int(stats.get("visible", 0))
                leased = int(stats.get("leased", 0))
                if visible == 0 and leased == 0:
                    now = time.time()
                    if now - self._last_idle_log >= self._idle_log_interval:
                        self._last_idle_log = now
                        self._logger.info("service.idle", message="队列空闲，等待新任务")
        except KeyboardInterrupt:
            self._logger.info("service.signal", signal="keyboard_interrupt")
            self._request_stop()
        finally:
            self._shutdown()
        return 0

    # ------------------------------------------------------------------
    # 事件处理
    # ------------------------------------------------------------------

    def _on_lease(self, executor: ParallelExecutor, leased: LeasedTask) -> None:
        hint = max(1, _extract_row_hint(leased.task.extras or {}))
        self._progress.discovered(hint)
        self._logger.debug(
            "task.leased",
            task_id=leased.task.task_id,
            attempt=leased.attempt,
            deadline=leased.lease_deadline,
            hint=hint,
        )

    def _on_success(self, executor: ParallelExecutor, leased: LeasedTask, latency: float, result: TaskResult) -> None:
        self._logger.info(
            "task.success",
            task_id=leased.task.task_id,
            attempt=leased.attempt,
            latency_ms=int(latency * 1000),
        )

    def _on_retry(self, executor: ParallelExecutor, leased: LeasedTask, attempt: int, exc: Exception) -> None:
        self._logger.warn(
            "task.retry",
            task_id=leased.task.task_id,
            attempt=attempt,
            error=str(exc),
        )

    def _on_dead(self, executor: ParallelExecutor, leased: LeasedTask, exc: Exception) -> None:
        self._logger.error(
            "task.dead",
            task_id=leased.task.task_id,
            error=str(exc),
        )

    def _result_handler(self, leased: LeasedTask, result: TaskResult) -> None:
        processed = int(result.processed or 0) or 1
        self._progress.advance(processed)
        written_path = result.written_path or result.payload
        self._logger.info(
            "task.result_persisted",
            task_id=leased.task.task_id,
            path=written_path,
            processed=processed,
        )

    # ------------------------------------------------------------------
    # 信号 & 退出
    # ------------------------------------------------------------------

    def _install_signal_handlers(self) -> None:
        signal.signal(signal.SIGINT, self._handle_signal)
        signal.signal(signal.SIGTERM, self._handle_signal)

    def _handle_signal(self, signum, frame) -> None:  # noqa: D401
        self._logger.info("service.signal", signal=signum)
        self._request_stop()

    def _request_stop(self) -> None:
        self._stop_event.set()
        if self._executor is not None:
            self._executor.stop()

    def _shutdown(self) -> None:
        if self._executor is not None:
            self._executor.wait()
        self._progress.close()
        self._logger.info("service.stop")


# ---------------------------------------------------------------------------
# 设置加载 & CLI
# ---------------------------------------------------------------------------


def _load_settings() -> ServiceSettings:
    cfg = ensure_task_config()
    worker_cfg = cfg.get("single_gpu_worker", dict, {})

    output_root = Path(worker_cfg.get("output_root", "server/infer_server/runtime/results")).expanduser().resolve()
    default_output_filename = worker_cfg.get("default_output_filename", "{task_id}.csv")
    progress_description = worker_cfg.get("progress_description", "TabICL 单 GPU Worker")
    default_data_format = worker_cfg.get("default_data_format", "csv").lower()
    include_probabilities = bool(worker_cfg.get("include_probabilities", True))
    job_id = worker_cfg.get("job_id", "tabicl-single-gpu")
    data_root_raw = worker_cfg.get("data_root")
    data_root = Path(data_root_raw).expanduser().resolve() if data_root_raw else None
    model_params = dict(worker_cfg.get("model", {}))
    if model_params.get("allow_auto_download") is None:
        model_params["allow_auto_download"] = False
    context_config_raw = worker_cfg.get("context_config_path")
    if not context_config_raw:
        raise ValueError("single_gpu_worker.context_config_path 未配置")
    context_config_path = Path(context_config_raw).expanduser().resolve()

    model_ckpt_raw = worker_cfg.get("model_ckpt_path")
    if not model_ckpt_raw:
        raise ValueError("single_gpu_worker.model_ckpt_path 未配置")
    model_ckpt_path = Path(model_ckpt_raw).expanduser().resolve()

    model_params.setdefault("allow_auto_download", False)
    model_params.setdefault("device", "cuda")
    model_params.setdefault("use_amp", True)
    model_params["model_path"] = str(model_ckpt_path)

    backend_cfg = cfg.get("task_system.pool.backend", dict, {})
    queue_path_raw = worker_cfg.get("task_queue_path") or backend_cfg.get("path")
    if not queue_path_raw:
        raise ValueError("任务队列路径未配置：请设置 task_system.pool.backend.path 或 single_gpu_worker.task_queue_path")
    task_queue_path = Path(queue_path_raw).expanduser().resolve()

    idle_log_interval = float(worker_cfg.get("idle_log_interval", 15.0))

    return ServiceSettings(
        output_root=output_root,
        default_output_filename=default_output_filename,
        progress_description=progress_description,
        default_data_format=default_data_format,
        include_probabilities=include_probabilities,
        job_id=job_id,
        data_root=data_root,
        model_params=model_params,
        context_config_path=context_config_path,
        model_ckpt_path=model_ckpt_path,
        task_queue_path=task_queue_path,
        idle_log_interval=idle_log_interval,
    )


def _configure_logging() -> None:
    cfg = ensure_task_config()
    StructuredLogger.configure_from_config(cfg)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TabICL 单 GPU 推理服务")
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="配置文件路径（缺省为 server/infer_server/worker_config.yaml）",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = parse_args(argv)
    if args.config is not None:
        config_path = args.config.expanduser().resolve()
    else:
        config_path = Path(__file__).with_name("worker_config.yaml")
    os.environ["CAD_TASK_CONFIG"] = str(config_path)

    _configure_logging()
    settings = _load_settings()
    service = SingleGPUWorkerService(settings)
    return service.run()


if __name__ == "__main__":
    sys.exit(main())
