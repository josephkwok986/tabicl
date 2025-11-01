## CSV 划分工具

该目录提供 `mycmd.slipt_csv_set` 包，用于使用 `base_components` 的多进程框架，
将 `embeddings_brepnet_csv` 目录下的 CSV 文件按 `dataset.json` 中的
`training_set`、`validation_set`、`test_set` 划分到目标目录中，同时为每行数据
插入 `design_id` 与 `part_uid` 两个字段。

### 功能特性
- 基于 `base_components` 的任务队列与并行执行器，默认 4 进程并发。
- 不修改原始 CSV，结果写入
  `/workspace/Gjj Local/data/fusion-360-gallery-dataset-s2.0.0/processed/embeddings_brepnet_csv_step_1`
  下的子目录。
- 自动从文件名解析 `design_id` / `part_uid`，并将其添加到每行的第 3 列之后。

### 使用方式

```bash
python -m mycmd.slipt_csv_set \
  --input-dir "/workspace/Gjj Local/data/fusion-360-gallery-dataset-s2.0.0/processed/embeddings_brepnet_csv" \
  --dataset-json "/workspace/Gjj Local/data/fusion-360-gallery-dataset-s2.0.0/processed/dataset.json" \
  --output-root "/workspace/Gjj Local/data/fusion-360-gallery-dataset-s2.0.0/processed/embeddings_brepnet_csv_step_1" \
  --queue-root "mycmd/slipt_csv_set/runtime/queue"
```

若需进一步调整（如最大并发、批大小等），可结合命令行参数（`--batch-size`、
`--queue-root`）或编辑同目录下的 `task_config.yaml`。当环境不允许创建子进程时，
程序会自动降级为顺序模式，进度条统计仍保持准确。
