# mycmd 工具任务概览

`mycmd` 目录下收纳了围绕 TabICL 数据准备与推理流水线的若干实用脚本，涵盖数据
拆分、上下文抽样、任务生产与推理服务。本文件总结各任务的核心输入/输出以及配置
入口，便于快速串联整套离线→在线流程。

## slipt_csv_set
- **入口**：`python -m mycmd.slipt_csv_set`
- **主要输入**  
  - `--input-dir`：原始嵌入文件根目录（包含多份 CAD 面级 CSV）  
  - `--dataset-json`：划分方案 `dataset.json`
- **主要输出**：将 CSV 拆分写入 `--output-root` 下的 `training_set/validation_set/test_set`
  子目录，并追加 `design_id`、`part_uid` 字段。
- **配置文件**：`mycmd/slipt_csv_set/task_config.yaml`（控制文件队列、并发度、日志等）。
- **运行时目录**：`mycmd/slipt_csv_set/runtime/`（默认队列与中间结果存放处，可通过
  CLI 的 `--queue-root` 覆盖）。

## infer_context/random
- **入口**：`python mycmd/infer_context/random/select_random_faces.py`
- **主要输入**：`--input-root` 指向上一步输出的 `training_set/validation_set/test_set`
  目录。
- **主要输出**：在 `--output-dir` 下生成 `sampled_faces.csv`，包含随机抽取的面级样本，
  并携带原始行信息。
- **配置方式**：仅通过命令行参数指定样本量、随机种子等，无额外配置文件。

## server/produce_server
- **入口**：`python mycmd/server/produce_server/task_producer.py`
- **主要输入**：`--input-dir`（默认 `mycmd/server/sample_data`）中待推理的 CSV；可选
  的 `--config`（默认 `mycmd/server/produce_server/producer_config.yaml`）用于覆盖批次
  策略与字段映射。
- **主要输出**：写入 `--queue-dir`（默认 `mycmd/server/infer_server/runtime/queue`）
  的任务队列文件，每条任务包含预测路径、格式、列配置等元数据。
- **关联配置**：`producer_config.yaml` 会引用推理侧配置
  `mycmd/server/infer_server/worker_config.yaml`，确保任务定义与 worker 行为一致。

## server/infer_server
- **入口**：`python mycmd/server/infer_server/single_gpu_worker.py`（或以守护进程方式常驻）
- **主要输入**：读取 `CAD_TASK_CONFIG`（默认即同目录下 `worker_config.yaml`），内部
  的 `task_queue_path` 指向上述文件队列；上下文样本通过
  `context_config_path`（默认 `mycmd/server/infer_server/context_data.yaml`）描述。
- **主要输出**：推理结果写入 `worker_config.yaml` 中配置的 `output_root`
  （默认 `mycmd/server/infer_server/runtime/results` 或覆盖后的绝对路径）。
- **重要目录**  
  - `mycmd/server/infer_server/runtime/queue`：文件队列工作目录  
  - `mycmd/server/infer_server/runtime/results`：推理结果默认落盘位置  
  - `mycmd/server/context_data/`：示例上下文 CSV  
  - `mycmd/server/sample_data/`：示例预测 CSV

## 使用顺序建议
1. **数据划分**：运行 `slipt_csv_set` 生成标准化的训练/验证/测试 CSV。
2. **上下文抽样**（可选）：使用 `infer_context/random` 构建轻量上下文集。
3. **任务生产**：执行 `server/produce_server` 将预测 CSV 写入队列。
4. **推理消费**：启动 `server/infer_server` 的 GPU worker 拉取队列并生成结果。

整个流水线可通过调整上述配置文件适配不同数据源与部署环境，保持路径、队列名称与
输出目录一致即可快速复用。

