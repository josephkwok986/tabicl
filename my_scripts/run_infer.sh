#!/usr/bin/env bash
# run_infer.sh — 支持文件 & 目录输入
# 用法示例：
#  1) 单个 CSV：
#     ./run_infer.sh --gpu_ids 0 \
#       --train_csv data/synth/set_00/train.csv --test_csv data/synth/set_00/test.csv \
#       --target label --out_dir runs/infer_$(date +%Y%m%d_%H%M%S)
#
#  2) 指定一个“数据集目录”（目录下有 train.csv/test.csv）：
#     ./run_infer.sh --gpu_ids 0 \
#       --data_dir data/synth/set_00 --target label --out_dir runs/infer_$(date +%Y%m%d_%H%M%S)
#
#  3) 指定顶层目录（包含多个 set_xx 子目录）批量跑：
#     ./run_infer.sh --gpu_ids 0 \
#       --data_dir data/synth --target label --out_dir runs/infer_$(date +%Y%m%d_%H%M%S)
#
#  4) 也可把 --train_csv/--test_csv 指到目录，脚本会补成 train.csv/test.csv。

set -euo pipefail

GPU_IDS="0"
OUT_DIR="runs/tabicl_infer_$(date +%Y%m%d_%H%M%S)"
DATA_DIR=""

# 其余参数透传给 infer_tabicl.py
RAW_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu_ids) GPU_IDS="$2"; shift 2 ;;
    --out_dir) OUT_DIR="$2"; shift 2 ;;
    --data_dir) DATA_DIR="$2"; shift 2 ;;
    *) RAW_ARGS+=("$1"); shift ;;
  esac
done

mkdir -p "$OUT_DIR"

# 解析 RAW_ARGS 里的 --train_csv/--test_csv 并做“目录 -> 文件”的修正
TRAIN_ARG=""
TEST_ARG=""
OTHER_ARGS=()
i=0
while [[ $i -lt ${#RAW_ARGS[@]} ]]; do
  key="${RAW_ARGS[$i]}"
  if [[ "$key" == "--train_csv" && $((i+1)) -lt ${#RAW_ARGS[@]} ]]; then
    val="${RAW_ARGS[$((i+1))]}"
    if [[ -d "$val" ]]; then
      if [[ -f "$val/train.csv" ]]; then
        TRAIN_ARG="$val/train.csv"
      else
        echo "ERR: 目录 '$val' 下未找到 train.csv" >&2; exit 1
      fi
    else
      TRAIN_ARG="$val"
    fi
    i=$((i+2)); continue
  elif [[ "$key" == "--test_csv" && $((i+1)) -lt ${#RAW_ARGS[@]} ]]; then
    val="${RAW_ARGS[$((i+1))]}"
    if [[ -d "$val" ]]; then
      if [[ -f "$val/test.csv" ]]; then
        TEST_ARG="$val/test.csv"
      else
        echo "ERR: 目录 '$val' 下未找到 test.csv" >&2; exit 1
      fi
    else
      TEST_ARG="$val"
    fi
    i=$((i+2)); continue
  else
    OTHER_ARGS+=("$key")
    i=$((i+1))
  fi
done

run_one () {
  local train_csv="$1"
  local test_csv="${2:-}"
  local out_dir="$3"

  mkdir -p "$out_dir"
  echo "[run_infer] Using GPUs: ${GPU_IDS}" | tee -a "${out_dir}/run.log"
  echo "[run_infer] train_csv=${train_csv}" | tee -a "${out_dir}/run.log"
  if [[ -n "$test_csv" ]]; then echo "[run_infer] test_csv=${test_csv}" | tee -a "${out_dir}/run.log"; fi

  export CUDA_VISIBLE_DEVICES="${GPU_IDS}"

  # 组装 Python 参数
  PY_ARGS=( --train_csv "$train_csv" --out_dir "$out_dir" )
  if [[ -n "$test_csv" ]]; then PY_ARGS+=( --test_csv "$test_csv" ); fi
  PY_ARGS+=( "${OTHER_ARGS[@]}" )

  # 跑
  python infer_tabicl.py "${PY_ARGS[@]}" | tee -a "${out_dir}/run.log"
  echo "[run_infer] Done -> ${out_dir}"
}

# 优先使用 --data_dir（若提供）
if [[ -n "$DATA_DIR" ]]; then
  if [[ -f "$DATA_DIR/train.csv" ]]; then
    # 单数据集目录
    TEST_PATH=""
    [[ -f "$DATA_DIR/test.csv" ]] && TEST_PATH="$DATA_DIR/test.csv"
    run_one "$DATA_DIR/train.csv" "$TEST_PATH" "$OUT_DIR"
  else
    # 批量：遍历子目录（优先匹配 set_*，否则任何含 train.csv 的子目录）
    shopt -s nullglob
    candidates=( "$DATA_DIR"/set_* "$DATA_DIR"/* )
    seen=()
    for d in "${candidates[@]}"; do
      [[ -d "$d" ]] || continue
      [[ -f "$d/train.csv" ]] || continue
      # 去重
      skip=0
      for s in "${seen[@]:-}"; do [[ "$s" == "$d" ]] && skip=1 && break; done
      [[ $skip -eq 1 ]] && continue
      seen+=("$d")

      base="$(basename "$d")"
      sub_out="${OUT_DIR}/${base}"
      testp=""
      [[ -f "$d/test.csv" ]] && testp="$d/test.csv"
      run_one "$d/train.csv" "$testp" "$sub_out"
    done
    if [[ ${#seen[@]} -eq 0 ]]; then
      echo "ERR: 在 '$DATA_DIR' 下没有找到包含 train.csv 的子目录" >&2
      exit 1
    fi
  fi
  exit 0
fi

# 没有 --data_dir：按（可能已被目录->文件修正过的）--train_csv/--test_csv 跑一次
if [[ -z "$TRAIN_ARG" ]]; then
  echo "ERR: 未提供 --data_dir 或 --train_csv" >&2
  exit 1
fi

# 如果只给了目录形式的 --train_csv，且同目录下存在 test.csv，则自动补上
if [[ -z "$TEST_ARG" ]]; then
  dir_guess="$(dirname "$TRAIN_ARG")"
  if [[ -f "$dir_guess/test.csv" ]]; then
    TEST_ARG="$dir_guess/test.csv"
  fi
fi

run_one "$TRAIN_ARG" "$TEST_ARG" "$OUT_DIR"
