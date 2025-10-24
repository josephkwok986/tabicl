# train_stage2_from_offline_prior.sh
#!/usr/bin/env bash
set -euo pipefail

# Launch TabICL Stage-2 training using OFFLINE prior data produced by src/tabicl/prior/genload.py
# Usage:
#   bash scripts/train_stage2_from_offline_prior.sh PRIOR_DIR [STEPS] [GPUS]
# Example:
#   CUDA_VISIBLE_DEVICES=1,2,3 bash ./train_stage2_from_offline_prior.sh 1,2,3
#   ./train_stage2_from_offline_prior.sh 1,2,3

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 PRIOR_DIR [STEPS] [GPUS]"
  exit 2
fi

GPUS="${1:-1,2,3}"

export CUDA_VISIBLE_DEVICES="${GPUS}"
echo "CUDA_VISIBLE_DEVICES = ${CUDA_VISIBLE_DEVICES}"

# Loading from disk and training
python -m torch.distributed.run --standalone --nproc_per_node=3 /home/amax/gjj_tabicl/tabicl/src/tabicl/train/run.py \
            --wandb_log False \
            --wandb_project TabICL \
            --wandb_name Stage2 \
            --wandb_dir /home/amax/gjj_tabicl/scripts2/other \
            --wandb_mode disabled \
            --device cuda \
            --amp True \
            --dtype float16 \
            --np_seed 42 \
            --torch_seed 42 \
            --max_steps 25 \
            --batch_size 512 \
            --micro_batch_size 1 \
            --lr 2e-5 \
            --scheduler polynomial_decay_warmup \
            --warmup_steps 0 \
            --warmup_proportion 0 \
            --poly_decay_lr_end 5e-6 \
            --poly_decay_power 2.0 \
            --gradient_clipping 1.0 \
            --prior_dir /home/amax/gjj_tabicl/scripts2/tabicl_prior \
            --load_prior_start 0 \
            --delete_after_load False \
            --prior_device cpu \
            --embed_dim 128 \
            --col_num_blocks 3 \
            --col_nhead 4 \
            --col_num_inds 128 \
            --row_num_blocks 3 \
            --row_nhead 8 \
            --row_num_cls 4 \
            --row_rope_base 100000 \
            --icl_num_blocks 12 \
            --icl_nhead 4 \
            --ff_factor 2 \
            --norm_first True \
            --checkpoint_dir /home/amax/gjj_tabicl/scripts2/other/checkpoint/dir \
            --min_seq_len 1000  \
            --max_seq_len 40000 \
            --log_seq_len True