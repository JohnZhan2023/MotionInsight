#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 ALIGNED_MODEL GRPO_JSONL OUTPUT_DIR" >&2
  exit 2
fi

ALIGNED_MODEL=$1
DATASET=$2
OUTPUT_DIR=$3
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

cd "$ROOT_DIR"
python scripts/patch_transformers.py
python scripts/patch_transformers.py --check

export DEBUG_MODE=${DEBUG_MODE:-false}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/triton_cache_${USER:-motioninsight}}
mkdir -p "$TRITON_CACHE_DIR" "$OUTPUT_DIR"

# Paper setting: GRPO for 5 epochs, G=8 and beta=0.001 on 8 A100 GPUs.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7} \
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
  --master_addr=${MASTER_ADDR:-127.0.0.1} \
  --master_port=${MASTER_PORT:-12365} \
  -m training.train_grpo \
  --model_name_or_path "$ALIGNED_MODEL" \
  --dataset_name "$DATASET" \
  --output_dir "$OUTPUT_DIR" \
  --deepspeed training/configs/zero3.json \
  --max_prompt_length 8192 \
  --max_completion_length 384 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-6 \
  --lr_scheduler_type cosine \
  --weight_decay 0.01 \
  --bf16 true \
  --logging_steps 1 \
  --report_to none \
  --gradient_checkpointing true \
  --temporal false \
  --len_control true \
  --attn_implementation "${ATTN_IMPLEMENTATION:-flash_attention_2}" \
  --max_pixels 262144 \
  --num_train_epochs 5 \
  --run_name MotionInsight-multidim-all \
  --save_steps 500 \
  --beta 0.001 \
  --reward_funcs accuracy \
  --max_grad_norm 5 \
  --save_only_model false \
  --num_generations 8 \
  --dataloader_drop_last true
