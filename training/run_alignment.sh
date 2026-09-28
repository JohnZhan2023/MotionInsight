#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 BASE_MODEL ALIGNMENT_JSONL OUTPUT_DIR" >&2
  exit 2
fi

BASE_MODEL=$1
DATASET=$2
OUTPUT_DIR=$3
ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)

cd "$ROOT_DIR"
python scripts/patch_transformers.py
python scripts/patch_transformers.py --check

export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-/tmp/triton_cache_${USER:-motioninsight}}
mkdir -p "$TRITON_CACHE_DIR" "$OUTPUT_DIR"

# Paper setting: modality alignment for 3 epochs on 8 A100 GPUs at 1e-6.
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7} \
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 \
  --master_addr=${MASTER_ADDR:-127.0.0.1} \
  --master_port=${MASTER_PORT:-12366} \
  -m training.train_alignment \
  --model_name_or_path "$BASE_MODEL" \
  --dataset_name "$DATASET" \
  --output_dir "$OUTPUT_DIR" \
  --deepspeed training/configs/zero2.json \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 1e-6 \
  --lr_scheduler_type cosine \
  --weight_decay 0.01 \
  --logging_steps 1 \
  --bf16 true \
  --report_to none \
  --gradient_checkpointing true \
  --attn_implementation "${ATTN_IMPLEMENTATION:-flash_attention_2}" \
  --num_train_epochs 3 \
  --save_strategy epoch \
  --run_name MotionInsight-alignment \
  --max_grad_norm 5 \
  --save_only_model false
