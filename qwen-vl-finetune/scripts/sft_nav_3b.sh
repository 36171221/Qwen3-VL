#!/bin/bash

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${WORLD_SIZE:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}

# DeepSpeed configuration
deepspeed=./scripts/zero3.json

# Model configuration
llm=${LLM_MODEL:-"/root/ljx_home/Qwen3-VL/data/Qwen2.5-VL-3B-Instruct"}

# Training hyperparameters
lr=${LR:-1e-5}
batch_size=${BATCH_SIZE:-2}
grad_accum_steps=${GRAD_ACCUM_STEPS:-8}
num_train_epochs=${NUM_TRAIN_EPOCHS:-1.0}

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Dataset configuration
# Default mix:
# - llava_nav_v4: original nav supervision with candidate-view stitching
# - tagavlm_dagger_r2r_20260412_193919: dagger-generated nav data
# You can append other VQA/video datasets by overriding DATASETS env var.
datasets=${DATASETS:-tagavlm_dagger_r2r_20260412_193919}

# Output configuration
run_name=${RUN_NAME:-qwen2_5_vl_3b_nav_graph}
output_dir=${OUTPUT_DIR:-./output/qwen2_5_vl_3b_nav_graph}
report_to=${REPORT_TO:-none}

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path ${llm} \
    --dataset_use ${datasets} \
    --use_nav_graph True \
    --attn_implementation sdpa \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs ${num_train_epochs} \
    --per_device_train_batch_size ${batch_size} \
    --per_device_eval_batch_size ${batch_size} \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 50176 \
    --min_pixels 784 \
    --eval_strategy no \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to ${report_to}"

echo "Launching navigation-graph SFT with:"
echo "  model: ${llm}"
echo "  datasets: ${datasets}"
echo "  output: ${output_dir}"
echo "  attn: sdpa"

CUDA_VISIBLE_DEVICES=1 torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}
