#!/bin/bash
set -euo pipefail

# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NNODES=${NNODES:-${WORLD_SIZE:-1}}
NODE_RANK=${NODE_RANK:-0}
NPROC_PER_NODE=${NPROC_PER_NODE:-$(python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
)}

# DeepSpeed configuration
# DeepSpeed is optional here. The current Blackwell stack is stable with plain DDP,
# while DeepSpeed ZeRO on this exact model path still triggers CUDA faults.
use_deepspeed=${USE_DEEPSPEED:-0}
deepspeed=${DEEPSPEED_CONFIG:-./scripts/zero3.json}

# Model configuration
# Override LLM_MODEL to switch between Qwen2.5-VL-3B-Instruct and Qwen3-VL-4B-Instruct.
llm=${LLM_MODEL:-"/root/autodl-fs/Qwen2.5-VL-3B-Instruct"}

# Training hyperparameters
lr=${LR:-1e-5}
batch_size=${BATCH_SIZE:-2}
grad_accum_steps=${GRAD_ACCUM_STEPS:-4}
num_train_epochs=${NUM_TRAIN_EPOCHS:-1.0}

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Dataset configuration
# Default mix:
# - llava_nav_v8: original nav supervision with candidate-view stitching
# - tagavlm_dagger_r2r_20260412_193919: dagger-generated nav data
# You can append other VQA/video datasets by overriding DATASETS env var.
datasets=${DATASETS:-llava_nav_v8}

# Output configuration
run_name=${RUN_NAME:-qwen2_5_vl_3b_nav_graph}
output_dir=${OUTPUT_DIR:-/root/autodl-tmp/output/qwen25_vl_3b_nav_graph_ddp_test}
report_to=${REPORT_TO:-none}

args=(
    --model_name_or_path "${llm}"
    --dataset_use "${datasets}"
    --use_nav_graph True
    --attn_implementation sdpa
    --tune_mm_vision False
    --tune_mm_mlp True
    --tune_mm_llm True
    --bf16
    --optim adamw_torch
    --output_dir "${output_dir}"
    --num_train_epochs "${num_train_epochs}"
    --per_device_train_batch_size "${batch_size}"
    --per_device_eval_batch_size "${batch_size}"
    --gradient_accumulation_steps "${grad_accum_steps}"
    --max_pixels 100352
    --min_pixels 784
    --eval_strategy no
    --save_strategy steps
    --save_steps 400
    --save_total_limit 1
    --learning_rate "${lr}"
    --weight_decay 0
    --warmup_ratio 0.03
    --max_grad_norm 1
    --lr_scheduler_type cosine
    --logging_steps 1
    --model_max_length 8192
    --gradient_checkpointing True
    --dataloader_num_workers 4
    --run_name "${run_name}"
    --report_to "${report_to}"
    --ddp_find_unused_parameters False
)

if [[ "${use_deepspeed}" == "1" ]]; then
    args+=(--deepspeed "${deepspeed}")
fi

if [[ -n "${MAX_STEPS:-}" ]]; then
    args+=(--max_steps "${MAX_STEPS}")
fi

echo "Launching navigation-graph SFT with:"
echo "  model: ${llm}"
echo "  datasets: ${datasets}"
echo "  output: ${output_dir}"
echo "  attn: sdpa"
echo "  deepspeed: ${use_deepspeed}"
echo "  nnodes: ${NNODES}"
echo "  node_rank: ${NODE_RANK}"
echo "  nproc_per_node: ${NPROC_PER_NODE}"

export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}

torchrun --nnodes="${NNODES}" \
         --node_rank="${NODE_RANK}" \
         --nproc_per_node="${NPROC_PER_NODE}" \
         --master_addr="${MASTER_ADDR}" \
         --master_port="${MASTER_PORT}" \
         "${entry_file}" "${args[@]}"
