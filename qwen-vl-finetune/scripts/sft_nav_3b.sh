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
print_data_stats=${PRINT_DATA_STATS:-1}
dataloader_num_workers=${DATALOADER_NUM_WORKERS:-2}

# Training entry point
entry_file=qwenvl/train/train_qwen.py

# Dataset configuration
# Default co-training mix:
# - Navigation: LLAVA_NAV_V8, v4-150k, VLNCE-style oracle/self-generated, RxR/RVR/SOON graph data
# - General image-text: ALFRED, CLEVR-Change, Multi-VQA, NLVR2, ScanNet, Spot-the-Diff
# - Video QA: ActivityNetQA and NextQA slices already extracted under /root/autodl-tmp/co-training-data
# - Instruction generation: R2R/RVR/RxR/SOON path summarization data
# Override DATASETS to change sampling or composition, e.g. "llava_nav_v8%50,m4_instruct_alfred_subset".
# datasets=${DATASETS:-llava_nav_v8,llava_nav_v4_150k,finetuning_data_gen,finetuning_data_gen_rxr_compressed,mp3d_anno_rvr,mp3d_anno_rxr,mp3d_anno_soon,m4_instruct_alfred_subset,m4_instruct_clevr_change_subset,m4_instruct_multi_vqa_subset,m4_instruct_nlvr2_subset,m4_instruct_scannet_subset,m4_instruct_spot_the_diff_subset,video_0_30_s_activitynetqa_oe,video_0_30_s_nextqa_oe,video_0_30_s_nextqa_mc,video_1_2_m_activitynetqa_oe,video_1_2_m_nextqa_oe,video_1_2_m_nextqa_mc,video_2_3_m_activitynetqa_oe,video_30_60_s_activitynetqa_oe,video_30_60_s_nextqa_oe,video_30_60_s_nextqa_mc,instruction_gen_r2r,instruction_gen_rvr,instruction_gen_rxr,instruction_gen_soon}
datasets=${DATASETS:-llava_nav_v8}
# Output configuration
run_name=${RUN_NAME:-qwen25_vl_3b_lr1e-5_epoch1_ac4_text_geo_mp3d_v2_shift_pano_6-8}
output_dir=${OUTPUT_DIR:-/root/autodl-tmp/output/qwen25_vl_3b_lr1e-5_epoch1_ac4_text_geo_mp3d_v2_shift_pano_6-8}
report_to=${REPORT_TO:-none}
video_reader_backend=${VIDEO_READER_BACKEND:-decord}

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
    --max_pixels 50176
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
    --model_max_length 32768
    --gradient_checkpointing True
    --dataloader_num_workers "${dataloader_num_workers}"
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
echo "  video_reader_backend: ${video_reader_backend}"

if [[ "${print_data_stats}" == "1" ]]; then
  echo "Training data statistics:"
  PYTHONPATH="$(pwd):${PYTHONPATH:-}" \
  DATASETS="${datasets}" \
  BATCH_SIZE="${batch_size}" \
  GRAD_ACCUM_STEPS="${grad_accum_steps}" \
  NPROC_PER_NODE="${NPROC_PER_NODE}" \
  NNODES="${NNODES}" \
  NUM_TRAIN_EPOCHS="${num_train_epochs}" \
  MAX_STEPS="${MAX_STEPS:-}" \
  python - <<'PY'
import math
import os
from qwenvl.data import dataset_statistics

dataset_names = [name.strip() for name in os.environ["DATASETS"].split(",") if name.strip()]
stats = dataset_statistics(dataset_names)

batch_size = int(os.environ["BATCH_SIZE"])
grad_accum_steps = int(os.environ["GRAD_ACCUM_STEPS"])
nproc_per_node = int(os.environ["NPROC_PER_NODE"])
nnodes = int(os.environ["NNODES"])
num_train_epochs = float(os.environ["NUM_TRAIN_EPOCHS"])
max_steps_env = os.environ.get("MAX_STEPS", "").strip()

global_batch_size = batch_size * grad_accum_steps * nproc_per_node * nnodes
total_effective = stats["total_effective_count"]

name_width = max(
    len("dataset"),
    max((len(item["dataset_name"]) for item in stats["datasets"]), default=0),
)
raw_width = max(
    len("raw"),
    max((len(str(item["raw_count"])) for item in stats["datasets"]), default=1),
)
rate_width = max(
    len("rate"),
    max((len(f'{item["sampling_rate"]:.2f}') for item in stats["datasets"]), default=4),
)
eff_width = max(
    len("effective"),
    max((len(str(item["effective_count"])) for item in stats["datasets"]), default=1),
)
type_width = len("type")

header = (
    f"  {'dataset':<{name_width}}  "
    f"{'type':<{type_width}}  "
    f"{'rate':>{rate_width}}  "
    f"{'raw':>{raw_width}}  "
    f"{'effective':>{eff_width}}"
)
print(header)
print(f"  {'-' * len(header.strip())}")
for item in stats["datasets"]:
    dataset_type = "nav" if item["nav_graph"] else "general"
    raw_count = item["raw_count"] if item["raw_count"] is not None else "missing"
    effective_count = (
        item["effective_count"] if item["effective_count"] is not None else "missing"
    )
    print(
        f"  {item['dataset_name']:<{name_width}}  "
        f"{dataset_type:<{type_width}}  "
        f"{item['sampling_rate']:>{rate_width}.2f}  "
        f"{str(raw_count):>{raw_width}}  "
        f"{str(effective_count):>{eff_width}}"
    )

steps_per_epoch = math.ceil(total_effective / global_batch_size) if total_effective > 0 else 0
estimated_total_steps = math.ceil(steps_per_epoch * num_train_epochs)

print("  summary:")
print(f"    total raw samples: {stats['total_raw_count']}")
print(f"    total effective samples: {total_effective}")
print(f"    per_device_train_batch_size: {batch_size}")
print(f"    gradient_accumulation_steps: {grad_accum_steps}")
print(f"    world_size: {nproc_per_node * nnodes}")
print(f"    global batch size: {global_batch_size}")
print(f"    num_train_epochs: {num_train_epochs}")
print(f"    estimated steps per epoch: {steps_per_epoch}")
if max_steps_env:
    print(f"    max_steps override: {max_steps_env}")
else:
    print(f"    estimated total optimizer steps: {estimated_total_steps}")

if stats["missing_datasets"]:
    print("    missing annotation files:")
    for item in stats["missing_datasets"]:
        print(f"      - {item['dataset_name']}: {item['annotation_path']}")
    raise SystemExit(1)
PY
fi

export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
export NCCL_ASYNC_ERROR_HANDLING=${NCCL_ASYNC_ERROR_HANDLING:-1}
export FORCE_QWENVL_VIDEO_READER="${video_reader_backend}"

torchrun --nnodes="${NNODES}" \
         --node_rank="${NODE_RANK}" \
         --nproc_per_node="${NPROC_PER_NODE}" \
         --master_addr="${MASTER_ADDR}" \
         --master_port="${MASTER_PORT}" \
         "${entry_file}" "${args[@]}"
