# Qwen2.5-VL / Qwen3-VL Navigation Graph Port

## Scope

This port brings the ETPNav / TagaVLM navigation-graph training path onto the local `qwen-vl-finetune` pipeline for `Qwen2.5-VL` and `Qwen3-VL`.
Qwen2.5 remains the reference path; Qwen3 reuses the same data, batch structure, and launch script.

Implemented goals:

- Load navigation supervision samples together with normal VQA samples in the same dataset / batch.
- Resolve navigation candidate-view ids into stitched images using `cand_viewids_list.json`.
- Load node-level pair-distance matrices from `.npz`.
- Expand node-level distances into token-level `graph_sprels`.
- Add a per-layer learnable `sprel_linear` in the Qwen2.5-VL / Qwen3-VL text decoder.
- Merge graph bias into the decoder attention mask.
- Force navigation-graph training onto `sdpa` and disable packing / flatten mode.

Not enabled by default:

- `use_geo_token` is not added into the active path here. This remains unverified and should stay off.

## Files Changed

- `qwen-vl-finetune/qwenvl/data/__init__.py`
  - Added named dataset presets:
    - `llava_nav_v8`
    - `tagavlm_dagger_r2r_20260412_193919`
  - Added support for passing a direct local JSON path in `dataset_use`.

- `qwen-vl-finetune/qwenvl/data/nav_utils.py`
  - New helper module.
  - Handles:
    - candidate-view image resolution
    - `.npz` loading
    - token-to-node mapping
    - node-distance expansion into token-level `graph_sprels`

- `qwen-vl-finetune/qwenvl/data/data_processor.py`
  - Supports candidate-view stitching during message build.
  - Attaches navigation metadata per sample.
  - Builds per-sample `graph_sprels`.
  - Pads `graph_sprels` in the collator for mixed navigation / VQA batches.

- `qwen-vl-finetune/qwenvl/train/argument.py`
  - Added `DataArguments.use_nav_graph`.
  - Added `TrainingArguments.attn_implementation`.

- `qwen-vl-finetune/qwenvl/train/train_qwen.py`
  - Detects nav-graph mode.
  - Forces:
    - `attn_implementation=sdpa`
    - `data_flatten=False`
    - `data_packing=False`
  - Enables `config.graph_sprels` for Qwen2.5-VL / Qwen3-VL.
  - Keeps other model families blocked.

- `transformers-4.57.0/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py`
  - Added `sprel_linear` to `Qwen2_5_VLAttention`.
  - Added `graph_sprels` plumbing through:
    - `Qwen2_5_VLForConditionalGeneration.forward`
    - `Qwen2_5_VLModel.forward`
    - `Qwen2_5_VLTextModel.forward`
    - `Qwen2_5_VLDecoderLayer.forward`
    - `Qwen2_5_VLAttention.forward`
    - `prepare_inputs_for_generation`
  - Graph bias is added on top of the causal / padding attention mask.

- `transformers-4.57.0/src/transformers/models/qwen3_vl/modeling_qwen3_vl.py`
  - Mirrors the Qwen2.5 nav-graph path for Qwen3 text attention / model forward.

- `transformers-4.57.0/src/transformers/models/qwen3_vl_moe/modeling_qwen3_vl_moe.py`
  - Mirrors the same nav-graph path for Qwen3 MoE variants.

- `qwen-vl-finetune/scripts/sft_nav_3b.sh`
  - Same entry script for Qwen2.5 and Qwen3.
  - Switch models by overriding `LLM_MODEL`.

## Backups

Backups were created before editing:

- `qwen-vl-finetune/qwenvl/data/__init__.py.bak`
- `qwen-vl-finetune/qwenvl/data/data_processor.py.bak`
- `qwen-vl-finetune/qwenvl/train/argument.py.bak`
- `qwen-vl-finetune/qwenvl/train/train_qwen.py.bak`
- `transformers-4.57.0/src/transformers/models/qwen2_5_vl/modeling_qwen2_5_vl.py.bak`

## Data Path Assumptions

Current presets assume these paths:

- LLaVA-Nav v8 annotations:
  - `/root/autodl-fs/v8/llava_nav_instruct_train.json`
- LLaVA-Nav v8 candidate-view metadata:
  - `/root/autodl-fs/v8/cand_viewids_list_train.json`
- Matterport candidate-view hdf5:
  - `/root/autodl-fs/view_images_bgr_from_mattersim.h5`
- Matterport skybox root:
  - `/root/autodl-tmp/mp3d_data_cropped`
- DAgger R2R annotations:
  - `/home/data/ljx/tagavlm_dager_data/finetuning_data_gen_r2r/20260412_193919/finetuning_lables.json`
- DAgger R2R pair-distance npz:
  - `/home/data/ljx/tagavlm_dager_data/finetuning_data_gen_r2r/20260412_193919/gmap_pair_dist.npz`

If your layout changes, update the dataset preset in `qwen-vl-finetune/qwenvl/data/__init__.py`.

## Mixed-Batch Behavior

This port explicitly supports mixing:

- navigation samples
- single-image VQA
- multi-image VQA
- video samples

within the same batch.

Behavior:

- navigation samples get a non-zero `graph_sprels` matrix
- non-navigation samples get an all-zero `graph_sprels`
- because `sprel_linear` uses `bias=False`, zero graph inputs produce zero additive bias

This avoids the old TagaVLM limitation where mixed batches effectively degraded into `batch_size=1`.

## Training Notes

For navigation graph training, do not use flash attention.

Reason:

- the graph bias is implemented as additive attention bias
- the current path relies on `sdpa`
- `flash_attention_2` is intentionally bypassed for this mode

Recommended launch pattern:

```bash
cd qwen-vl-finetune
LLM_MODEL=/root/autodl-tmp/Qwen2.5-VL-3B-Instruct bash scripts/sft_nav_3b.sh
LLM_MODEL=/root/autodl-tmp/Qwen3-VL-4B-Instruct RUN_NAME=qwen3vl_4b_nav_graph bash scripts/sft_nav_3b.sh
```

Even if `--attn_implementation` is omitted, nav mode will force `sdpa`.

## Validation Done

Completed:

- Python syntax check with `py_compile` for the modified training/data/model files.

Not completed in this environment:

- full runtime training test
- actual sample decoding
- hdf5 / numpy runtime validation inside the active Python environment

The current shell Python here does not have all training dependencies installed, so runtime verification still needs to be done in the real training environment.

## Known Limits

- Navigation graph bias is implemented for `Qwen2.5-VL` and `Qwen3-VL`.
- `use_geo_token` is intentionally left inactive and should remain off unless separately validated.
- Navigation graph mode currently disables `data_flatten` / `data_packing`.
- The implementation assumes one graph-distance matrix per sample id in the configured `.npz`.
