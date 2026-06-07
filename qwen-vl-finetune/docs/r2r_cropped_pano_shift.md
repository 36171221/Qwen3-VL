# R2R Cropped Panorama Shift Labels

## Scope

This document describes the cropped panorama shift support used by the
`llava_nav_v8` dataset preset.

The shift labels are generated for R2R historical/current panoramas built from
the official Matterport skybox merge with `skybox0` and `skybox5` removed:

```text
mp3d_data_cropped/{scan}/matterport_skybox_images/{viewpoint}_skybox_small_cropped.jpg
```

The labels should not be applied to MatterSim candidate-view images, candidate
node ids such as `scan_viewpoint`, or non-cropped skybox debug panoramas.

## Dataset Switch

`qwen-vl-finetune/qwenvl/data/__init__.py` exposes the shift feature as dataset
config fields, matching the style of `nav_graph` and `use_geo_token`:

```python
LLAVA_NAV_V8 = {
    "annotation_path": "/root/autodl-tmp/co-training-data/v8/llava_nav_instruct_train_co.json",
    "data_path": "/root/autodl-tmp/co-training-data",
    "nav_graph": True,
    "use_geo_token": False,
    "use_pano_shift": True,
    "pano_shift_path": "/root/autodl-tmp/co-training-data/v8/R2R_train_enc_skybox_cropped_shift.json",
    "nav_pair_dist_path": "/root/autodl-tmp/co-training-data/v8/gmap_pair_dist_train.npz",
    "nav_pos_fts_path": "/root/autodl-tmp/co-training-data/v8/gmap_pos_fts_train.npz",
    "cand_viewids_path": "/root/autodl-tmp/co-training-data/v8/cand_viewids_list_train.json",
    "nav_view_root": "/root/autodl-tmp/co-training-data/view_images_bgr_from_mattersim",
}
```

To disable the feature for an ablation, set:

```python
"use_pano_shift": False
```

When enabled, `pano_shift_path` must point to the matching R2R split shift
label file on the training machine.

## Runtime Behavior

During message construction, `data_processor.py` passes `pano_shift_path` into
`resolve_nav_image()` only when the sample has `use_pano_shift=True`.

`nav_utils.py` then:

- loads the shift labels once with an LRU cache
- extracts `path_id` from sample ids such as `{path_id}_{instr_idx}_{step_idx}`
- detects direct cropped panorama paths containing `mp3d_data_cropped/`
- looks up the roll by `path_id + image_rel_path`, with a viewpoint fallback
- applies the roll before returning the PIL image

The applied operation is:

```python
aligned = np.roll(raw_cropped_pano, roll_px_applied, axis=1)
```

Use `roll_px_applied` directly from the label. Do not negate it again.

Candidate images are resolved only after direct image loading fails, so
MatterSim candidate views remain unchanged.

## Label Files

The generated labels are stored in the data-generation project at:

```text
/home/ljx/mp3d_data_gen/data/mp3d_data_cropped_shift_annotations
```

For the current `llava_nav_v8` training preset, copy or mount this file to the
path configured by `pano_shift_path`:

```text
R2R_train_enc_skybox_cropped_shift.json
```

Other generated splits are available for validation or analysis:

```text
R2R_val_seen_enc_skybox_cropped_shift.json
R2R_val_unseen_enc_skybox_cropped_shift.json
R2R_val_train_seen_enc_skybox_cropped_shift.json
R2R_test_enc_skybox_cropped_shift.json
```

## Label Semantics

Each step contains the cropped panorama path and the exact roll to apply:

```json
{
  "step": 0,
  "scan": "VLzqgDo317F",
  "viewpoint": "af3af33b0120469c9a00daa0d0b36799",
  "image_rel_path": "VLzqgDo317F/matterport_skybox_images/af3af33b0120469c9a00daa0d0b36799_skybox_small_cropped.jpg",
  "target_heading_deg": 214.916,
  "cropped_center_heading_deg": 169.919,
  "shift_px": 256,
  "roll_px_applied": -256
}
```

Step heading convention:

- `step = 0`: target heading comes from the R2R start heading
- `step > 0`: target heading is the XY motion heading from previous viewpoint
  to current viewpoint

This matches the historical/current panorama sequence used by
`get_sap_data_with_graph_full_path_v8.py`.

## Local Validation

Syntax check:

```bash
python -m py_compile \
  qwen-vl-finetune/qwenvl/data/nav_utils.py \
  qwen-vl-finetune/qwenvl/data/data_processor.py \
  qwen-vl-finetune/qwenvl/data/__init__.py
```

Minimal behavior check:

```python
from pathlib import Path
import json
import numpy as np
from PIL import Image
from qwenvl.data.nav_utils import resolve_nav_image

base = Path("/home/ljx/LLaVA-NeXT-graph/data/co-training_data")
anno = Path("/home/ljx/mp3d_data_gen/data/mp3d-anno-80k-r2r-text-geo-v2/llava_nav_instruct_train_large_v4.json")
shift = Path("/home/ljx/mp3d_data_gen/data/mp3d_data_cropped_shift_annotations/R2R_train_enc_skybox_cropped_shift.json")

item = json.load(open(anno, encoding="utf-8"))[0]
image_spec = item["image"][0]
aligned = resolve_nav_image(
    image_spec,
    base,
    sample_id=item["id"],
    pano_shift_path=str(shift),
)

labels = json.load(open(shift, encoding="utf-8"))
step = labels["items"][0]["steps"][0]
raw = Image.open(base / "mp3d_data_cropped" / step["image_rel_path"]).convert("RGB")
manual = np.roll(np.asarray(raw), int(step["roll_px_applied"]), axis=1)

assert np.array_equal(np.asarray(aligned), manual)
```
