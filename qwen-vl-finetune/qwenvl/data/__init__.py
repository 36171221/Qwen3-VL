import re
import json
import math
from pathlib import Path

# Define placeholders for dataset paths
CAMBRIAN_737K = {
    "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION",
    "data_path": "",
}

CAMBRIAN_737K_PACK = {
    "annotation_path": f"PATH_TO_CAMBRIAN_737K_ANNOTATION_PACKED",
    "data_path": f"",
}

MP_DOC = {
    "annotation_path": "PATH_TO_MP_DOC_ANNOTATION",
    "data_path": "PATH_TO_MP_DOC_DATA",
}

CLEVR_MC = {
    "annotation_path": "PATH_TO_CLEVR_MC_ANNOTATION",
    "data_path": "PATH_TO_CLEVR_MC_DATA",
}

VIDEOCHATGPT = {
    "annotation_path": "PATH_TO_VIDEOCHATGPT_ANNOTATION",
    "data_path": "PATH_TO_VIDEOCHATGPT_DATA",
}

M4_INSTRUCT_NUSCENES_SUBSET = {
    "annotation_path": "/root/autodl-tmp/co-training-data/m4_instruct_nuscenes_subset.json",
    "data_path": "/root/autodl-tmp/co-training-data",
}

M4_INSTRUCT_ALFRED_SUBSET = {
    "annotation_path": "/root/autodl-tmp/co-training-data/m4_instruct_alfred_subset.json",
    "data_path": "/root/autodl-tmp/co-training-data",
}

M4_INSTRUCT_CLEVR_CHANGE_SUBSET = {
    "annotation_path": "/root/autodl-tmp/co-training-data/m4_instruct_clevr_change_subset.json",
    "data_path": "/root/autodl-tmp/co-training-data",
}

M4_INSTRUCT_MULTI_VQA_SUBSET = {
    "annotation_path": "/root/autodl-tmp/co-training-data/m4_instruct_multi_vqa_subset.json",
    "data_path": "/root/autodl-tmp/co-training-data",
}

M4_INSTRUCT_NLVR2_SUBSET = {
    "annotation_path": "/root/autodl-tmp/co-training-data/m4_instruct_nlvr2_subset.json",
    "data_path": "/root/autodl-tmp/co-training-data",
}

M4_INSTRUCT_SCANNET_SUBSET = {
    "annotation_path": "/root/autodl-tmp/co-training-data/m4_instruct_scannet_subset.json",
    "data_path": "/root/autodl-tmp/co-training-data",
}

M4_INSTRUCT_SPOT_THE_DIFF_SUBSET = {
    "annotation_path": "/root/autodl-tmp/co-training-data/m4_instruct_spot_the_diff_subset.json",
    "data_path": "/root/autodl-tmp/co-training-data",
}

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

LLAVA_NAV_V4_150K = {
    "annotation_path": "/root/autodl-tmp/co-training-data/v4-150k/llava_nav_instruct_train_v7_final_co.json",
    "data_path": "/root/autodl-tmp/co-training-data",
    "nav_graph": True,
<<<<<<< HEAD
    "nav_pair_dist_path": "/root/autodl-tmp/co-training-data/v4-150k/gmap_pair_dist_train_v7_combined.npz",
=======
    "use_geo_token": False,
    "nav_pair_dist_path": "/root/autodl-tmp/co-training-data/v4-150k/gmap_pair_dist_train_v7_combined.npz",
    "nav_pos_fts_path": "/root/autodl-tmp/co-training-data/v4-150k/gmap_pos_fts_train_v7_combined.npz",
>>>>>>> feature/pano-correction
    "cand_viewids_path": "/root/autodl-tmp/co-training-data/v4-150k/cand_viewids_list_train_v7_final.json",
    "nav_view_root": [
        "/root/autodl-tmp/co-training-data/view_images_bgr_from_mattersim",
        "/root/autodl-tmp/co-training-data/view_images_hm3d",
    ],
}

TAGAVLM_DAGGER_R2R_20260412_193919 = {
    "annotation_path": "/root/autodl-tmp/finetuning_data_gen_r2r/20260412_193919/finetuning_lables.json",
    "data_path": "/root/autodl-tmp",
    "nav_graph": True,
    "use_geo_token": False,
    "nav_pair_dist_path": "/root/autodl-tmp/finetuning_data_gen_r2r/20260412_193919/gmap_pair_dist.npz",
    "nav_pos_fts_path": "/root/autodl-tmp/finetuning_data_gen_r2r/20260412_193919/gmap_pos_fts.npz",
}

FINETUNING_DATA_GEN = {
    "annotation_path": "/root/autodl-tmp/co-training-data/finetuning_data_gen/finetuning_lables.json",
    "data_path": "/root/autodl-tmp/co-training-data/finetuning_data_gen/images",
    "nav_graph": True,
    "use_geo_token": False,
    "nav_pair_dist_path": "/root/autodl-tmp/co-training-data/finetuning_data_gen/gmap_pair_dist.npz",
    "nav_pos_fts_path": "/root/autodl-tmp/co-training-data/finetuning_data_gen/gmap_pos_fts.npz",
}

FINETUNING_DATA_GEN_RXR_COMPRESSED = {
    "annotation_path": "/root/autodl-tmp/co-training-data/finetuning_data_gen_rxr_compressed/finetuning_lables_co.json",
    "data_path": "/root/autodl-tmp/co-training-data/finetuning_data_gen_rxr_compressed/images",
    "nav_graph": True,
    "use_geo_token": False,
    "nav_pair_dist_path": "/root/autodl-tmp/co-training-data/finetuning_data_gen_rxr_compressed/gmap_pair_dist.npz",
    "nav_pos_fts_path": "/root/autodl-tmp/co-training-data/finetuning_data_gen_rxr_compressed/gmap_pos_fts.npz",
    "path_prefix_replacements": {
        "finetuning_data_gen_rxr/images/": "",
    },
}

MP3D_ANNO_RXR = {
    "annotation_path": "/root/autodl-tmp/co-training-data/mp3d-anno-rxr/llava_nav_instruct_rxr_train.json",
    "data_path": "/root/autodl-tmp/co-training-data",
    "nav_graph": True,
    "use_geo_token": False,
    "nav_pair_dist_path": "/root/autodl-tmp/co-training-data/mp3d-anno-rxr/gmap_pair_dist_rxr_train.npz",
    "nav_pos_fts_path": "/root/autodl-tmp/co-training-data/mp3d-anno-rxr/gmap_pos_fts_rxr_train.npz",
    "cand_viewids_path": "/root/autodl-tmp/co-training-data/mp3d-anno-rxr/cand_viewids_list_rxr_train.json",
    "nav_view_root": "/root/autodl-tmp/co-training-data/view_images_bgr_from_mattersim",
}

MP3D_ANNO_RVR = {
    "annotation_path": "/root/autodl-tmp/co-training-data/mp3d-anno-rvr/llava_nav_instruct_rvr_train.json",
    "data_path": "/root/autodl-tmp/co-training-data",
    "nav_graph": True,
    "use_geo_token": False,
    "nav_pair_dist_path": "/root/autodl-tmp/co-training-data/mp3d-anno-rvr/gmap_pair_dist_rvr_train.npz",
    "nav_pos_fts_path": "/root/autodl-tmp/co-training-data/mp3d-anno-rvr/gmap_pos_fts_rvr_train.npz",
    "cand_viewids_path": "/root/autodl-tmp/co-training-data/mp3d-anno-rvr/cand_viewids_list_rvr_train.json",
    "nav_view_root": "/root/autodl-tmp/co-training-data/view_images_bgr_from_mattersim",
}

MP3D_ANNO_SOON = {
    "annotation_path": "/root/autodl-tmp/co-training-data/mp3d-anno-soon/llava_nav_instruct_soon_train.json",
    "data_path": "/root/autodl-tmp/co-training-data",
    "nav_graph": True,
    "use_geo_token": False,
    "nav_pair_dist_path": "/root/autodl-tmp/co-training-data/mp3d-anno-soon/gmap_pair_dist_soon_train.npz",
    "nav_pos_fts_path": "/root/autodl-tmp/co-training-data/mp3d-anno-soon/gmap_pos_fts_soon_train.npz",
    "cand_viewids_path": "/root/autodl-tmp/co-training-data/mp3d-anno-soon/cand_viewids_list_soon_train.json",
    "nav_view_root": "/root/autodl-tmp/co-training-data/view_images_bgr_from_mattersim",
}

INSTRUCTION_GEN_R2R = {
    "annotation_path": "/root/autodl-tmp/co-training-data/instruction_gen_r2r.json",
    "data_path": "/root/autodl-tmp/co-training-data",
}

INSTRUCTION_GEN_RVR = {
    "annotation_path": "/root/autodl-tmp/co-training-data/instruction_gen_rvr.json",
    "data_path": "/root/autodl-tmp/co-training-data",
}

INSTRUCTION_GEN_RXR = {
    "annotation_path": "/root/autodl-tmp/co-training-data/instruction_gen_rxr.json",
    "data_path": "/root/autodl-tmp/co-training-data",
}

INSTRUCTION_GEN_SOON = {
    "annotation_path": "/root/autodl-tmp/co-training-data/instruction_gen_soon.json",
    "data_path": "/root/autodl-tmp/co-training-data",
}

FINETUNING_DATA_GEN = {
    "annotation_path": "/root/autodl-tmp/co-training-data/finetuning_data_gen/finetuning_lables.json",
    "data_path": "/root/autodl-tmp/co-training-data/finetuning_data_gen/images",
    "nav_graph": True,
    "nav_pair_dist_path": "/root/autodl-tmp/co-training-data/finetuning_data_gen/gmap_pair_dist.npz",
}

FINETUNING_DATA_GEN_RXR_COMPRESSED = {
    "annotation_path": "/root/autodl-tmp/co-training-data/finetuning_data_gen_rxr_compressed/finetuning_lables_co.json",
    "data_path": "/root/autodl-tmp/co-training-data/finetuning_data_gen_rxr_compressed/images",
    "nav_graph": True,
    "nav_pair_dist_path": "/root/autodl-tmp/co-training-data/finetuning_data_gen_rxr_compressed/gmap_pair_dist.npz",
    "path_prefix_replacements": {
        "finetuning_data_gen_rxr/images/": "",
    },
}

MP3D_ANNO_RXR = {
    "annotation_path": "/root/autodl-tmp/co-training-data/mp3d-anno-rxr/llava_nav_instruct_rxr_train.json",
    "data_path": "/root/autodl-tmp/co-training-data",
    "nav_graph": True,
    "nav_pair_dist_path": "/root/autodl-tmp/co-training-data/mp3d-anno-rxr/gmap_pair_dist_rxr_train.npz",
    "cand_viewids_path": "/root/autodl-tmp/co-training-data/mp3d-anno-rxr/cand_viewids_list_rxr_train.json",
    "nav_view_root": "/root/autodl-tmp/co-training-data/view_images_bgr_from_mattersim",
}

MP3D_ANNO_RVR = {
    "annotation_path": "/root/autodl-tmp/co-training-data/mp3d-anno-rvr/llava_nav_instruct_rvr_train.json",
    "data_path": "/root/autodl-tmp/co-training-data",
    "nav_graph": True,
    "nav_pair_dist_path": "/root/autodl-tmp/co-training-data/mp3d-anno-rvr/gmap_pair_dist_rvr_train.npz",
    "cand_viewids_path": "/root/autodl-tmp/co-training-data/mp3d-anno-rvr/cand_viewids_list_rvr_train.json",
    "nav_view_root": "/root/autodl-tmp/co-training-data/view_images_bgr_from_mattersim",
}

MP3D_ANNO_SOON = {
    "annotation_path": "/root/autodl-tmp/co-training-data/mp3d-anno-soon/llava_nav_instruct_soon_train.json",
    "data_path": "/root/autodl-tmp/co-training-data",
    "nav_graph": True,
    "nav_pair_dist_path": "/root/autodl-tmp/co-training-data/mp3d-anno-soon/gmap_pair_dist_soon_train.npz",
    "cand_viewids_path": "/root/autodl-tmp/co-training-data/mp3d-anno-soon/cand_viewids_list_soon_train.json",
    "nav_view_root": "/root/autodl-tmp/co-training-data/view_images_bgr_from_mattersim",
}

INSTRUCTION_GEN_R2R = {
    "annotation_path": "/root/autodl-tmp/co-training-data/instruction_gen_r2r.json",
    "data_path": "/root/autodl-tmp/co-training-data",
}

INSTRUCTION_GEN_RVR = {
    "annotation_path": "/root/autodl-tmp/co-training-data/instruction_gen_rvr.json",
    "data_path": "/root/autodl-tmp/co-training-data",
}

INSTRUCTION_GEN_RXR = {
    "annotation_path": "/root/autodl-tmp/co-training-data/instruction_gen_rxr.json",
    "data_path": "/root/autodl-tmp/co-training-data",
}

INSTRUCTION_GEN_SOON = {
    "annotation_path": "/root/autodl-tmp/co-training-data/instruction_gen_soon.json",
    "data_path": "/root/autodl-tmp/co-training-data",
}

VLNCE_ORACLE = {
    "annotation_path": "/root/autodl-fs/finetuning_lables_co.json",
    "data_path": "/root/autodl-tmp",
    "nav_graph": True,
    "use_geo_token": False,
    "nav_pair_dist_path": "/root/autodl-tmp/finetuning_data_gen_r2r/gmap_pair_dist.npz",
    "nav_pos_fts_path": "/root/autodl-tmp/finetuning_data_gen_r2r/gmap_pos_fts.npz",
}

VIDEO_0_30_S_ACTIVITYNETQA_OE = {
    "annotation_path": "/root/autodl-tmp/co-training-data/0_30_s_activitynetqa_oe_qa_processed.json",
    "data_path": "/root/autodl-tmp/co-training-data/0_30_s_activitynetqa",
}

VIDEO_0_30_S_NEXTQA_OE = {
    "annotation_path": "/root/autodl-tmp/co-training-data/0_30_s_nextqa_oe_qa_processed.json",
    "data_path": "/root/autodl-tmp/co-training-data/0_30_s_nextqa",
}

VIDEO_0_30_S_NEXTQA_MC = {
    "annotation_path": "/root/autodl-tmp/co-training-data/0_30_s_nextqa_mc_qa_processed.json",
    "data_path": "/root/autodl-tmp/co-training-data/0_30_s_nextqa",
}

VIDEO_1_2_M_ACTIVITYNETQA_OE = {
    "annotation_path": "/root/autodl-tmp/co-training-data/1_2_m_activitynetqa_oe_qa_processed.json",
    "data_path": "/root/autodl-tmp/co-training-data/1_2_m_activitynetqa",
}

VIDEO_1_2_M_NEXTQA_OE = {
    "annotation_path": "/root/autodl-tmp/co-training-data/1_2_m_nextqa_oe_qa_processed.json",
    "data_path": "/root/autodl-tmp/co-training-data/1_2_m_nextqa",
}

VIDEO_1_2_M_NEXTQA_MC = {
    "annotation_path": "/root/autodl-tmp/co-training-data/1_2_m_nextqa_mc_qa_processed.json",
    "data_path": "/root/autodl-tmp/co-training-data/1_2_m_nextqa",
}

VIDEO_2_3_M_ACTIVITYNETQA_OE = {
    "annotation_path": "/root/autodl-tmp/co-training-data/2_3_m_activitynetqa_oe_qa_processed.json",
    "data_path": "/root/autodl-tmp/co-training-data/2_3_m_activitynetqa",
}

VIDEO_30_60_S_ACTIVITYNETQA_OE = {
    "annotation_path": "/root/autodl-tmp/co-training-data/30_60_s_activitynetqa_oe_qa_processed.json",
    "data_path": "/root/autodl-tmp/co-training-data/30_60_s_activitynetqa",
}

VIDEO_30_60_S_NEXTQA_OE = {
    "annotation_path": "/root/autodl-tmp/co-training-data/30_60_s_nextqa_oe_qa_processed.json",
    "data_path": "/root/autodl-tmp/co-training-data/30_60_s_nextqa",
}

VIDEO_30_60_S_NEXTQA_MC = {
    "annotation_path": "/root/autodl-tmp/co-training-data/30_60_s_nextqa_mc_qa_processed.json",
    "data_path": "/root/autodl-tmp/co-training-data/30_60_s_nextqa",
}

VIDEO_0_30_S_ACTIVITYNETQA_OE = {
    "annotation_path": "/root/autodl-tmp/co-training-data/0_30_s_activitynetqa_oe_qa_processed.json",
    "data_path": "/root/autodl-tmp/co-training-data/0_30_s_activitynetqa",
}

VIDEO_0_30_S_NEXTQA_OE = {
    "annotation_path": "/root/autodl-tmp/co-training-data/0_30_s_nextqa_oe_qa_processed.json",
    "data_path": "/root/autodl-tmp/co-training-data/0_30_s_nextqa",
}

VIDEO_0_30_S_NEXTQA_MC = {
    "annotation_path": "/root/autodl-tmp/co-training-data/0_30_s_nextqa_mc_qa_processed.json",
    "data_path": "/root/autodl-tmp/co-training-data/0_30_s_nextqa",
}

VIDEO_1_2_M_ACTIVITYNETQA_OE = {
    "annotation_path": "/root/autodl-tmp/co-training-data/1_2_m_activitynetqa_oe_qa_processed.json",
    "data_path": "/root/autodl-tmp/co-training-data/1_2_m_activitynetqa",
}

VIDEO_1_2_M_NEXTQA_OE = {
    "annotation_path": "/root/autodl-tmp/co-training-data/1_2_m_nextqa_oe_qa_processed.json",
    "data_path": "/root/autodl-tmp/co-training-data/1_2_m_nextqa",
}

VIDEO_1_2_M_NEXTQA_MC = {
    "annotation_path": "/root/autodl-tmp/co-training-data/1_2_m_nextqa_mc_qa_processed.json",
    "data_path": "/root/autodl-tmp/co-training-data/1_2_m_nextqa",
}

VIDEO_2_3_M_ACTIVITYNETQA_OE = {
    "annotation_path": "/root/autodl-tmp/co-training-data/2_3_m_activitynetqa_oe_qa_processed.json",
    "data_path": "/root/autodl-tmp/co-training-data/2_3_m_activitynetqa",
}

VIDEO_30_60_S_ACTIVITYNETQA_OE = {
    "annotation_path": "/root/autodl-tmp/co-training-data/30_60_s_activitynetqa_oe_qa_processed.json",
    "data_path": "/root/autodl-tmp/co-training-data/30_60_s_activitynetqa",
}

VIDEO_30_60_S_NEXTQA_OE = {
    "annotation_path": "/root/autodl-tmp/co-training-data/30_60_s_nextqa_oe_qa_processed.json",
    "data_path": "/root/autodl-tmp/co-training-data/30_60_s_nextqa",
}

VIDEO_30_60_S_NEXTQA_MC = {
    "annotation_path": "/root/autodl-tmp/co-training-data/30_60_s_nextqa_mc_qa_processed.json",
    "data_path": "/root/autodl-tmp/co-training-data/30_60_s_nextqa",
}

data_dict = {
    "cambrian_737k": CAMBRIAN_737K,
    "cambrian_737k_pack": CAMBRIAN_737K_PACK,
    "mp_doc": MP_DOC,
    "clevr_mc": CLEVR_MC,
    "videochatgpt": VIDEOCHATGPT,
    "m4_instruct_alfred_subset": M4_INSTRUCT_ALFRED_SUBSET,
    "m4_instruct_clevr_change_subset": M4_INSTRUCT_CLEVR_CHANGE_SUBSET,
    "m4_instruct_multi_vqa_subset": M4_INSTRUCT_MULTI_VQA_SUBSET,
    "m4_instruct_nlvr2_subset": M4_INSTRUCT_NLVR2_SUBSET,
    "m4_instruct_nuscenes_subset": M4_INSTRUCT_NUSCENES_SUBSET,
    "m4_instruct_scannet_subset": M4_INSTRUCT_SCANNET_SUBSET,
    "m4_instruct_spot_the_diff_subset": M4_INSTRUCT_SPOT_THE_DIFF_SUBSET,
    "llava_nav_v4_150k": LLAVA_NAV_V4_150K,
    "llava_nav_v8": LLAVA_NAV_V8,
    "finetuning_data_gen": FINETUNING_DATA_GEN,
    "finetuning_data_gen_rxr_compressed": FINETUNING_DATA_GEN_RXR_COMPRESSED,
    "mp3d_anno_rxr": MP3D_ANNO_RXR,
    "mp3d_anno_rvr": MP3D_ANNO_RVR,
    "mp3d_anno_soon": MP3D_ANNO_SOON,
    "instruction_gen_r2r": INSTRUCTION_GEN_R2R,
    "instruction_gen_rvr": INSTRUCTION_GEN_RVR,
    "instruction_gen_rxr": INSTRUCTION_GEN_RXR,
    "instruction_gen_soon": INSTRUCTION_GEN_SOON,
    "tagavlm_dagger_r2r_20260412_193919": TAGAVLM_DAGGER_R2R_20260412_193919,
    "vlnce_oracle": VLNCE_ORACLE,
    "video_0_30_s_activitynetqa_oe": VIDEO_0_30_S_ACTIVITYNETQA_OE,
    "video_0_30_s_nextqa_oe": VIDEO_0_30_S_NEXTQA_OE,
    "video_0_30_s_nextqa_mc": VIDEO_0_30_S_NEXTQA_MC,
    "video_1_2_m_activitynetqa_oe": VIDEO_1_2_M_ACTIVITYNETQA_OE,
    "video_1_2_m_nextqa_oe": VIDEO_1_2_M_NEXTQA_OE,
    "video_1_2_m_nextqa_mc": VIDEO_1_2_M_NEXTQA_MC,
    "video_2_3_m_activitynetqa_oe": VIDEO_2_3_M_ACTIVITYNETQA_OE,
    "video_30_60_s_activitynetqa_oe": VIDEO_30_60_S_ACTIVITYNETQA_OE,
    "video_30_60_s_nextqa_oe": VIDEO_30_60_S_NEXTQA_OE,
    "video_30_60_s_nextqa_mc": VIDEO_30_60_S_NEXTQA_MC,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        dataset_name = dataset_name.strip()
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        dataset_key = dataset_name.lower()
        if dataset_key in data_dict.keys():
            config = data_dict[dataset_key].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        elif Path(dataset_name).exists():
            config_list.append(
                {
                    "annotation_path": dataset_name,
                    "data_path": str(Path(dataset_name).resolve().parent),
                    "sampling_rate": sampling_rate,
                }
            )
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


def _load_annotation_count(annotation_path: str) -> int:
    file_format = annotation_path.split(".")[-1]
    if file_format == "jsonl":
        with open(annotation_path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)

    with open(annotation_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    if isinstance(annotations, list):
        return len(annotations)
    if isinstance(annotations, dict):
        return len(annotations)
    raise ValueError(f"Unsupported annotation format in {annotation_path}")


def dataset_statistics(dataset_names):
    stats = []
    total_raw = 0
    total_effective = 0
    missing_datasets = []

    for raw_name, config in zip(dataset_names, data_list(dataset_names)):
        annotation_path = config["annotation_path"]
        exists = Path(annotation_path).exists()
        raw_count = None
        effective_count = None
        if exists:
            raw_count = _load_annotation_count(annotation_path)
        else:
            missing_datasets.append(
                {
                    "dataset_name": raw_name.strip(),
                    "annotation_path": annotation_path,
                }
            )

        sampling_rate = config.get("sampling_rate", 1.0)
        if raw_count is not None:
            effective_count = raw_count
        if raw_count is not None and sampling_rate < 1.0:
            effective_count = int(raw_count * sampling_rate)

        if raw_count is not None:
            total_raw += raw_count
        if effective_count is not None:
            total_effective += effective_count
        stats.append(
            {
                "dataset_name": raw_name.strip(),
                "resolved_name": re.sub(r"%(\d+)$", "", raw_name.strip()).lower(),
                "annotation_path": annotation_path,
                "data_path": config["data_path"],
                "sampling_rate": sampling_rate,
                "raw_count": raw_count,
                "effective_count": effective_count,
                "nav_graph": bool(config.get("nav_graph", False)),
<<<<<<< HEAD
=======
                "use_pano_shift": bool(config.get("use_pano_shift", False)),
                "pano_shift_path": config.get("pano_shift_path"),
>>>>>>> feature/pano-correction
                "annotation_exists": exists,
            }
        )

    return {
        "datasets": stats,
        "total_raw_count": total_raw,
        "total_effective_count": total_effective,
        "missing_datasets": missing_datasets,
    }


if __name__ == "__main__":
    dataset_names = ["cambrian_737k"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
