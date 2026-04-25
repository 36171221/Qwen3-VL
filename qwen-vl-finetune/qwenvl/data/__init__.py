import re
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

LLAVA_NAV_V4 = {
    "annotation_path": "/home/ljx/LLaVA-NeXT-graph/data/v4/llava_nav_instruct_train_large_v4.json",
    "data_path": "/home/ljx/unzip_202510181824_mp3d_data/mp3d_data",
    "nav_graph": True,
    "nav_pair_dist_path": "/home/ljx/LLaVA-NeXT-graph/data/v4/gmap_pair_dist_train_v4.npz",
    "cand_viewids_path": "/home/ljx/LLaVA-NeXT-graph/data/v4/cand_viewids_list.json",
    "nav_view_hdf5_path": "/home/ljx/LLaVA-NeXT-graph/data/view_images_bgr_from_mattersim.h5",
}

TAGAVLM_DAGGER_R2R_20260412_193919 = {
    "annotation_path": "/root/autodl-tmp/finetuning_data_gen_r2r/20260412_193919/finetuning_lables.json",
    "data_path": "/root/autodl-tmp",
    "nav_graph": True,
    "nav_pair_dist_path": "/root/autodl-tmp/finetuning_data_gen_r2r/20260412_193919/gmap_pair_dist.npz",
}

VLNCE_ORACLE = {
    "annotation_path": "/root/autodl-fs/finetuning_lables_co.json",
    "data_path": "/root/autodl-tmp",
    "nav_graph": True,
    "nav_pair_dist_path": "/root/autodl-tmp/finetuning_data_gen_r2r/gmap_pair_dist.npz",
}

data_dict = {
    "cambrian_737k": CAMBRIAN_737K,
    "cambrian_737k_pack": CAMBRIAN_737K_PACK,
    "mp_doc": MP_DOC,
    "clevr_mc": CLEVR_MC,
    "videochatgpt": VIDEOCHATGPT,
    "llava_nav_v4": LLAVA_NAV_V4,
    "tagavlm_dagger_r2r_20260412_193919": TAGAVLM_DAGGER_R2R_20260412_193919,
    'vlnce_oracle': VLNCE_ORACLE,
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


if __name__ == "__main__":
    dataset_names = ["cambrian_737k"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
