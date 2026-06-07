import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np
import torch
from PIL import Image


try:
    import h5py
except ImportError:  # pragma: no cover - optional dependency
    h5py = None


def _to_rgb_image(image: Any) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        return Image.fromarray(image).convert("RGB")
    if torch.is_tensor(image):
        array = image.detach().cpu().numpy()
        return Image.fromarray(array).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(image)!r}")


@lru_cache(maxsize=8)
def load_json_cached(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=8)
def load_npz_cached(path: str):
    return np.load(path, allow_pickle=True)


@lru_cache(maxsize=8)
def load_pano_shift_index(path: str) -> Dict[str, Any]:
    """Load R2R cropped-pano shift labels for fast image-path lookup."""
    label_path = Path(path)
    if not label_path.exists():
        raise FileNotFoundError(f"Panorama shift label file not found: {path}")

    by_path: dict[str, dict[str, int]] = {}
    by_viewpoint: dict[str, dict[str, int]] = {}

    def add_step(path_id: Any, step: dict[str, Any]) -> None:
        path_key = str(path_id)
        rel_path = step.get("image_rel_path")
        viewpoint = step.get("viewpoint")
        roll_px = int(step["roll_px_applied"])
        step_idx = int(step["step"])
        if rel_path:
            by_path.setdefault(path_key, {})[str(rel_path)] = roll_px
            by_path[path_key][Path(str(rel_path)).as_posix()] = roll_px
        if viewpoint:
            by_viewpoint.setdefault(path_key, {})[str(viewpoint)] = roll_px
        by_viewpoint.setdefault(path_key, {})[str(step_idx)] = roll_px

    if label_path.suffix == ".jsonl":
        with label_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                add_step(record["path_id"], record)
    else:
        payload = load_json_cached(str(label_path))
        for item in payload.get("items", []):
            for step in item.get("steps", []):
                add_step(item["path_id"], step)

    return {"by_path": by_path, "by_viewpoint": by_viewpoint}


@lru_cache(maxsize=8)
def load_hdf5_cached(path: str):
    if h5py is None:
        raise ImportError("h5py is required to load candidate-view hdf5 files.")
    return h5py.File(path, "r")


def _parse_sample_path_id(sample_id: Optional[str]) -> Optional[str]:
    if sample_id is None:
        return None
    sample_id = str(sample_id)
    match = re.match(r"^([^_]+)_", sample_id)
    if match:
        return match.group(1)
    return sample_id or None


def _extract_mp3d_cropped_rel_path(image_spec: str, candidate: Path) -> Optional[str]:
    marker = "mp3d_data_cropped/"
    for value in (image_spec, candidate.as_posix()):
        if marker in value:
            return value.split(marker, 1)[1]
    return None


def _extract_viewpoint_from_cropped_rel_path(rel_path: str) -> Optional[str]:
    filename = Path(rel_path).name
    suffix = "_skybox_small_cropped.jpg"
    if not filename.endswith(suffix):
        return None
    return filename[: -len(suffix)]


def _lookup_pano_roll_px(
    image_spec: str,
    candidate: Path,
    sample_id: Optional[str],
    pano_shift_path: Optional[str],
) -> Optional[int]:
    if not pano_shift_path:
        return None
    rel_path = _extract_mp3d_cropped_rel_path(image_spec, candidate)
    if not rel_path:
        return None
    path_id = _parse_sample_path_id(sample_id)
    if path_id is None:
        return None

    index = load_pano_shift_index(pano_shift_path)
    rel_path = Path(rel_path).as_posix()
    path_rolls = index["by_path"].get(path_id, {})
    if rel_path in path_rolls:
        return int(path_rolls[rel_path])

    viewpoint = _extract_viewpoint_from_cropped_rel_path(rel_path)
    if viewpoint is not None:
        vp_rolls = index["by_viewpoint"].get(path_id, {})
        if viewpoint in vp_rolls:
            return int(vp_rolls[viewpoint])
    return None


def _roll_image(image: Image.Image, roll_px: int) -> Image.Image:
    if roll_px == 0:
        return image
    array = np.asarray(image)
    return Image.fromarray(np.roll(array, int(roll_px), axis=1)).convert("RGB")


def _open_direct_image(
    image_spec: Any,
    base_path: Path,
    sample_id: Optional[str] = None,
    pano_shift_path: Optional[str] = None,
) -> Optional[Image.Image]:
    if isinstance(image_spec, Image.Image):
        return image_spec.convert("RGB")
    if isinstance(image_spec, np.ndarray) or torch.is_tensor(image_spec):
        return _to_rgb_image(image_spec)
    if not isinstance(image_spec, str):
        return None

    candidates = []
    if image_spec.startswith("file://"):
        candidates.append(Path(image_spec[7:]))
    else:
        candidate = Path(image_spec)
        if candidate.is_absolute():
            candidates.append(candidate)
        else:
            candidates.append(base_path / image_spec)
            candidates.append(candidate)

    for candidate in candidates:
        if candidate.exists():
            image = Image.open(candidate).convert("RGB")
            roll_px = _lookup_pano_roll_px(
                image_spec,
                candidate,
                sample_id=sample_id,
                pano_shift_path=pano_shift_path,
            )
            if roll_px is not None:
                image = _roll_image(image, roll_px)
            return image
    return None


def _normalize_root_list(
    root_or_roots: Optional[str | Sequence[str]],
) -> list[Path]:
    if root_or_roots is None:
        return []
    if isinstance(root_or_roots, (str, Path)):
        return [Path(root_or_roots)]
    return [Path(root) for root in root_or_roots]


def resolve_nav_image(
    image_spec: Any,
    base_path: Path,
    sample_id: Optional[str] = None,
    cand_viewids_path: Optional[str] = None,
    nav_view_hdf5_path: Optional[str] = None,
    nav_view_root: Optional[str | Sequence[str]] = None,
    pano_shift_path: Optional[str] = None,
) -> Image.Image:
    direct_image = _open_direct_image(
        image_spec,
        base_path,
        sample_id=sample_id,
        pano_shift_path=pano_shift_path,
    )
    if direct_image is not None:
        return direct_image

    if not isinstance(image_spec, str):
        raise ValueError(f"Unsupported navigation image spec: {type(image_spec)!r}")

    if cand_viewids_path is None or sample_id is None:
        raise ValueError(
            f"Could not resolve nav image spec {image_spec!r} without cand_viewids_path and sample_id."
        )

    cand_viewids = load_json_cached(cand_viewids_path)
    sample_entry = cand_viewids.get(str(sample_id))
    if sample_entry is None:
        raise KeyError(f"Sample id {sample_id!r} not found in {cand_viewids_path}")

    if "_" not in image_spec:
        raise ValueError(f"Candidate view spec must contain scan and view id: {image_spec!r}")

    scan, cand_view_id = image_spec.split("_", 1)
    cand_sources = sample_entry["cand_viewids"].get(cand_view_id)
    if not cand_sources:
        raise KeyError(f"Candidate view id {cand_view_id!r} not found for sample {sample_id!r}")

    frames = []
    nav_view_roots = _normalize_root_list(nav_view_root)
    if nav_view_hdf5_path and Path(nav_view_hdf5_path).is_dir():
        nav_view_roots = nav_view_roots or [Path(nav_view_hdf5_path)]
        nav_view_hdf5_path = None
    hdf5_file = load_hdf5_cached(nav_view_hdf5_path) if nav_view_hdf5_path else None
    for source in cand_sources:
        [(viewpoint_id, view_idx)] = source.items()
        if hdf5_file is not None:
            frame = hdf5_file[f"{scan}_{viewpoint_id}"][view_idx]
            frames.append(np.asarray(frame))
            continue

        if not nav_view_roots:
            raise ValueError(
                "nav_view_root is required when nav_view_hdf5_path is not provided."
            )

        frame_path = None
        for root in nav_view_roots:
            candidate = root / scan / viewpoint_id / f"{view_idx}.jpg"
            if candidate.exists():
                frame_path = candidate
                break
        if frame_path is None:
            raise FileNotFoundError(
                f"Could not resolve candidate frame for scan={scan}, "
                f"viewpoint={viewpoint_id}, view_idx={view_idx} under roots "
                f"{[str(root) for root in nav_view_roots]}"
            )
        frames.append(np.asarray(Image.open(frame_path).convert("RGB")))

    if len(frames) == 1:
        return Image.fromarray(frames[0]).convert("RGB")

    stitched = np.concatenate(frames, axis=1)
    return Image.fromarray(stitched).convert("RGB")


def build_token_node_ids(
    input_ids: Sequence[int],
    image_token_id: int,
    video_token_id: int,
) -> list[int]:
    token_node_ids: list[int] = []
    node_idx = 0
    i = 0
    while i < len(input_ids):
        token_id = int(input_ids[i])
        if token_id == image_token_id:
            j = i
            while j < len(input_ids) and int(input_ids[j]) == image_token_id:
                j += 1
            token_node_ids.extend([node_idx] * (j - i))
            node_idx += 1
            i = j
            continue
        if token_id == video_token_id:
            j = i
            while j < len(input_ids) and int(input_ids[j]) == video_token_id:
                j += 1
            token_node_ids.extend([-1] * (j - i))
            i = j
            continue

        token_node_ids.append(-1)
        i += 1
    return token_node_ids


def expand_distance_matrix_with_node_ids(
    dist_matrix: Optional[Any],
    token_node_ids: Sequence[int],
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    if device is None:
        device = torch.device("cpu")
    if dtype is None:
        dtype = torch.float32

    token_node_ids_tensor = torch.as_tensor(token_node_ids, dtype=torch.long, device=device)
    total_tokens = token_node_ids_tensor.shape[0]
    expanded = torch.zeros((total_tokens, total_tokens), dtype=dtype, device=device)

    if dist_matrix is None:
        return expanded

    if not torch.is_tensor(dist_matrix):
        dist_tensor = torch.as_tensor(dist_matrix, dtype=dtype, device=device)
    else:
        dist_tensor = dist_matrix.to(device=device, dtype=dtype)

    if dist_tensor.ndim != 2 or dist_tensor.numel() == 0:
        return expanded

    node_mask = token_node_ids_tensor >= 0
    if not node_mask.any():
        return expanded

    num_nodes = int(token_node_ids_tensor[node_mask].max().item()) + 1
    if dist_tensor.shape[0] == num_nodes + 1 and dist_tensor.shape[1] == num_nodes + 1:
        dist_tensor = dist_tensor[1:, 1:]

    aligned = torch.zeros((num_nodes, num_nodes), dtype=dtype, device=device)
    copy_rows = min(num_nodes, dist_tensor.shape[0])
    copy_cols = min(num_nodes, dist_tensor.shape[1])
    aligned[:copy_rows, :copy_cols] = dist_tensor[:copy_rows, :copy_cols]

    node_i = token_node_ids_tensor.unsqueeze(1).expand(total_tokens, total_tokens)
    node_j = token_node_ids_tensor.unsqueeze(0).expand(total_tokens, total_tokens)
    valid_pairs = node_mask.unsqueeze(0) & node_mask.unsqueeze(1) & (node_i != node_j)

    if valid_pairs.any():
        expanded[valid_pairs] = aligned[node_i[valid_pairs], node_j[valid_pairs]]

    return expanded
