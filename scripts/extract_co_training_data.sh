#!/usr/bin/env bash

set -euo pipefail

SRC_DIR="${SRC_DIR:-/root/autodl-fs/co-training-data}"
DEST_DIR="${DEST_DIR:-/root/autodl-tmp/co-training-data}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/extract_co_training_data.sh
  bash scripts/extract_co_training_data.sh /root/autodl-fs/co-training-data/0_30_s_activitynetqa_videos_1.tar.gz

Behavior:
  - Without arguments: extract all *_videos_*.tar.gz files under SRC_DIR.
  - With arguments: extract only the specified tar.gz files.
  - Copy matching JSON labels with the same dataset prefix into DEST_DIR.

Environment variables:
  SRC_DIR   Source directory, default: /root/autodl-fs/co-training-data
  DEST_DIR  Target directory, default: /root/autodl-tmp/co-training-data
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

mkdir -p "${DEST_DIR}"

extract_one() {
  local archive_path="$1"
  local archive_name base_prefix dataset_dir
  archive_name="$(basename "${archive_path}")"

  if [[ ! -f "${archive_path}" ]]; then
    echo "[skip] archive not found: ${archive_path}" >&2
    return 1
  fi

  if [[ ! "${archive_name}" =~ ^(.+)_videos_[0-9]+\.tar\.gz$ ]]; then
    echo "[skip] unsupported archive name: ${archive_name}" >&2
    return 1
  fi

  base_prefix="${BASH_REMATCH[1]}"
  dataset_dir="${DEST_DIR}/${base_prefix}"

  mkdir -p "${dataset_dir}"

  echo "[extract] ${archive_name} -> ${dataset_dir}"
  tar -xzf "${archive_path}" -C "${dataset_dir}"

  shopt -s nullglob
  local json_candidates=("${SRC_DIR}/${base_prefix}"*.json)
  shopt -u nullglob

  if (( ${#json_candidates[@]} > 0 )); then
    echo "[copy-json] ${base_prefix}*.json -> ${DEST_DIR}"
    cp -f "${json_candidates[@]}" "${DEST_DIR}/"
  else
    echo "[warn] no matching json found for prefix: ${base_prefix}" >&2
  fi
}

declare -a archives=()

if (( $# > 0 )); then
  archives=("$@")
else
  while IFS= read -r archive; do
    archives+=("${archive}")
  done < <(find "${SRC_DIR}" -maxdepth 1 -type f -name '*_videos_*.tar.gz' | sort)
fi

if (( ${#archives[@]} == 0 )); then
  echo "No matching archives found in ${SRC_DIR}" >&2
  exit 1
fi

for archive in "${archives[@]}"; do
  extract_one "${archive}"
done

echo "[done] extracted ${#archives[@]} archive(s) into ${DEST_DIR}"
