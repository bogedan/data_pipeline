#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
config_path="${1:-${script_dir}/config.yaml}"
python_bin="${DATA_PIPELINE_PYTHON:-python}"

if [[ ! -f "${config_path}" ]]; then
  echo "Config file not found: ${config_path}" >&2
  echo "Usage: $0 [config.yaml]" >&2
  exit 2
fi
config_path="$(realpath "${config_path}")"

if ! command -v "${python_bin}" >/dev/null 2>&1; then
  echo "Python interpreter not found: ${python_bin}" >&2
  echo "Activate the project environment or set DATA_PIPELINE_PYTHON=/path/to/python." >&2
  exit 2
fi

echo "Python: $(command -v "${python_bin}")"
cd "${script_dir}"

stages=(
  00_merge_datasets.py
  01_add_gripper_width.py
  02_inspect_source.py
  03_convert_trajectories.py
  04_smooth_trajectories.py
  05_limit_tcp_steps.py
  06_trim_static_segments.py
  07_sparsify_dense_turns.py
  08_convert_videos.py
  09_finalize_metadata.py
  10_visualize_tcp_distribution.py
)

for stage in "${stages[@]}"; do
  echo
  echo "==> Running ${stage}"
  "${python_bin}" -u "${stage}" --config "${config_path}"
done

echo
echo "Data pipeline completed successfully."
