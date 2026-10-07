#!/usr/bin/env bash
set -Eeuo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
config_path="${1:-${script_dir}/config.yaml}"

if [[ ! -f "${config_path}" ]]; then
  echo "Config file not found: ${config_path}" >&2
  echo "Usage: $0 [config.yaml]" >&2
  exit 2
fi
config_path="$(realpath "${config_path}")"

source /share/project/liyuanyuan/anaconda3/bin/activate data_pipeline
cd "${script_dir}"

stages=(
  00_inspect_source.py
  01_convert_trajectories.py
  02_smooth_trajectories.py
  03_limit_tcp_steps.py
  04_trim_static_segments.py
  05_sparsify_dense_turns.py
  06_convert_videos.py
  07_finalize_metadata.py
  08_validate_openpi.py
  09_visualize_tcp_distribution.py
)

for stage in "${stages[@]}"; do
  echo
  echo "==> Running ${stage}"
  python -u "${stage}" --config "${config_path}"
done

echo
echo "Data pipeline completed successfully."
