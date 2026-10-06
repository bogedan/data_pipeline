#!/usr/bin/env python3
from __future__ import annotations

import argparse
import colorsys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import rerun as rr

from pipeline_common import episode_path
from pipeline_common import load_config
from pipeline_common import output_root
from pipeline_common import read_json
from pipeline_common import read_jsonl
from pipeline_common import work_root


def episode_color(episode_index: int) -> list[int]:
    """Return a deterministic, visually separated RGB color."""
    hue = (episode_index * 0.618033988749895) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.72, 1.0)
    return [round(red * 255), round(green * 255), round(blue * 255)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Write all episode TCP positions to one Rerun 3D recording.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", type=Path, default=None, help="Output .rrd path; defaults to work_dir.")
    parser.add_argument("--point-radius-m", type=float, default=0.0012)
    parser.add_argument("--with-paths", action="store_true", help="Also draw a line strip for every episode.")
    parser.add_argument("--path-radius-m", type=float, default=0.00035)
    args = parser.parse_args()
    if args.point_radius_m <= 0 or args.path_radius_m <= 0:
        raise SystemExit("Point and path radii must be positive")

    config = load_config(args.config)
    dataset = output_root(config)
    info_path = dataset / "meta/info.json"
    episodes_path = dataset / "meta/episodes.jsonl"
    if not info_path.exists() or not episodes_path.exists():
        raise SystemExit(f"Missing finalized dataset metadata under {dataset}; run 06_finalize_metadata.py first")

    info = read_json(info_path)
    episodes = read_jsonl(episodes_path)
    chunks_size = int(info.get("chunks_size", config["output"].get("chunks_size", 1000)))
    recording_path = args.output or (work_root(config) / "05_all_episode_tcp_pointcloud.rrd")
    recording_path = recording_path.expanduser().resolve()
    recording_path.parent.mkdir(parents=True, exist_ok=True)

    rr.init("openpi_all_episode_tcp_distribution", spawn=False)
    rr.save(str(recording_path))
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

    total_points = 0
    bounds_min = np.full(3, np.inf, dtype=np.float64)
    bounds_max = np.full(3, -np.inf, dtype=np.float64)
    for number, episode in enumerate(episodes, start=1):
        episode_index = int(episode["episode_index"])
        parquet_path = episode_path(dataset, info["data_path"], episode_index, chunks_size)
        if not parquet_path.exists():
            raise SystemExit(f"Missing episode parquet: {parquet_path}")
        table = pq.read_table(parquet_path, columns=["observation.state"])
        state = np.asarray(table["observation.state"].combine_chunks().to_pylist(), dtype=np.float32)
        if state.ndim != 2 or state.shape[1] < 3 or len(state) == 0:
            raise ValueError(f"Episode {episode_index} has invalid observation.state shape {state.shape}")
        xyz = state[:, :3]
        if not np.all(np.isfinite(xyz)):
            raise ValueError(f"Episode {episode_index} contains NaN or infinity in TCP positions")

        color = episode_color(episode_index)
        entity = f"world/episodes/episode_{episode_index:06d}"
        rr.log(f"{entity}/points", rr.Points3D(xyz, colors=color, radii=args.point_radius_m), static=True)
        if args.with_paths:
            rr.log(
                f"{entity}/path",
                rr.LineStrips3D([xyz], colors=color, radii=args.path_radius_m),
                static=True,
            )
        rr.log(f"{entity}/start", rr.Points3D(xyz[:1], colors=[255, 255, 255], radii=args.point_radius_m * 1.8), static=True)
        rr.log(f"{entity}/end", rr.Points3D(xyz[-1:], colors=[20, 20, 20], radii=args.point_radius_m * 1.8), static=True)

        total_points += len(xyz)
        bounds_min = np.minimum(bounds_min, np.min(xyz, axis=0))
        bounds_max = np.maximum(bounds_max, np.max(xyz, axis=0))
        print(f"[{number}/{len(episodes)}] episode {episode_index:06d}: {len(xyz)} TCP points")

    description = (
        f"Dataset: {dataset}\n\n"
        f"Episodes: {len(episodes)}\n\n"
        f"TCP points: {total_points}\n\n"
        "Each episode has a deterministic distinct color. White point=start; black point=end."
    )
    rr.log("description", rr.TextDocument(description), static=True)
    print(f"Wrote Rerun recording: {recording_path}")
    print(f"Episodes: {len(episodes)}; TCP points: {total_points}")
    print(f"XYZ bounds min: {bounds_min.tolist()}")
    print(f"XYZ bounds max: {bounds_max.tolist()}")


if __name__ == "__main__":
    main()
