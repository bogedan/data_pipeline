#!/usr/bin/env python3
from __future__ import annotations

import argparse
import colorsys
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import rerun as rr
import rerun.blueprint as rrb

from pipeline_common import episode_path
from pipeline_common import find_task_config
from pipeline_common import load_config
from pipeline_common import output_root
from pipeline_common import read_json
from pipeline_common import read_jsonl
from pipeline_common import resolve_arm_layouts
from pipeline_common import source_root
from pipeline_common import trajectory_arm_blocks


def episode_color(episode_index: int) -> list[int]:
    """Return a deterministic, visually separated RGB color."""
    hue = (episode_index * 0.618033988749895) % 1.0
    red, green, blue = colorsys.hsv_to_rgb(hue, 0.72, 1.0)
    return [round(red * 255), round(green * 255), round(blue * 255)]


def description_view() -> rrb.TextDocumentView:
    kwargs = {
        "origin": "/description",
        "contents": "/description",
        "name": "Description",
        "visible": False,
    }
    entity_behavior = getattr(rrb, "EntityBehavior", None)
    if entity_behavior is not None:
        kwargs["overrides"] = {"/description": entity_behavior(visible=False)}
    return rrb.TextDocumentView(**kwargs)


def write_recording(
    dataset: Path,
    recording_path: Path,
    roles: list[str],
    state_blocks: Callable[[np.ndarray], list[np.ndarray]],
    point_radius_m: float,
    path_radius_m: float,
    with_paths: bool,
    label: str,
    episode_indices: list[int] | None = None,
) -> None:
    info_path = dataset / "meta/info.json"
    episodes_path = dataset / "meta/episodes.jsonl"
    if not info_path.exists() or not episodes_path.exists():
        raise SystemExit(f"Missing dataset metadata under {dataset}")

    info = read_json(info_path)
    episodes = read_jsonl(episodes_path)
    if episode_indices is not None:
        episodes_by_index = {int(episode["episode_index"]): episode for episode in episodes}
        missing = [index for index in episode_indices if index not in episodes_by_index]
        if missing:
            raise ValueError(f"Dataset {dataset} is missing requested episodes: {missing[:10]}")
        episodes = [episodes_by_index[index] for index in episode_indices]
    if not episodes:
        raise ValueError(f"Dataset contains no episodes: {dataset}")
    chunks_size = int(info.get("chunks_size", 1000))
    recording_path = recording_path.expanduser().resolve()
    recording_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = recording_path.with_name(f".{recording_path.stem}.partial.rrd")
    temporary_path.unlink(missing_ok=True)
    blueprint = rrb.Blueprint(
        rrb.Spatial3DView(origin="/world", contents="/world/**", name="TCP trajectories"),
        description_view(),
        auto_views=False,
    )
    recording = rr.new_recording(f"openpi_{label}_tcp_distribution", make_default=False)
    total_points = 0
    bounds_min = np.full(3, np.inf, dtype=np.float64)
    bounds_max = np.full(3, -np.inf, dtype=np.float64)
    try:
        try:
            recording.save(str(temporary_path), default_blueprint=blueprint)
            recording.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
            for number, episode in enumerate(episodes, start=1):
                episode_index = int(episode["episode_index"])
                parquet_path = episode_path(dataset, info["data_path"], episode_index, chunks_size)
                if not parquet_path.exists():
                    raise SystemExit(f"Missing episode parquet: {parquet_path}")
                table = pq.read_table(parquet_path, columns=["observation.state"])
                state = np.asarray(table["observation.state"].combine_chunks().to_pylist(), dtype=np.float32)
                blocks = state_blocks(state)
                if len(blocks) != len(roles):
                    raise ValueError(f"Expected {len(roles)} arm blocks, got {len(blocks)}")

                color = episode_color(episode_index)
                entity = f"world/episodes/episode_{episode_index:06d}"
                for role, block in zip(roles, blocks, strict=True):
                    xyz = np.asarray(block[:, :3], dtype=np.float32)
                    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) == 0:
                        raise ValueError(f"Episode {episode_index} has invalid {role} TCP shape {xyz.shape}")
                    if not np.all(np.isfinite(xyz)):
                        raise ValueError(f"Episode {episode_index} contains NaN or infinity in {role} TCP positions")
                    arm_entity = f"{entity}/{role}"
                    recording.log(f"{arm_entity}/points", rr.Points3D(xyz, colors=color, radii=point_radius_m), static=True)
                    if with_paths:
                        recording.log(f"{arm_entity}/path", rr.LineStrips3D([xyz], colors=color, radii=path_radius_m), static=True)
                    recording.log(f"{arm_entity}/start", rr.Points3D(xyz[:1], colors=[255, 255, 255], radii=point_radius_m * 1.8), static=True)
                    recording.log(f"{arm_entity}/end", rr.Points3D(xyz[-1:], colors=[20, 20, 20], radii=point_radius_m * 1.8), static=True)
                    total_points += len(xyz)
                    bounds_min = np.minimum(bounds_min, np.min(xyz, axis=0))
                    bounds_max = np.maximum(bounds_max, np.max(xyz, axis=0))
                print(f"[{label} {number}/{len(episodes)}] episode {episode_index:06d}: {len(state)} frames x {len(blocks)} arms")

            description = (
                f"Dataset: {dataset}\n\nStage: {label}\n\n"
                f"Episodes: {len(episodes)}\n\nTCP points: {total_points}\n\n"
                "Each episode has a deterministic distinct color. White point=start; black point=end."
            )
            recording.log("description", rr.TextDocument(description), static=True)
            recording.flush()
        finally:
            recording.disconnect()
        temporary_path.replace(recording_path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    print(f"Wrote {label} Rerun recording: {recording_path}")
    print(f"Episodes: {len(episodes)}; TCP points: {total_points}")
    print(f"XYZ bounds min: {bounds_min.tolist()}")
    print(f"XYZ bounds max: {bounds_max.tolist()}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write original and processed TCP positions to Rerun recordings.")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Processed .rrd path; defaults to <output.root>/assets/<task_config>.rrd.",
    )
    parser.add_argument(
        "--origin-output", type=Path, default=None,
        help="Original .rrd path; defaults to <output.root>/assets/<task_config>_origin.rrd.",
    )
    parser.add_argument(
        "--point-radius-m",
        type=float,
        default=None,
        help="Override visualization.point_radius_m from the config.",
    )
    parser.add_argument(
        "--with-paths",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override visualization.with_paths from the config.",
    )
    parser.add_argument(
        "--path-radius-m",
        type=float,
        default=None,
        help="Override visualization.path_radius_m from the config.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    visualization = config.get("visualization", {})
    point_radius_m = (
        float(args.point_radius_m)
        if args.point_radius_m is not None
        else float(visualization.get("point_radius_m", 0.0012))
    )
    path_radius_m = (
        float(args.path_radius_m)
        if args.path_radius_m is not None
        else float(visualization.get("path_radius_m", 0.00035))
    )
    configured_with_paths = visualization.get("with_paths", True)
    if not isinstance(configured_with_paths, bool):
        raise ValueError("visualization.with_paths must be true or false")
    with_paths = configured_with_paths if args.with_paths is None else args.with_paths
    if point_radius_m <= 0 or path_radius_m <= 0:
        raise SystemExit("visualization point and path radii must be positive")

    source = source_root(config)
    output = output_root(config)
    output_episodes_path = output / "meta/episodes.jsonl"
    if not output_episodes_path.is_file():
        raise SystemExit(f"Missing finalized dataset metadata under {output}; run 09_finalize_metadata.py first")
    output_episode_indices = [int(episode["episode_index"]) for episode in read_jsonl(output_episodes_path)]
    source_info = read_json(source / "meta/info.json")
    layouts = resolve_arm_layouts(config, source_info)
    roles = [layout.role for layout in layouts]
    task_stem = find_task_config(config).stem
    assets = output / "assets"
    processed_path = args.output or (assets / f"{task_stem}.rrd")
    origin_path = args.origin_output or (assets / f"{task_stem}_origin.rrd")
    if processed_path.expanduser().resolve() == origin_path.expanduser().resolve():
        raise ValueError("Processed and original RRD paths must differ")

    def source_blocks(state: np.ndarray) -> list[np.ndarray]:
        return [state[:, layout.pose_indices[:3]] for layout in layouts]

    write_recording(
        source, origin_path, roles, source_blocks,
        point_radius_m, path_radius_m, with_paths, "origin", output_episode_indices,
    )
    write_recording(
        output, processed_path, roles, trajectory_arm_blocks,
        point_radius_m, path_radius_m, with_paths, "processed",
    )


if __name__ == "__main__":
    main()
