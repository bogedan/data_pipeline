#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import rerun as rr

from pipeline_common import PIPELINE_MARKER
from pipeline_common import array_stats
from pipeline_common import episode_path
from pipeline_common import load_config
from pipeline_common import output_root
from pipeline_common import pose10_rotation_matrix
from pipeline_common import read_json
from pipeline_common import read_jsonl
from pipeline_common import rotation_quality
from pipeline_common import shifted_actions
from pipeline_common import smooth_pose10
from pipeline_common import source_root
from pipeline_common import work_root
from pipeline_common import write_json
from pipeline_common import write_jsonl


def fixed_float_list(values: np.ndarray) -> pa.FixedSizeListArray:
    values = np.asarray(values, dtype=np.float32)
    return pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1), type=pa.float32()), values.shape[1])


def replace_pose_columns(table: pa.Table, state: np.ndarray, action: np.ndarray) -> pa.Table:
    arrays: list[pa.Array | pa.ChunkedArray] = []
    for name in table.column_names:
        if name == "observation.state":
            arrays.append(fixed_float_list(state))
        elif name == "action":
            arrays.append(fixed_float_list(action))
        else:
            arrays.append(table[name].combine_chunks())
    return pa.Table.from_arrays(arrays, names=table.column_names)


def episode_stats(table: pa.Table, state: np.ndarray, action: np.ndarray) -> dict:
    stats = {"observation.state": array_stats(state), "action": array_stats(action)}
    for name in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        if name in table.column_names:
            stats[name] = array_stats(np.asarray(table[name].combine_chunks().to_pylist()))
    return stats


def smoothing_metrics(raw: np.ndarray, smoothed: np.ndarray) -> dict[str, float]:
    position_error = np.linalg.norm(smoothed[:, :3] - raw[:, :3], axis=1)
    raw_rotation = pose10_rotation_matrix(raw)
    smooth_rotation = pose10_rotation_matrix(smoothed)
    relative = np.einsum("nji,njk->nik", raw_rotation, smooth_rotation)
    cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    rotation_error = np.degrees(np.arccos(cosine))

    def jerk_rms(position: np.ndarray) -> float:
        if len(position) < 4:
            return 0.0
        jerk = np.diff(position, n=3, axis=0)
        return float(np.sqrt(np.mean(np.sum(jerk * jerk, axis=1))))

    return {
        "position_change_rms_m": float(np.sqrt(np.mean(position_error * position_error))),
        "position_change_max_m": float(np.max(position_error)),
        "rotation_change_rms_deg": float(np.sqrt(np.mean(rotation_error * rotation_error))),
        "rotation_change_max_deg": float(np.max(rotation_error)),
        "raw_position_jerk_rms": jerk_rms(raw[:, :3]),
        "smoothed_position_jerk_rms": jerk_rms(smoothed[:, :3]),
        "gripper_change_max_rad": float(np.max(np.abs(smoothed[:, 9] - raw[:, 9]))),
    }


def log_pose_axes(entity_path: str, poses: np.ndarray, stride: int, axis_length: float) -> None:
    sampled = poses[::stride]
    rotations = pose10_rotation_matrix(sampled)
    origins = np.repeat(sampled[:, None, :3], 3, axis=1).reshape(-1, 3)
    vectors = (rotations * axis_length).transpose(0, 2, 1).reshape(-1, 3)
    colors = np.tile(np.asarray([[255, 70, 70], [70, 255, 70], [70, 140, 255]], dtype=np.uint8), (len(sampled), 1))
    rr.log(entity_path, rr.Arrows3D(origins=origins, vectors=vectors, colors=colors, radii=0.00025), static=True)


def write_comparison_rrd(path, raw: np.ndarray, smoothed: np.ndarray, episode_index: int, config: dict) -> None:
    rr.init("openpi_umi_smoothing_comparison", spawn=False)
    rr.save(str(path))
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log("world/raw/path", rr.LineStrips3D([raw[:, :3]], colors=[150, 150, 150], radii=0.0007), static=True)
    rr.log("world/raw/points", rr.Points3D(raw[:, :3], colors=[150, 150, 150], radii=0.0012), static=True)
    rr.log("world/smoothed/path", rr.LineStrips3D([smoothed[:, :3]], colors=[70, 140, 255], radii=0.0009), static=True)
    rr.log("world/smoothed/points", rr.Points3D(smoothed[:, :3], colors=[70, 140, 255], radii=0.0015), static=True)
    stride = max(1, int(config.get("visualization_pose_stride", 10)))
    axis_length = float(config.get("visualization_axis_length_m", 0.005))
    log_pose_axes("world/raw/orientations", raw, stride, axis_length)
    log_pose_axes("world/smoothed/orientations", smoothed, stride, axis_length)
    rr.log("description", rr.TextDocument(f"Episode {episode_index}: gray=raw, blue=smoothed"), static=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Smooth converted TCP translation and SO(3) rotation trajectories.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    smoothing = config.get("smoothing", {})
    if not bool(smoothing.get("enabled", False)):
        print("smoothing.enabled=false; leaving converted trajectories unchanged")
        return

    output, source, work = output_root(config), source_root(config), work_root(config)
    marker_path = output / PIPELINE_MARKER
    if not marker_path.exists():
        raise SystemExit(f"Missing {marker_path}; run the trajectory stages first")
    marker = read_json(marker_path)
    if marker.get("smoothing_status") in {"running", "complete"}:
        raise SystemExit(
            "This output has already entered smoothing. Rerun 01_convert_trajectories.py "
            "with output.overwrite=true before smoothing again."
        )

    source_info = read_json(source / "meta/info.json")
    episodes = read_jsonl(output / "meta/episodes.jsonl")
    chunks_size = int(config["output"].get("chunks_size", 1000))
    offset = int(config["trajectory"].get("action_offset_frames", 1))
    tail_policy = str(config["trajectory"].get("tail_policy", "repeat_last"))
    kwargs = {
        "window_size": int(smoothing.get("window_size", 5)),
        "passes": int(smoothing.get("passes", 1)),
        "preserve_endpoint_frames": int(smoothing.get("preserve_endpoint_frames", 2)),
    }

    plans: list[dict] = []
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        path = episode_path(output, source_info["data_path"], episode_index, chunks_size)
        table = pq.read_table(path, columns=["observation.state"])
        raw = np.asarray(table["observation.state"].combine_chunks().to_pylist(), dtype=np.float32)
        smoothed = smooth_pose10(raw, **kwargs)
        orthogonality, determinant = rotation_quality(smoothed)
        if max(orthogonality, determinant) > 1e-4:
            raise ValueError(
                f"Episode {episode_index}: invalid smoothed rotations "
                f"(orthogonality={orthogonality}, determinant={determinant})"
            )
        plans.append({"episode_index": episode_index, "path": path, "raw": raw, "smoothed": smoothed})

    marker["smoothing_status"] = "running"
    write_json(marker_path, marker)
    manifest: list[dict] = []
    partial_stats: list[dict] = []
    visualization_episode = int(smoothing.get("visualization_episode_index", 0))
    visualization_written = False
    for number, plan in enumerate(plans, start=1):
        table = pq.read_table(plan["path"])
        action = shifted_actions(plan["smoothed"], offset, tail_policy)
        converted = replace_pose_columns(table, plan["smoothed"], action)
        temporary = plan["path"].with_suffix(".smoothed.parquet.tmp")
        pq.write_table(converted, temporary, compression="zstd")
        temporary.replace(plan["path"])
        metrics = smoothing_metrics(plan["raw"], plan["smoothed"])
        manifest.append({"episode_index": plan["episode_index"], "frames": len(converted), **metrics})
        partial_stats.append(
            {"episode_index": plan["episode_index"], "stats": episode_stats(converted, plan["smoothed"], action)}
        )
        if plan["episode_index"] == visualization_episode:
            write_comparison_rrd(
                work / "02_smoothing_comparison.rrd",
                plan["raw"],
                plan["smoothed"],
                plan["episode_index"],
                smoothing,
            )
            visualization_written = True
        print(f"[{number}/{len(plans)}] smooth episode {plan['episode_index']:06d}: {len(converted)} frames")

    write_jsonl(work / "02_smoothing_manifest.jsonl", manifest)
    write_jsonl(work / "02_partial_episode_stats.jsonl", partial_stats)
    report = {
        "episodes": len(manifest),
        "frames": sum(int(item["frames"]) for item in manifest),
        "parameters": kwargs,
        "position_change_rms_m": float(np.mean([item["position_change_rms_m"] for item in manifest])),
        "position_change_max_m": float(np.max([item["position_change_max_m"] for item in manifest])),
        "rotation_change_rms_deg": float(np.mean([item["rotation_change_rms_deg"] for item in manifest])),
        "rotation_change_max_deg": float(np.max([item["rotation_change_max_deg"] for item in manifest])),
        "raw_position_jerk_rms": float(np.mean([item["raw_position_jerk_rms"] for item in manifest])),
        "smoothed_position_jerk_rms": float(np.mean([item["smoothed_position_jerk_rms"] for item in manifest])),
        "gripper_change_max_rad": float(np.max([item["gripper_change_max_rad"] for item in manifest])),
        "visualization": str(work / "02_smoothing_comparison.rrd") if visualization_written else None,
    }
    write_json(work / "02_smoothing_report.json", report)
    marker = read_json(marker_path)
    marker["smoothing_status"] = "complete"
    marker["smoothing_report"] = str(work / "02_smoothing_report.json")
    write_json(marker_path, marker)
    print(f"Smoothed {report['episodes']} episodes / {report['frames']} frames")


if __name__ == "__main__":
    main()
