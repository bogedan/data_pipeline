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
from pipeline_common import read_json
from pipeline_common import read_jsonl
from pipeline_common import shifted_actions
from pipeline_common import source_root
from pipeline_common import sample_spatial_prefix
from pipeline_common import sparsify_dense_turns
from pipeline_common import work_root
from pipeline_common import write_json
from pipeline_common import write_jsonl


def fixed_float_list(values: np.ndarray) -> pa.FixedSizeListArray:
    values = np.asarray(values, dtype=np.float32)
    return pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1), type=pa.float32()), values.shape[1])


def scalar_array(values: np.ndarray, original: pa.ChunkedArray) -> pa.Array:
    return pa.array(values, type=original.type)


def replace_columns(
    table: pa.Table,
    *,
    state: np.ndarray,
    action: np.ndarray,
    fps: float,
    global_start_index: int,
) -> pa.Table:
    length = len(table)
    arrays: list[pa.Array | pa.ChunkedArray] = []
    for name in table.column_names:
        column = table[name]
        if name == "observation.state":
            arrays.append(fixed_float_list(state))
        elif name == "action":
            arrays.append(fixed_float_list(action))
        elif name == "timestamp":
            arrays.append(scalar_array(np.arange(length, dtype=np.float64) / fps, column))
        elif name == "frame_index":
            arrays.append(scalar_array(np.arange(length), column))
        elif name == "index":
            arrays.append(scalar_array(np.arange(global_start_index, global_start_index + length), column))
        else:
            arrays.append(column.combine_chunks())
    return pa.Table.from_arrays(arrays, names=table.column_names)


def episode_stats(table: pa.Table, state: np.ndarray, action: np.ndarray) -> dict:
    stats = {"observation.state": array_stats(state), "action": array_stats(action)}
    for name in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
        if name in table.column_names:
            stats[name] = array_stats(np.asarray(table[name].combine_chunks().to_pylist()))
    return stats


def write_comparison_rrd(path, original: np.ndarray, keep_indices: np.ndarray, removed_indices: list[int], episode: int) -> None:
    kept = original[keep_indices]
    rr.init("openpi_umi_dense_turn_sparsification", spawn=False)
    rr.save(str(path))
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log("world/smoothed/path", rr.LineStrips3D([original[:, :3]], colors=[70, 140, 255], radii=0.0007), static=True)
    rr.log("world/smoothed/points", rr.Points3D(original[:, :3], colors=[70, 140, 255], radii=0.0010), static=True)
    rr.log("world/kept/path", rr.LineStrips3D([kept[:, :3]], colors=[70, 255, 120], radii=0.0009), static=True)
    rr.log("world/kept/points", rr.Points3D(kept[:, :3], colors=[70, 255, 120], radii=0.0014), static=True)
    if removed_indices:
        rr.log(
            "world/removed_dense_points",
            rr.Points3D(original[removed_indices, :3], colors=[255, 210, 70], radii=0.0020),
            static=True,
        )
    rr.log(
        "description",
        rr.TextDocument(f"Episode {episode}: blue=smoothed input, green=kept, yellow=removed dense points"),
        static=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sparsify only overly dense TCP samples inside turns.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    sparsification = config.get("turn_sparsification", {})
    if not bool(sparsification.get("enabled", False)):
        print("turn_sparsification.enabled=false; leaving trajectories unchanged")
        return

    output, source, work = output_root(config), source_root(config), work_root(config)
    marker_path = output / PIPELINE_MARKER
    if not marker_path.exists():
        raise SystemExit(f"Missing {marker_path}; run the trajectory stages first")
    marker = read_json(marker_path)
    if bool(config.get("smoothing", {}).get("enabled", False)) and marker.get("smoothing_status") != "complete":
        raise SystemExit("Turn sparsification requires a completed 02_smooth_trajectories.py stage")
    if bool(config.get("trim", {}).get("enabled", False)) and marker.get("trim_status") != "complete":
        raise SystemExit("Turn sparsification requires a completed 03_trim_static_segments.py stage")
    if marker.get("turn_sparsification_status") in {"running", "complete"}:
        raise SystemExit(
            "This output has already entered turn sparsification. Rerun 01_convert_trajectories.py "
            "with output.overwrite=true before sparsifying again."
        )

    source_info = read_json(source / "meta/info.json")
    fps = float(source_info["fps"])
    episodes_path = output / "meta/episodes.jsonl"
    episodes = read_jsonl(episodes_path)
    chunks_size = int(config["output"].get("chunks_size", 1000))
    offset = int(config["trajectory"].get("action_offset_frames", 1))
    tail_policy = str(config["trajectory"].get("tail_policy", "repeat_last"))
    kwargs = {
        "turn_angle_threshold_deg": float(sparsification["turn_angle_threshold_deg"]),
        "turn_context_frames": int(sparsification["turn_context_frames"]),
        "min_turn_context_distance_m": float(sparsification["min_turn_context_distance_m"]),
        "dense_step_threshold_m": float(sparsification["dense_step_threshold_m"]),
        "max_kept_step_m": float(sparsification["max_kept_step_m"]),
        "max_position_error_m": float(sparsification["max_position_error_m"]),
        "max_rotation_error_deg": float(sparsification["max_rotation_error_deg"]),
        "gripper_change_threshold_rad": float(sparsification["gripper_change_threshold_rad"]),
        "max_consecutive_removed": int(sparsification["max_consecutive_removed"]),
    }
    prefix_kwargs = {
        "prefix_frames": int(sparsification.get("prefix_frames", 50)),
        "min_spacing_m": float(sparsification.get("prefix_min_spacing_m", 0.001)),
    }

    plans: list[dict] = []
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        path = episode_path(output, source_info["data_path"], episode_index, chunks_size)
        table = pq.read_table(path, columns=["observation.state"])
        state = np.asarray(table["observation.state"].combine_chunks().to_pylist(), dtype=np.float32)
        prefix_keep, prefix_details = sample_spatial_prefix(state, **prefix_kwargs)
        turn_keep_relative, turn_details = sparsify_dense_turns(state[prefix_keep], **kwargs)
        keep_indices = prefix_keep[turn_keep_relative]
        turn_removed_indices = [int(prefix_keep[index]) for index in turn_details["removed_indices"]]
        protected_turn_indices = [int(prefix_keep[index]) for index in turn_details["protected_turn_indices"]]
        details = {
            **prefix_details,
            **turn_details,
            "removed_indices": sorted(set(prefix_details["prefix_removed_indices"] + turn_removed_indices)),
            "turn_removed_indices": turn_removed_indices,
            "protected_turn_indices": protected_turn_indices,
        }
        plans.append({"episode_index": episode_index, "path": path, "state": state, "keep": keep_indices, "details": details})

    marker["turn_sparsification_status"] = "running"
    write_json(marker_path, marker)
    manifest: list[dict] = []
    partial_stats: list[dict] = []
    converted_episodes: list[dict] = []
    global_index = 0
    visualization_episode = int(sparsification.get("visualization_episode_index", 0))
    visualization_written = False
    for number, (episode, plan) in enumerate(zip(episodes, plans, strict=True), start=1):
        table = pq.read_table(plan["path"])
        selected = table.take(pa.array(plan["keep"], type=pa.int64()))
        state = plan["state"][plan["keep"]]
        action = shifted_actions(state, offset, tail_policy)
        converted = replace_columns(selected, state=state, action=action, fps=fps, global_start_index=global_index)
        temporary = plan["path"].with_suffix(".sparsified.parquet.tmp")
        pq.write_table(converted, temporary, compression="zstd")
        temporary.replace(plan["path"])

        original_length = len(plan["state"])
        max_step = float(np.max(np.linalg.norm(np.diff(state[:, :3], axis=0), axis=1), initial=0.0))
        details = {
            "episode_index": plan["episode_index"],
            "original_length": original_length,
            "sparsified_length": len(converted),
            "removed_frames": original_length - len(converted),
            "max_resulting_tcp_step_m": max_step,
            "kept_trimmed_frame_indices": plan["keep"].tolist(),
            **plan["details"],
        }
        manifest.append(details)
        partial_stats.append({"episode_index": plan["episode_index"], "stats": episode_stats(converted, state, action)})
        converted_episodes.append({**episode, "length": len(converted)})
        global_index += len(converted)
        if plan["episode_index"] == visualization_episode:
            write_comparison_rrd(
                work / "01_7_turn_sparsification_comparison.rrd",
                plan["state"],
                plan["keep"],
                plan["details"]["removed_indices"],
                plan["episode_index"],
            )
            visualization_written = True
        print(
            f"[{number}/{len(episodes)}] sparsify episode {plan['episode_index']:06d}: "
            f"{original_length} -> {len(converted)} (-{details['removed_frames']})"
        )

    write_jsonl(work / "01_7_turn_sparsification_manifest.jsonl", manifest)
    write_jsonl(work / "01_7_partial_episode_stats.jsonl", partial_stats)
    write_jsonl(episodes_path, converted_episodes)
    report = {
        "episodes": len(manifest),
        "frames_before": sum(int(item["original_length"]) for item in manifest),
        "frames_after": sum(int(item["sparsified_length"]) for item in manifest),
        "removed_frames": sum(int(item["removed_frames"]) for item in manifest),
        "turn_frames": sum(int(item["turn_frames"]) for item in manifest),
        "turn_regions": sum(int(item["turn_regions"]) for item in manifest),
        "prefix_removed_frames": sum(len(item["prefix_removed_indices"]) for item in manifest),
        "prefix_parameters": prefix_kwargs,
        "parameters": kwargs,
        "visualization": str(work / "01_7_turn_sparsification_comparison.rrd") if visualization_written else None,
    }
    report["removal_fraction"] = report["removed_frames"] / max(report["frames_before"], 1)
    write_json(work / "01_7_turn_sparsification_report.json", report)
    marker = read_json(marker_path)
    marker["turn_sparsification_status"] = "complete"
    marker["turn_sparsification_report"] = str(work / "01_7_turn_sparsification_report.json")
    write_json(marker_path, marker)
    print(f"Sparsified {report['frames_before']} -> {report['frames_after']} frames")


if __name__ == "__main__":
    main()
