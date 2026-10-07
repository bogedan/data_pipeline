#!/usr/bin/env python3
from __future__ import annotations

import argparse

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pipeline_common import PIPELINE_MARKER
from pipeline_common import array_stats
from pipeline_common import episode_path
from pipeline_common import find_static_trim_bounds
from pipeline_common import load_config
from pipeline_common import output_root
from pipeline_common import read_json
from pipeline_common import read_jsonl
from pipeline_common import shifted_actions
from pipeline_common import source_root
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
            timestamps = np.asarray(column.combine_chunks().to_pylist())
            arrays.append(scalar_array(timestamps - timestamps[0], column))
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Trim long static prefixes and suffixes from converted episodes.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    trim = config.get("trim", {})
    if not bool(trim.get("enabled", False)):
        print("trim.enabled=false; leaving converted trajectories unchanged")
        return

    output, source, work = output_root(config), source_root(config), work_root(config)
    marker_path = output / PIPELINE_MARKER
    if not marker_path.exists():
        raise SystemExit(f"Missing {marker_path}; run 01_convert_trajectories.py first")
    marker = read_json(marker_path)
    if bool(config.get("smoothing", {}).get("enabled", False)) and marker.get("smoothing_status") != "complete":
        raise SystemExit("Trimming requires a completed 02_smooth_trajectories.py stage")
    if bool(config.get("step_limit", {}).get("enabled", False)) and marker.get("step_limit_status") != "complete":
        raise SystemExit("Trimming requires a completed 03_limit_tcp_steps.py stage")
    if marker.get("trim_status") in {"running", "complete"}:
        raise SystemExit(
            "This output has already entered the trim stage. Rerun 01_convert_trajectories.py "
            "with output.overwrite=true before trimming again."
        )
    source_info = read_json(source / "meta/info.json")
    episodes_path = output / "meta/episodes.jsonl"
    episodes = read_jsonl(episodes_path)
    chunks_size = int(config["output"].get("chunks_size", 1000))
    offset = int(config["trajectory"].get("action_offset_frames", 1))
    tail_policy = str(config["trajectory"].get("tail_policy", "repeat_last"))
    bounds_kwargs = {
        "position_threshold_m": float(trim["position_threshold_m"]),
        "rotation_threshold_deg": float(trim["rotation_threshold_deg"]),
        "gripper_threshold_rad": float(trim["gripper_threshold_rad"]),
        "pre_roll_frames": int(trim["pre_roll_frames"]),
        "post_roll_frames": int(trim["post_roll_frames"]),
        "min_episode_frames": int(trim.get("min_episode_frames", config["validation"]["action_horizon"])),
        "min_active_frames": int(trim.get("min_active_frames", 1)),
    }

    # Validate every episode before changing any parquet. This ensures that a bad
    # threshold or all-static episode fails without leaving a partially trimmed dataset.
    plans: list[dict] = []
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        path = episode_path(output, source_info["data_path"], episode_index, chunks_size)
        table = pq.read_table(path, columns=["observation.state"])
        state = np.asarray(table["observation.state"].combine_chunks().to_pylist(), dtype=np.float32)
        try:
            start, end, details = find_static_trim_bounds(state, **bounds_kwargs)
        except ValueError as error:
            raise ValueError(f"Episode {episode_index}: {error}") from error
        plans.append({"path": path, "start": start, "end": end, "details": details})

    marker["trim_status"] = "running"
    write_json(marker_path, marker)

    manifest: list[dict] = []
    partial_stats: list[dict] = []
    trimmed_episodes: list[dict] = []
    global_index = 0
    for number, (episode, plan) in enumerate(zip(episodes, plans, strict=True), start=1):
        episode_index = int(episode["episode_index"])
        path = plan["path"]
        table = pq.read_table(path)
        state = np.asarray(table["observation.state"].combine_chunks().to_pylist(), dtype=np.float32)
        start, end = int(plan["start"]), int(plan["end"])
        details = plan["details"]

        trimmed_state = state[start:end]
        trimmed_action = shifted_actions(trimmed_state, offset, tail_policy)
        trimmed_table = replace_columns(
            table.slice(start, end - start),
            state=trimmed_state,
            action=trimmed_action,
            global_start_index=global_index,
        )
        temporary = path.with_suffix(".trimmed.parquet.tmp")
        pq.write_table(trimmed_table, temporary, compression="zstd")
        temporary.replace(path)

        details = {"episode_index": episode_index, **details}
        manifest.append(details)
        partial_stats.append({"episode_index": episode_index, "stats": episode_stats(trimmed_table, trimmed_state, trimmed_action)})
        trimmed_episodes.append({**episode, "length": len(trimmed_table)})
        global_index += len(trimmed_table)
        print(
            f"[{number}/{len(episodes)}] trim episode {episode_index:06d}: "
            f"{details['original_length']} -> {details['trimmed_length']} "
            f"(head -{details['trimmed_head_frames']}, tail -{details['trimmed_tail_frames']})"
        )

    write_jsonl(work / "04_trim_manifest.jsonl", manifest)
    write_jsonl(work / "04_partial_episode_stats.jsonl", partial_stats)
    write_jsonl(episodes_path, trimmed_episodes)
    report = {
        "episodes": len(trimmed_episodes),
        "frames_before": sum(int(item["original_length"]) for item in manifest),
        "frames_after": sum(int(item["trimmed_length"]) for item in manifest),
        "trimmed_head_frames": sum(int(item["trimmed_head_frames"]) for item in manifest),
        "trimmed_tail_frames": sum(int(item["trimmed_tail_frames"]) for item in manifest),
        "thresholds": bounds_kwargs,
    }
    write_json(work / "04_trim_report.json", report)
    marker = read_json(marker_path)
    marker["trim_status"] = "complete"
    marker["trim_report"] = str(work / "04_trim_report.json")
    write_json(marker_path, marker)
    print(f"Trimmed {len(trimmed_episodes)} episodes: {report['frames_before']} -> {report['frames_after']} frames")


if __name__ == "__main__":
    main()
