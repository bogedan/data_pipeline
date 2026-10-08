#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from pipeline_common import PIPELINE_MARKER
from pipeline_common import array_stats
from pipeline_common import assert_safe_output
from pipeline_common import episode_path
from pipeline_common import load_config
from pipeline_common import output_root
from pipeline_common import pose10_step_metrics
from pipeline_common import read_json
from pipeline_common import read_jsonl
from pipeline_common import shifted_actions
from pipeline_common import source_root
from pipeline_common import source_feature_keys
from pipeline_common import state8_to_pose10
from pipeline_common import work_root
from pipeline_common import write_json
from pipeline_common import write_jsonl


def fixed_float_list(values: np.ndarray) -> pa.FixedSizeListArray:
    values = np.asarray(values, dtype=np.float32)
    return pa.FixedSizeListArray.from_arrays(pa.array(values.reshape(-1), type=pa.float32()), values.shape[1])


def prepare_output(config: dict) -> Path:
    assert_safe_output(config)
    output = output_root(config)
    if output.exists():
        marker = output / PIPELINE_MARKER
        if not bool(config["output"].get("overwrite", False)):
            raise FileExistsError(f"Output exists; set output.overwrite=true to rebuild it: {output}")
        if not marker.exists():
            raise RuntimeError(f"Refusing to remove unmarked directory: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)
    write_json(
        output / PIPELINE_MARKER,
        {"pipeline": "openpi_umi_v21", "config": config["_config_path"], "complete": False},
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert state trajectories to OpenPI UMI pose10 state/action.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    source = source_root(config)
    output = prepare_output(config)
    work = work_root(config)
    info = read_json(source / "meta/info.json")
    episodes = read_jsonl(source / "meta/episodes.jsonl")
    state_key, _ = source_feature_keys(info)
    source_chunks_size = int(info.get("chunks_size", 1000))
    output_chunks_size = int(config["output"].get("chunks_size", 1000))
    offset = int(config["trajectory"].get("action_offset_frames", 1))
    tail_policy = str(config["trajectory"].get("tail_policy", "repeat_last"))
    limit = config.get("validation", {}).get("episode_limit")
    if limit is not None:
        episodes = episodes[: int(limit)]

    step_limit = config.get("step_limit", {})
    step_limit_enabled = bool(step_limit.get("enabled", False))
    percentile = float(step_limit.get("percentile", 95.0))
    if step_limit_enabled and not 0.0 < percentile < 100.0:
        raise ValueError("step_limit.percentile must be greater than 0 and less than 100")

    partial_stats: list[dict] = []
    converted_episodes: list[dict] = []
    position_steps: list[np.ndarray] = []
    rotation_steps: list[np.ndarray] = []
    for number, episode in enumerate(episodes, start=1):
        episode_index = int(episode["episode_index"])
        source_path = episode_path(source, info["data_path"], episode_index, source_chunks_size)
        output_path = episode_path(output, info["data_path"], episode_index, output_chunks_size)
        table = pq.read_table(source_path)
        state = np.asarray(table[state_key].combine_chunks().to_pylist(), dtype=np.float32)
        pose10 = state8_to_pose10(state, config)
        action = shifted_actions(pose10, offset, tail_policy)
        if step_limit_enabled:
            episode_position_steps, episode_rotation_steps = pose10_step_metrics(pose10)
            position_steps.append(episode_position_steps)
            rotation_steps.append(episode_rotation_steps)

        arrays: list[pa.Array] = []
        names: list[str] = []
        for name in table.column_names:
            if name == state_key:
                arrays.append(fixed_float_list(pose10))
            elif name == "action":
                arrays.append(fixed_float_list(action))
            elif name == "task_index" and config["task"].get("prompt") is not None:
                arrays.append(pa.array(np.zeros(len(table), dtype=np.int64)))
            else:
                arrays.append(table[name].combine_chunks())
            names.append(name)
        if "action" not in names:
            insert_at = names.index(state_key) + 1
            names.insert(insert_at, "action")
            arrays.insert(insert_at, fixed_float_list(action))
        converted = pa.Table.from_arrays(arrays, names=names)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_suffix(".parquet.tmp")
        pq.write_table(converted, temporary, compression="zstd")
        temporary.replace(output_path)

        stats = {"observation.state": array_stats(pose10), "action": array_stats(action)}
        for scalar in ("timestamp", "frame_index", "episode_index", "index", "task_index"):
            if scalar in converted.column_names:
                values = np.asarray(converted[scalar].combine_chunks().to_pylist())
                stats[scalar] = array_stats(values)
        partial_stats.append({"episode_index": episode_index, "stats": stats})
        prompt = config["task"].get("prompt")
        converted_episodes.append(
            {
                "episode_index": episode_index,
                "tasks": [prompt] if prompt is not None else episode.get("tasks", []),
                "length": len(table),
            }
        )
        print(f"[{number}/{len(episodes)}] trajectory episode {episode_index:06d}")

    if step_limit_enabled:
        if not position_steps or not any(len(values) for values in position_steps):
            raise ValueError("Cannot compute a step-limit percentile without adjacent trajectory frames")
        all_position_steps = np.concatenate(position_steps)
        all_rotation_steps = np.concatenate(rotation_steps)
        position_limit_m = float(np.percentile(all_position_steps, percentile))
        rotation_limit_deg = float(np.percentile(all_rotation_steps, percentile))
        if position_limit_m <= 0.0 or rotation_limit_deg <= 0.0:
            raise ValueError(
                "The configured step-limit percentile produced a non-positive position or rotation limit"
            )
        reference_path = work / "01_step_limit_reference.json"
        write_json(
            reference_path,
            {
                "percentile": percentile,
                "position_limit_m": position_limit_m,
                "rotation_limit_deg": rotation_limit_deg,
                "adjacent_pairs": int(len(all_position_steps)),
                "raw_max_position_step_m": float(np.max(all_position_steps)),
                "raw_max_rotation_step_deg": float(np.max(all_rotation_steps)),
                "source": "converted trajectories before smoothing",
            },
        )
        marker = read_json(output / PIPELINE_MARKER)
        marker["step_limit_reference"] = str(reference_path)
        write_json(output / PIPELINE_MARKER, marker)
        print(
            f"Step-limit P{percentile:g}: position={position_limit_m * 1000:.3f} mm, "
            f"rotation={rotation_limit_deg:.3f} deg"
        )

    write_jsonl(work / "01_partial_episode_stats.jsonl", partial_stats)
    write_jsonl(output / "meta/episodes.jsonl", converted_episodes)
    prompt = config["task"].get("prompt")
    if prompt is None:
        shutil.copy2(source / "meta/tasks.jsonl", output / "meta/tasks.jsonl")
    else:
        write_jsonl(output / "meta/tasks.jsonl", [{"task_index": 0, "task": prompt}])
    write_json(
        work / "01_trajectory_report.json",
        {
            "episodes": len(converted_episodes),
            "frames": sum(item["length"] for item in converted_episodes),
            "action_offset_frames": offset,
            "tail_policy": tail_policy,
            "gripper_mapping": config["trajectory"]["gripper"],
        },
    )
    print(f"Converted {len(converted_episodes)} trajectory episodes into {output}")


if __name__ == "__main__":
    main()
