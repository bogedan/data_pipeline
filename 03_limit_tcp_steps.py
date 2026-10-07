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
from pipeline_common import limit_pose10_steps
from pipeline_common import load_config
from pipeline_common import output_root
from pipeline_common import pose10_step_metrics
from pipeline_common import read_json
from pipeline_common import read_jsonl
from pipeline_common import rotation_quality
from pipeline_common import shifted_actions
from pipeline_common import source_root
from pipeline_common import work_root
from pipeline_common import write_json
from pipeline_common import write_jsonl


def fixed_float_list(values: np.ndarray) -> pa.FixedSizeListArray:
    values = np.asarray(values, dtype=np.float32)
    return pa.FixedSizeListArray.from_arrays(
        pa.array(values.reshape(-1), type=pa.float32()), values.shape[1]
    )


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


def write_comparison_rrd(
    path, before: np.ndarray, after: np.ndarray, episode_index: int
) -> None:
    rr.init("openpi_umi_step_limit_comparison", spawn=False)
    rr.save(str(path))
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log(
        "world/before/path",
        rr.LineStrips3D([before[:, :3]], colors=[150, 150, 150], radii=0.0007),
        static=True,
    )
    rr.log(
        "world/limited/path",
        rr.LineStrips3D([after[:, :3]], colors=[255, 150, 60], radii=0.0009),
        static=True,
    )
    rr.log(
        "description",
        rr.TextDocument(f"Episode {episode_index}: gray=stage input, orange=step limited"),
        static=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Limit adjacent TCP translation and rotation using a raw-trajectory percentile."
    )
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    settings = config.get("step_limit", {})
    if not bool(settings.get("enabled", False)):
        print("step_limit.enabled=false; leaving trajectories unchanged")
        return

    output, source, work = output_root(config), source_root(config), work_root(config)
    marker_path = output / PIPELINE_MARKER
    if not marker_path.exists():
        raise SystemExit(f"Missing {marker_path}; run 01_convert_trajectories.py first")
    marker = read_json(marker_path)
    if bool(config.get("smoothing", {}).get("enabled", False)) and marker.get("smoothing_status") != "complete":
        raise SystemExit("Step limiting requires a completed 02_smooth_trajectories.py stage")
    if marker.get("step_limit_status") in {"running", "complete"}:
        raise SystemExit(
            "This output has already entered step limiting. Rerun 01_convert_trajectories.py "
            "with output.overwrite=true before limiting again."
        )

    percentile = float(settings.get("percentile", 95.0))
    if not 0.0 < percentile < 100.0:
        raise ValueError("step_limit.percentile must be greater than 0 and less than 100")
    reference_path = work / "01_step_limit_reference.json"
    if not reference_path.exists():
        raise SystemExit(f"Missing {reference_path}; rerun 01_convert_trajectories.py")
    reference = read_json(reference_path)
    if not np.isclose(float(reference["percentile"]), percentile, atol=1e-12, rtol=0.0):
        raise ValueError(
            f"Configured percentile P{percentile:g} does not match the converted reference "
            f"P{float(reference['percentile']):g}; rerun 01_convert_trajectories.py"
        )
    storage_margin = float(settings.get("storage_margin", 0.999))
    if not 0.0 < storage_margin <= 1.0:
        raise ValueError("step_limit.storage_margin must be greater than 0 and at most 1")
    position_reference_m = float(reference["position_limit_m"])
    rotation_reference_deg = float(reference["rotation_limit_deg"])
    position_limit_m = position_reference_m * storage_margin
    rotation_limit_deg = rotation_reference_deg * storage_margin
    kwargs = {
        "position_limit_m": position_limit_m,
        "rotation_limit_deg": rotation_limit_deg,
        "preserve_endpoint_frames": int(settings.get("preserve_endpoint_frames", 1)),
        "max_sweeps": int(settings.get("max_sweeps", 3000)),
    }

    source_info = read_json(source / "meta/info.json")
    episodes = read_jsonl(output / "meta/episodes.jsonl")
    chunks_size = int(config["output"].get("chunks_size", 1000))
    offset = int(config["trajectory"].get("action_offset_frames", 1))
    tail_policy = str(config["trajectory"].get("tail_policy", "repeat_last"))
    plans: list[dict] = []
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        path = episode_path(output, source_info["data_path"], episode_index, chunks_size)
        table = pq.read_table(path, columns=["observation.state"])
        before = np.asarray(table["observation.state"].combine_chunks().to_pylist(), dtype=np.float32)
        before_position, before_rotation = pose10_step_metrics(before)
        try:
            limited, details = limit_pose10_steps(before, **kwargs)
        except (ValueError, RuntimeError) as error:
            raise type(error)(f"Episode {episode_index}: {error}") from error
        orthogonality, determinant = rotation_quality(limited)
        if max(orthogonality, determinant) > 1e-4:
            raise ValueError(
                f"Episode {episode_index}: invalid limited rotations "
                f"(orthogonality={orthogonality}, determinant={determinant})"
            )
        after_position, after_rotation = pose10_step_metrics(limited)
        if np.max(after_position, initial=0.0) > position_reference_m + 1e-7:
            raise RuntimeError(f"Episode {episode_index}: position limit was not satisfied after float32 storage")
        if np.max(after_rotation, initial=0.0) > rotation_reference_deg + 1e-5:
            raise RuntimeError(f"Episode {episode_index}: rotation limit was not satisfied after float32 storage")
        plans.append(
            {
                "episode_index": episode_index,
                "path": path,
                "before": before,
                "limited": limited,
                "before_position": before_position,
                "before_rotation": before_rotation,
                "after_position": after_position,
                "after_rotation": after_rotation,
                "details": details,
            }
        )

    marker["step_limit_status"] = "running"
    write_json(marker_path, marker)
    manifest: list[dict] = []
    partial_stats: list[dict] = []
    visualization_episode = int(settings.get("visualization_episode_index", 0))
    visualization_written = False
    for number, plan in enumerate(plans, start=1):
        table = pq.read_table(plan["path"])
        action = shifted_actions(plan["limited"], offset, tail_policy)
        converted = replace_pose_columns(table, plan["limited"], action)
        temporary = plan["path"].with_suffix(".step-limited.parquet.tmp")
        pq.write_table(converted, temporary, compression="zstd")
        temporary.replace(plan["path"])
        details = {
            "episode_index": plan["episode_index"],
            "frames": len(converted),
            "position_steps_over_reference_before": int(
                np.count_nonzero(plan["before_position"] > position_reference_m)
            ),
            "rotation_steps_over_reference_before": int(
                np.count_nonzero(plan["before_rotation"] > rotation_reference_deg)
            ),
            "max_position_step_before_m": float(np.max(plan["before_position"], initial=0.0)),
            "max_rotation_step_before_deg": float(np.max(plan["before_rotation"], initial=0.0)),
            **plan["details"],
        }
        manifest.append(details)
        partial_stats.append(
            {
                "episode_index": plan["episode_index"],
                "stats": episode_stats(converted, plan["limited"], action),
            }
        )
        if plan["episode_index"] == visualization_episode:
            write_comparison_rrd(
                work / "03_step_limit_comparison.rrd",
                plan["before"],
                plan["limited"],
                plan["episode_index"],
            )
            visualization_written = True
        print(
            f"[{number}/{len(plans)}] limit episode {plan['episode_index']:06d}: "
            f"position {details['max_position_step_before_m'] * 1000:.3f} -> "
            f"{details['max_position_step_m'] * 1000:.3f} mm; rotation "
            f"{details['max_rotation_step_before_deg']:.3f} -> "
            f"{details['max_rotation_step_deg']:.3f} deg"
        )

    write_jsonl(work / "03_step_limit_manifest.jsonl", manifest)
    write_jsonl(work / "03_partial_episode_stats.jsonl", partial_stats)
    report = {
        "episodes": len(manifest),
        "frames": sum(int(item["frames"]) for item in manifest),
        "percentile": percentile,
        "raw_reference": reference,
        "storage_margin": storage_margin,
        "applied_position_limit_m": position_limit_m,
        "applied_rotation_limit_deg": rotation_limit_deg,
        "position_steps_over_reference_before": sum(
            int(item["position_steps_over_reference_before"]) for item in manifest
        ),
        "rotation_steps_over_reference_before": sum(
            int(item["rotation_steps_over_reference_before"]) for item in manifest
        ),
        "max_position_step_after_m": max(
            (float(item["max_position_step_m"]) for item in manifest), default=0.0
        ),
        "max_rotation_step_after_deg": max(
            (float(item["max_rotation_step_deg"]) for item in manifest), default=0.0
        ),
        "visualization": str(work / "03_step_limit_comparison.rrd") if visualization_written else None,
    }
    report_path = work / "03_step_limit_report.json"
    write_json(report_path, report)
    marker = read_json(marker_path)
    marker["step_limit_status"] = "complete"
    marker["step_limit_report"] = str(report_path)
    write_json(marker_path, marker)
    print(
        f"Limited {report['episodes']} episodes using raw P{percentile:g}: "
        f"position <= {position_reference_m * 1000:.3f} mm, "
        f"rotation <= {rotation_reference_deg:.3f} deg"
    )


if __name__ == "__main__":
    main()
