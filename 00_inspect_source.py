#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil

import pyarrow.parquet as pq
import yaml

from pipeline_common import episode_path
from pipeline_common import gripper_calibration_path
from pipeline_common import load_config
from pipeline_common import read_json
from pipeline_common import read_jsonl
from pipeline_common import source_root
from pipeline_common import source_feature_keys
from pipeline_common import work_root
from pipeline_common import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the source LeRobot dataset before conversion.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    root = source_root(config)
    info = read_json(root / "meta/info.json")
    episodes = read_jsonl(root / "meta/episodes.jsonl")
    state_key, camera_key = source_feature_keys(info)
    errors: list[str] = []
    warnings: list[str] = []

    if info.get("codebase_version") != "v2.1":
        errors.append(f"Expected LeRobot v2.1, found {info.get('codebase_version')!r}")
    if state_key not in info.get("features", {}):
        errors.append(f"Missing source state feature {state_key!r}")
    if camera_key not in info.get("features", {}):
        errors.append(f"Missing source camera feature {camera_key!r}")
    state_shape = info.get("features", {}).get(state_key, {}).get("shape")
    state_names = info.get("features", {}).get(state_key, {}).get("names", [])
    if state_shape != [8]:
        warnings.append(f"Configured conversion was designed for state shape [8], found {state_shape}")
    if int(info.get("fps", 0)) != 30:
        warnings.append(f"OpenPI UMI configs currently assume 30 Hz; source reports {info.get('fps')}")
    pose_transform_enabled = config["trajectory"].get("pose_transform", {}).get("enabled")
    if pose_transform_enabled is None:
        errors.append(
            "Set trajectory.pose_transform.enabled explicitly to true or false after confirming whether "
            "observation.state already includes the configured transform"
        )

    gripper_config = config["trajectory"]["gripper"]
    calibration_path = gripper_calibration_path(config)
    calibration_values = None
    if not calibration_path.exists():
        errors.append(f"Missing gripper calibration file: {calibration_path}")
    else:
        with calibration_path.open(encoding="utf-8") as stream:
            calibration = yaml.safe_load(stream)
        distance_section_name = str(gripper_config.get("distance_section", "distance"))
        radians_section_name = str(gripper_config.get("radians_section", "width_to_rad"))
        distance_values = calibration.get(distance_section_name)
        radians_values = calibration.get(radians_section_name)
        distance_keys = {"observed_min_m", "observed_max_m"}
        radians_keys = {"min_rad", "max_rad"}
        if not isinstance(distance_values, dict) or not distance_keys <= set(distance_values):
            errors.append(
                f"Calibration section {distance_section_name!r} must contain "
                f"{sorted(distance_keys)}: {calibration_path}"
            )
        if not isinstance(radians_values, dict) or not radians_keys <= set(radians_values):
            errors.append(
                f"Calibration section {radians_section_name!r} must contain "
                f"{sorted(radians_keys)}: {calibration_path}"
            )
        calibration_values = {
            "input_distance": distance_values,
            "output_radians": radians_values,
        }

    chunks_size = int(info.get("chunks_size", 1000))
    parquet_columns: list[str] = []
    if episodes:
        first = int(episodes[0]["episode_index"])
        parquet_path = episode_path(root, info["data_path"], first, chunks_size)
        if not parquet_path.exists():
            errors.append(f"Missing first episode parquet: {parquet_path}")
        else:
            # schema.names returns nested leaf names (often just "element") for
            # fixed-size list columns; schema_arrow.names are the top-level keys.
            parquet_columns = pq.ParquetFile(parquet_path).schema_arrow.names
            if state_key not in parquet_columns:
                errors.append(f"Parquet does not contain {state_key!r}; columns={parquet_columns}")
        video_path = episode_path(
            root,
            info["video_path"],
            first,
            chunks_size,
            video_key=camera_key,
        )
        if not video_path.exists():
            errors.append(f"Missing first episode video: {video_path}")

    for executable_key in ("ffmpeg", "ffprobe"):
        executable = str(config["video"][executable_key])
        if shutil.which(executable) is None:
            errors.append(f"Executable not found: {executable}")

    report = {
        "source": str(root),
        "episodes": len(episodes),
        "frames": info.get("total_frames"),
        "fps": info.get("fps"),
        "state_key": state_key,
        "state_shape": state_shape,
        "state_names": state_names,
        "camera_key": camera_key,
        "camera_shape": info.get("features", {}).get(camera_key, {}).get("shape"),
        "parquet_columns": parquet_columns,
        "action_semantics": "action[t] = converted observation.state[t + action_offset_frames]",
        "pose_transform_enabled": pose_transform_enabled,
        "gripper_calibration": str(calibration_path),
        "gripper_mapping_values": calibration_values,
        "errors": errors,
        "warnings": warnings,
    }
    report_path = work_root(config) / "00_inspection.json"
    write_json(report_path, report)
    print(f"Inspection report: {report_path}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    if errors:
        raise SystemExit("Source inspection failed:\n- " + "\n- ".join(errors))
    print("Source inspection passed")


if __name__ == "__main__":
    main()
