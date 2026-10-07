#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pyarrow.parquet as pq

from pipeline_common import episode_path
from pipeline_common import load_config
from pipeline_common import output_root
from pipeline_common import pose10_step_metrics
from pipeline_common import read_json
from pipeline_common import read_jsonl
from pipeline_common import rotation_quality
from pipeline_common import work_root
from pipeline_common import write_json


def probe_video(path: Path, ffprobe: str) -> dict:
    command = [
        ffprobe,
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,pix_fmt,codec_name,nb_read_frames",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(result.stdout)["streams"][0]


def validate_lerobot_loader(root: Path, fps: int, horizon: int, openpi_root: Path) -> None:
    sys.path.insert(0, str(openpi_root / "src"))
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

    print("LeRobot loader: reading dataset metadata...", flush=True)
    metadata = LeRobotDatasetMetadata(str(root))
    print("LeRobot loader: building parquet index (this may take a few minutes on shared storage)...", flush=True)
    dataset = LeRobotDataset(
        str(root),
        delta_timestamps={"action": [step / fps for step in range(horizon)]},
        tolerance_s=1e-4,
    )
    sample_indices = sorted({0, max(0, len(dataset) // 2), max(0, len(dataset) - 1)})
    for index in sample_indices:
        print(f"LeRobot loader: decoding validation sample {index}/{len(dataset) - 1}...", flush=True)
        sample = dataset[index]
        action = np.asarray(sample["action"])
        image = np.asarray(sample["observation.images.fisheye_img"])
        if action.shape != (horizon, 10):
            raise ValueError(f"LeRobot action chunk has shape {action.shape}, expected {(horizon, 10)}")
        if image.ndim != 3:
            raise ValueError(f"LeRobot image has shape {image.shape}, expected 3 dimensions")
    if metadata.fps != fps:
        raise ValueError(f"LeRobot metadata fps={metadata.fps}, expected {fps}")
    print("LeRobot loader: strict validation passed", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the converted dataset and OpenPI action chunks.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    root = output_root(config)
    info_path = root / "meta/info.json"
    if not info_path.exists():
        raise SystemExit(f"Missing {info_path}; run 07_finalize_metadata.py successfully before validation")
    info = read_json(info_path)
    episodes = read_jsonl(root / "meta/episodes.jsonl")
    chunks_size = int(info["chunks_size"])
    horizon = int(config["validation"].get("action_horizon", 32))
    errors: list[str] = []
    maxima = {
        "action_alignment": 0.0,
        "rotation_orthogonality": 0.0,
        "rotation_determinant": 0.0,
        "position_step_m": 0.0,
        "rotation_step_deg": 0.0,
    }
    step_limit_enabled = bool(config.get("step_limit", {}).get("enabled", False))
    if step_limit_enabled:
        reference_path = work_root(config) / "01_step_limit_reference.json"
        if not reference_path.exists():
            raise SystemExit(f"Missing {reference_path}; rerun the trajectory stages")
        step_reference = read_json(reference_path)
        position_step_limit_m = float(step_reference["position_limit_m"])
        rotation_step_limit_deg = float(step_reference["rotation_limit_deg"])

    required = {"observation.state", "action", "observation.images.fisheye_img"}
    missing = required - set(info["features"])
    if missing:
        errors.append(f"Missing required features: {sorted(missing)}")
    for number, episode in enumerate(episodes, start=1):
        episode_index = int(episode["episode_index"])
        parquet_path = episode_path(root, info["data_path"], episode_index, chunks_size)
        table = pq.read_table(parquet_path, columns=["observation.state", "action"])
        state = np.asarray(table["observation.state"].combine_chunks().to_pylist(), dtype=np.float32)
        action = np.asarray(table["action"].combine_chunks().to_pylist(), dtype=np.float32)
        if state.shape != action.shape or state.ndim != 2 or state.shape[1] != 10:
            errors.append(
                f"Episode {episode_index}: state/action shapes must match [N,10], got "
                f"state={state.shape}, action={action.shape}"
            )
            continue
        offset = int(config["trajectory"]["action_offset_frames"])
        if offset == 0:
            alignment_error = float(np.max(np.abs(action - state)))
            maxima["action_alignment"] = max(maxima["action_alignment"], alignment_error)
        elif len(state) > offset:
            maxima["action_alignment"] = max(
                maxima["action_alignment"], float(np.max(np.abs(action[:-offset] - state[offset:])))
            )
            maxima["action_alignment"] = max(
                maxima["action_alignment"], float(np.max(np.abs(action[-offset:] - state[-1])))
            )
        orthogonality, determinant = rotation_quality(np.concatenate([state, action], axis=0))
        maxima["rotation_orthogonality"] = max(maxima["rotation_orthogonality"], orthogonality)
        maxima["rotation_determinant"] = max(maxima["rotation_determinant"], determinant)
        position_steps, rotation_steps = pose10_step_metrics(state)
        maxima["position_step_m"] = max(
            maxima["position_step_m"], float(np.max(position_steps, initial=0.0))
        )
        maxima["rotation_step_deg"] = max(
            maxima["rotation_step_deg"], float(np.max(rotation_steps, initial=0.0))
        )
        video_path = episode_path(
            root,
            info["video_path"],
            episode_index,
            chunks_size,
            video_key=config["video"]["output_key"],
        )
        try:
            stream = probe_video(video_path, str(config["video"]["ffprobe"]))
            expected_size = (int(config["video"]["width"]), int(config["video"]["height"]))
            if (int(stream["width"]), int(stream["height"])) != expected_size:
                errors.append(f"Episode {episode_index}: video size mismatch: {stream}")
            if int(stream["nb_read_frames"]) != len(state):
                errors.append(
                    f"Episode {episode_index}: video has {stream['nb_read_frames']} frames, parquet has {len(state)}"
                )
        except Exception as error:  # keep validating other episodes
            errors.append(f"Episode {episode_index}: video probe failed: {error}")
        print(f"[{number}/{len(episodes)}] validated episode {episode_index:06d}")

    if maxima["action_alignment"] > 1e-6:
        errors.append(f"Action/state temporal alignment error: {maxima['action_alignment']}")
    if maxima["rotation_orthogonality"] > 1e-5 or maxima["rotation_determinant"] > 1e-5:
        errors.append(f"Invalid rot6d geometry: {maxima}")
    if step_limit_enabled:
        if maxima["position_step_m"] > position_step_limit_m + 1e-7:
            errors.append(
                f"Final TCP position step {maxima['position_step_m']} m exceeds "
                f"P{float(step_reference['percentile']):g} limit {position_step_limit_m} m"
            )
        if maxima["rotation_step_deg"] > rotation_step_limit_deg + 1e-5:
            errors.append(
                f"Final TCP rotation step {maxima['rotation_step_deg']} deg exceeds "
                f"P{float(step_reference['percentile']):g} limit {rotation_step_limit_deg} deg"
            )

    loader_status = "skipped"
    if bool(config["validation"].get("require_lerobot_loader", True)):
        print("Static episode checks passed; starting strict LeRobot loader validation...", flush=True)
        try:
            validate_lerobot_loader(
                root,
                int(info["fps"]),
                horizon,
                Path(config["validation"]["openpi_root"]).resolve(),
            )
            loader_status = "passed"
        except Exception as error:
            loader_status = f"failed: {error}"
            errors.append(f"LeRobot/OpenPI loader validation failed: {error}")

    report = {
        "dataset": str(root),
        "episodes": len(episodes),
        "frames": info.get("total_frames"),
        "max_errors": maxima,
        "lerobot_loader": loader_status,
        "errors": errors,
    }
    report_path = work_root(config) / "08_validation.json"
    write_json(report_path, report)
    print(f"Validation report: {report_path}")
    if errors:
        raise SystemExit("Validation failed:\n- " + "\n- ".join(errors))
    print("Dataset is compatible with the current OpenPI UMI data contract")


if __name__ == "__main__":
    main()
