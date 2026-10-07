#!/usr/bin/env python3
from __future__ import annotations

import argparse

import cv2
import numpy as np

from pipeline_common import PIPELINE_MARKER
from pipeline_common import POSE_NAMES
from pipeline_common import episode_path
from pipeline_common import load_config
from pipeline_common import merge_stats
from pipeline_common import output_root
from pipeline_common import read_json
from pipeline_common import read_jsonl
from pipeline_common import source_root
from pipeline_common import work_root
from pipeline_common import write_json
from pipeline_common import write_jsonl


def sample_video_stats(path, requested_frames: int) -> dict[str, list]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        capture.release()
        raise RuntimeError(f"Video contains no readable frames: {path}")
    sample_count = min(max(requested_frames, 1), frame_count)
    indices = set(np.linspace(0, frame_count - 1, sample_count, dtype=np.int64).tolist())
    minimum = np.full(3, np.inf, dtype=np.float64)
    maximum = np.full(3, -np.inf, dtype=np.float64)
    total = np.zeros(3, dtype=np.float64)
    total_square = np.zeros(3, dtype=np.float64)
    pixel_count = 0
    sampled = 0
    index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index in indices:
            rgb = frame[..., ::-1].astype(np.float64) / 255.0
            flat = rgb.reshape(-1, 3)
            minimum = np.minimum(minimum, np.min(flat, axis=0))
            maximum = np.maximum(maximum, np.max(flat, axis=0))
            total += np.sum(flat, axis=0)
            total_square += np.sum(flat * flat, axis=0)
            pixel_count += len(flat)
            sampled += 1
        index += 1
    capture.release()
    if sampled != sample_count:
        raise RuntimeError(f"Expected {sample_count} sampled frames from {path}, decoded {sampled}")
    mean = total / pixel_count
    std = np.sqrt(np.maximum(total_square / pixel_count - mean * mean, 0.0))
    def wrap(values: np.ndarray) -> list[list[list[float]]]:
        return [[[float(value)]] for value in values]

    return {
        "min": wrap(minimum),
        "max": wrap(maximum),
        "mean": wrap(mean),
        "std": wrap(std),
        "count": [sampled],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Write final LeRobot v2.1 metadata and statistics.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    source, output, work = source_root(config), output_root(config), work_root(config)
    episodes_path = output / "meta/episodes.jsonl"
    if not episodes_path.exists():
        raise SystemExit(f"Missing {episodes_path}; run 01_convert_trajectories.py first")
    source_info = read_json(source / "meta/info.json")
    if "robot_type" not in source_info:
        raise ValueError(f"Source dataset metadata has no robot_type: {source / 'meta/info.json'}")
    episodes = read_jsonl(episodes_path)
    if bool(config.get("turn_sparsification", {}).get("enabled", False)):
        partial_name = "01_7_partial_episode_stats.jsonl"
    elif bool(config.get("trim", {}).get("enabled", False)):
        partial_name = "01_5_partial_episode_stats.jsonl"
    elif bool(config.get("smoothing", {}).get("enabled", False)):
        partial_name = "01_6_partial_episode_stats.jsonl"
    else:
        partial_name = "01_partial_episode_stats.jsonl"
    partial_path = work / partial_name
    if not partial_path.exists():
        raise SystemExit(f"Missing {partial_path}; run the preceding trajectory stages first")
    partial = {row["episode_index"]: row for row in read_jsonl(partial_path)}
    output_key = config["video"]["output_key"]
    chunks_size = int(config["output"].get("chunks_size", 1000))
    sample_frames = int(config["metadata"].get("video_stats_frames_per_episode", 100))

    episode_stats = []
    for number, episode in enumerate(episodes, start=1):
        episode_index = int(episode["episode_index"])
        video_path = episode_path(
            output, source_info["video_path"], episode_index, chunks_size, video_key=output_key
        )
        if not video_path.exists():
            raise SystemExit(f"Missing {video_path}; 05_convert_videos.py did not complete successfully")
        stats = partial[episode_index]["stats"]
        stats[output_key] = sample_video_stats(video_path, sample_frames)
        episode_stats.append({"episode_index": episode_index, "stats": stats})
        print(f"[{number}/{len(episodes)}] metadata episode {episode_index:06d}")
    write_jsonl(output / "meta/episodes_stats.jsonl", episode_stats)

    keys = sorted({key for row in episode_stats for key in row["stats"]})
    global_stats = {
        key: merge_stats([row["stats"][key] for row in episode_stats if key in row["stats"]]) for key in keys
    }
    write_json(output / "meta/stats.json", global_stats)

    total_frames = sum(int(item["length"]) for item in episodes)
    width, height = int(config["video"]["width"]), int(config["video"]["height"])
    scalar_features = {
        key: source_info["features"][key]
        for key in ("timestamp", "frame_index", "episode_index", "index", "task_index")
        if key in source_info["features"]
    }
    features = {
        "observation.state": {"dtype": "float32", "shape": [10], "names": [POSE_NAMES]},
        "action": {"dtype": "float32", "shape": [10], "names": [POSE_NAMES]},
        output_key: {
            "dtype": "video",
            "shape": [height, width, 3],
            "names": ["height", "width", "channels"],
            "info": {
                "video.height": height,
                "video.width": width,
                "video.codec": "h264",
                "video.pix_fmt": str(config["video"]["pixel_format"]),
                "video.is_depth_map": False,
                "video.fps": int(source_info["fps"]),
                "video.channels": 3,
                "has_audio": False,
            },
        },
        **scalar_features,
    }
    tasks = read_jsonl(output / "meta/tasks.jsonl")
    info = {
        "codebase_version": "v2.1",
        "robot_type": source_info["robot_type"],
        "total_episodes": len(episodes),
        "total_frames": total_frames,
        "total_tasks": len(tasks),
        "total_videos": len(episodes),
        "total_chunks": (len(episodes) + chunks_size - 1) // chunks_size,
        "chunks_size": chunks_size,
        "fps": int(source_info["fps"]),
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": source_info["data_path"],
        "video_path": source_info["video_path"],
        "features": features,
    }
    write_json(output / "meta/info.json", info)
    marker = read_json(output / PIPELINE_MARKER)
    marker["complete"] = True
    marker["episodes"] = len(episodes)
    marker["frames"] = total_frames
    write_json(output / PIPELINE_MARKER, marker)
    print(f"Finalized LeRobot metadata in {output}")


if __name__ == "__main__":
    main()
