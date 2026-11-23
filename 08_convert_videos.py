#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from pathlib import Path
import subprocess

import numpy as np
import yaml

from pipeline_common import episode_path
from pipeline_common import find_task_config
from pipeline_common import load_config
from pipeline_common import load_frame_selection_manifests
from pipeline_common import output_root
from pipeline_common import read_json
from pipeline_common import read_jsonl
from pipeline_common import selected_source_frame_indices
from pipeline_common import source_root
from pipeline_common import resolve_arm_layouts
from pipeline_common import work_root
from pipeline_common import write_jsonl
from video_statistics import RgbStatistics
from video_statistics import sample_indices


def resolve_crop_mode(config: dict, task: dict, mvs_name: str) -> str:
    mode = config["video"].get("crop_mode", "auto")
    if mode != "auto":
        return str(mode)
    try:
        camera = task["robot"]["mvs"][mvs_name]
    except (KeyError, TypeError) as error:
        raise ValueError(f"Cannot resolve acquisition crop setting for MVS {mvs_name!r}") from error
    crop_enabled = camera.get("crop", True)
    if not isinstance(crop_enabled, bool):
        raise ValueError(f"robot.mvs.{mvs_name}.crop must be a boolean")
    return "center_square" if crop_enabled else "stretch"


def frame_select_expression(indices: list[int]) -> str:
    if not indices:
        raise ValueError("Video frame selection cannot be empty")
    ranges: list[tuple[int, int]] = []
    start = previous = int(indices[0])
    for value in indices[1:]:
        value = int(value)
        if value != previous + 1:
            ranges.append((start, previous))
            start = value
        previous = value
    ranges.append((start, previous))
    terms = [f"eq(n\\,{start})" if start == end else f"between(n\\,{start}\\,{end})" for start, end in ranges]
    return "+".join(terms)


def video_filter(
    config: dict,
    crop_mode: str,
    trim_bounds: dict | None = None,
    kept_trimmed_indices: list[int] | None = None,
    fps: float | None = None,
) -> str:
    video = config["video"]
    width, height = int(video["width"]), int(video["height"])
    mode = crop_mode
    scale = (
        f"scale={width}:{height}:flags=lanczos:"
        "force_original_aspect_ratio=disable,setsar=1"
    )
    if mode == "center_square":
        spatial = f"crop='min(iw,ih)':'min(iw,ih)',{scale}"
    elif mode == "stretch":
        spatial = scale
    else:
        raise ValueError(f"Unsupported video.crop_mode: {mode!r}")
    if kept_trimmed_indices is not None:
        if fps is None or fps <= 0:
            raise ValueError("A positive fps is required for non-contiguous video frame selection")
        source_start = int(trim_bounds["start_frame"]) if trim_bounds is not None else 0
        source_indices = [source_start + int(index) for index in kept_trimmed_indices]
        selection = frame_select_expression(source_indices)
        return f"select='{selection}',setpts=N/({fps}*TB),{spatial}"
    if trim_bounds is None:
        return spatial
    start, end = int(trim_bounds["start_frame"]), int(trim_bounds["end_frame"])
    return f"trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS,{spatial}"


def stats_filter(frame_count: int, requested_frames: int) -> tuple[str, int]:
    indices = sample_indices(frame_count, requested_frames)
    return frame_select_expression(indices), len(indices)


def convert_one(command: list[str], temporary: Path, output: Path, sampled_frames: int, width: int, height: int) -> dict[str, list]:
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        raw = np.frombuffer(completed.stdout, dtype=np.uint8)
        expected_size = sampled_frames * height * width * 3
        if raw.size != expected_size:
            raise RuntimeError(f"Expected {expected_size} sampled RGB bytes for {output}, got {raw.size}")
        statistics = RgbStatistics()
        for frame in raw.reshape(sampled_frames, height, width, 3):
            statistics.update(frame)
        temporary.replace(output)
        return statistics.result()
    except subprocess.CalledProcessError as error:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed for {output}:\n{error.stderr.decode(errors='replace')[-4000:]}") from error
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert the selected camera to OpenPI's fisheye video feature.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    source, output, work = source_root(config), output_root(config), work_root(config)
    source_info = read_json(source / "meta/info.json")
    episodes = read_jsonl(output / "meta/episodes.jsonl")
    source_chunks_size = int(source_info.get("chunks_size", 1000))
    output_chunks_size = int(config["output"].get("chunks_size", 1000))
    layouts = resolve_arm_layouts(config, source_info)
    with find_task_config(config).open(encoding="utf-8") as stream:
        task = yaml.safe_load(stream)
    ffmpeg = str(config["video"]["ffmpeg"])
    width, height = int(config["video"]["width"]), int(config["video"]["height"])
    commands: list[tuple[list[str], Path, Path, int, str, str, int]] = []
    trim_by_episode, sparsification_by_episode = load_frame_selection_manifests(config)
    source_episodes = {
        int(row["episode_index"]): row
        for row in read_jsonl(source / "meta/episodes.jsonl")
    }

    for episode in episodes:
        episode_index = int(episode["episode_index"])
        source_length = int(source_episodes[episode_index]["length"])
        source_indices = selected_source_frame_indices(
            config,
            episode_index,
            source_length,
            trim_by_episode=trim_by_episode,
            sparsification_by_episode=sparsification_by_episode,
        )
        if len(source_indices) != int(episode["length"]):
            raise RuntimeError(
                f"Episode {episode_index} selects {len(source_indices)} video frames, "
                f"but final Parquet contains {episode['length']} rows"
            )
        trim_bounds = trim_by_episode.get(episode_index)
        kept_trimmed_indices = (
            sparsification_by_episode[episode_index]["kept_trimmed_frame_indices"]
            if episode_index in sparsification_by_episode else None
        )
        for layout in layouts:
            crop_mode = resolve_crop_mode(config, task, layout.mvs_name)
            source_path = episode_path(
                source, source_info["video_path"], episode_index, source_chunks_size, video_key=layout.source_video_key
            )
            output_path = episode_path(
                output, source_info["video_path"], episode_index, output_chunks_size, video_key=layout.output_video_key
            )
            temporary = output_path.with_suffix(".tmp.mp4")
            sampled_select, sampled_frames = stats_filter(int(episode["length"]), int(config["metadata"].get("video_stats_frames_per_episode", 32)))
            filter_chain = video_filter(config, crop_mode, trim_bounds, kept_trimmed_indices, float(source_info["fps"]))
            filter_complex = f"[0:v]{filter_chain},split=2[encoded][stats];[stats]select='{sampled_select}',format=rgb24[statsout]"
            command = [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source_path), "-an", "-filter_complex", filter_complex,
                "-map", "[encoded]", "-c:v", str(config["video"]["codec"]), "-preset", str(config["video"]["preset"]),
                "-crf", str(config["video"]["crf"]), "-pix_fmt", str(config["video"]["pixel_format"]),
                "-vsync", "0", "-movflags", "+faststart", "-f", "mp4", str(temporary),
                "-map", "[statsout]", "-f", "rawvideo", "-pix_fmt", "rgb24", "-",
            ]
            commands.append((command, temporary, output_path, episode_index, layout.role, layout.output_video_key, sampled_frames))

    workers = max(1, int(config["video"].get("workers", 1)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(convert_one, command, temporary, output_path, sampled_frames, width, height): (episode_index, role, key)
                   for command, temporary, output_path, episode_index, role, key, sampled_frames in commands}
        completed = 0
        stats_by_episode: dict[int, dict[str, dict[str, list]]] = {}
        for future in as_completed(futures):
            statistics = future.result()
            completed += 1
            episode_index, role, key = futures[future]
            stats_by_episode.setdefault(episode_index, {})[key] = statistics
            print(f"[{completed}/{len(commands)}] video episode {episode_index:06d} arm={role}")
    write_jsonl(work / "08_video_stats.jsonl", [
        {"episode_index": episode_index, "video_stats": stats_by_episode[episode_index]}
        for episode_index in sorted(stats_by_episode)
    ])
    print(f"Converted {len(commands)} videos into {output}")


if __name__ == "__main__":
    main()
