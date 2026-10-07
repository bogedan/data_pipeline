#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import as_completed
from pathlib import Path
import subprocess

from pipeline_common import episode_path
from pipeline_common import load_config
from pipeline_common import output_root
from pipeline_common import read_json
from pipeline_common import read_jsonl
from pipeline_common import source_root
from pipeline_common import work_root


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
    trim_bounds: dict | None = None,
    kept_trimmed_indices: list[int] | None = None,
    fps: float | None = None,
) -> str:
    video = config["video"]
    width, height = int(video["width"]), int(video["height"])
    mode = video.get("crop_mode", "center_square")
    if mode == "center_square":
        spatial = f"crop='min(iw,ih)':'min(iw,ih)',scale={width}:{height}:flags=lanczos"
    elif mode == "stretch":
        spatial = f"scale={width}:{height}:flags=lanczos"
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


def convert_one(command: list[str], temporary: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        temporary.replace(output)
    except subprocess.CalledProcessError as error:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"ffmpeg failed for {output}:\n{error.stderr[-4000:]}") from error


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert the selected camera to OpenPI's fisheye video feature.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    source, output = source_root(config), output_root(config)
    source_info = read_json(source / "meta/info.json")
    episodes = read_jsonl(output / "meta/episodes.jsonl")
    source_chunks_size = int(source_info.get("chunks_size", 1000))
    output_chunks_size = int(config["output"].get("chunks_size", 1000))
    input_key = config["source"]["camera_key"]
    output_key = config["video"]["output_key"]
    ffmpeg = str(config["video"]["ffmpeg"])
    commands: list[tuple[list[str], Path, Path, int]] = []
    trim_by_episode: dict[int, dict] = {}
    if bool(config.get("trim", {}).get("enabled", False)):
        manifest_path = work_root(config) / "04_trim_manifest.jsonl"
        if not manifest_path.exists():
            raise SystemExit(f"Missing {manifest_path}; run 04_trim_static_segments.py first")
        trim_by_episode = {int(row["episode_index"]): row for row in read_jsonl(manifest_path)}
    sparsification_by_episode: dict[int, dict] = {}
    if bool(config.get("turn_sparsification", {}).get("enabled", False)):
        manifest_path = work_root(config) / "05_turn_sparsification_manifest.jsonl"
        if not manifest_path.exists():
            raise SystemExit(f"Missing {manifest_path}; run 05_sparsify_dense_turns.py first")
        sparsification_by_episode = {int(row["episode_index"]): row for row in read_jsonl(manifest_path)}

    for episode in episodes:
        episode_index = int(episode["episode_index"])
        trim_bounds = None
        if trim_by_episode:
            if episode_index not in trim_by_episode:
                raise RuntimeError(f"Trim manifest has no entry for episode {episode_index}")
            trim_bounds = trim_by_episode[episode_index]
        kept_trimmed_indices = None
        if sparsification_by_episode:
            if episode_index not in sparsification_by_episode:
                raise RuntimeError(f"Turn sparsification manifest has no entry for episode {episode_index}")
            kept_trimmed_indices = sparsification_by_episode[episode_index]["kept_trimmed_frame_indices"]
        source_path = episode_path(
            source, source_info["video_path"], episode_index, source_chunks_size, video_key=input_key
        )
        output_path = episode_path(
            output, source_info["video_path"], episode_index, output_chunks_size, video_key=output_key
        )
        temporary = output_path.with_suffix(".tmp.mp4")
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source_path),
            "-an",
            "-vf",
            video_filter(config, trim_bounds, kept_trimmed_indices, float(source_info["fps"])),
            "-c:v",
            str(config["video"]["codec"]),
            "-preset",
            str(config["video"]["preset"]),
            "-crf",
            str(config["video"]["crf"]),
            "-pix_fmt",
            str(config["video"]["pixel_format"]),
            # Compatible with the older FFmpeg available on the data machine.
            # This preserves input timestamps/frame count without resampling.
            "-vsync",
            "0",
            "-movflags",
            "+faststart",
            "-f",
            "mp4",
            str(temporary),
        ]
        commands.append((command, temporary, output_path, episode_index))

    workers = max(1, int(config["video"].get("workers", 1)))
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(convert_one, command, temporary, output_path): episode_index
            for command, temporary, output_path, episode_index in commands
        }
        completed = 0
        for future in as_completed(futures):
            future.result()
            completed += 1
            print(f"[{completed}/{len(commands)}] video episode {futures[future]:06d}")
    print(f"Converted {len(commands)} videos into {output}")


if __name__ == "__main__":
    main()
