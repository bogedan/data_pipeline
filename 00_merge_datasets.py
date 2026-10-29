#!/usr/bin/env python3
"""Merge compatible LeRobot v2.1 datasets without transcoding videos."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

STATE_KEY = "observation.state"


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def dataset_path(root: Path, template: str, episode_index: int, chunks_size: int, **extra: str) -> Path:
    return root / template.format(
        episode_index=episode_index,
        episode_chunk=episode_index // chunks_size,
        **extra,
    )


def discover_inputs(discover: dict[str, Any]) -> list[Path]:
    directory = Path(discover["directory"]).expanduser().resolve()
    name = str(discover["dataset_name"])
    date = str(discover["date"])
    if not re.fullmatch(r"\d{8}", date):
        raise ValueError("preprocessing.merge.discover.date must contain exactly 8 digits")
    pattern = re.compile(rf"^{re.escape(name)}_{date}_(\d{{6}})$")
    if not directory.is_dir():
        raise FileNotFoundError(f"Dataset discovery directory does not exist: {directory}")
    matches = [path.resolve() for path in directory.iterdir() if path.is_dir() and pattern.fullmatch(path.name)]
    matches.sort(key=lambda path: pattern.fullmatch(path.name).group(1))
    if not matches:
        raise FileNotFoundError(f"No datasets match {name}_{date}_HHMMSS under {directory}")
    return matches


def resolve_inputs(settings: dict[str, Any]) -> list[Path]:
    has_discover = settings.get("discover") is not None
    has_inputs = settings.get("inputs") is not None
    if has_discover == has_inputs:
        raise ValueError("Configure exactly one of preprocessing.merge.discover or inputs")
    if has_discover:
        if not isinstance(settings["discover"], dict):
            raise ValueError("preprocessing.merge.discover must be a mapping")
        result = discover_inputs(settings["discover"])
    else:
        values = settings["inputs"]
        if not isinstance(values, list) or not values:
            raise ValueError("preprocessing.merge.inputs must be a non-empty list")
        result = [Path(value).expanduser().resolve() for value in values]
    if len(result) != len(set(result)):
        raise ValueError("preprocessing.merge inputs contain duplicates")
    if len(result) < 2:
        raise ValueError("Merging requires at least two input datasets")
    return result


def find_acquisition_config(root: Path) -> Path:
    candidates: list[Path] = []
    for suffix in ("*.yaml", "*.yml"):
        for path in sorted((root / "assets").glob(suffix)):
            with path.open(encoding="utf-8") as stream:
                value = yaml.safe_load(stream)
            if (
                isinstance(value, dict)
                and isinstance(value.get("robot"), dict)
                and isinstance(value.get("dataset"), dict)
                and isinstance(value.get("gripper_detect_online"), dict)
            ):
                candidates.append(path)
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one acquisition YAML under {root / 'assets'}, found {[path.name for path in candidates]}"
        )
    return candidates[0]


def asset_manifest(root: Path) -> dict[str, str]:
    assets = root / "assets"
    if not assets.is_dir():
        raise FileNotFoundError(f"Missing dataset assets: {assets}")
    return {
        path.relative_to(assets).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(assets.rglob("*"))
        if path.is_file() and path.suffix.lower() != ".rrd"
    }


def compatible_info(info: dict[str, Any]) -> dict[str, Any]:
    ignored = {"total_episodes", "total_frames", "total_videos", "total_chunks", "splits"}
    return {key: value for key, value in info.items() if key not in ignored}


def scalar_stats(values: np.ndarray) -> dict[str, list[Any]]:
    values = np.asarray(values)
    return {
        "min": [values.min().item()],
        "max": [values.max().item()],
        "mean": [float(values.mean())],
        "std": [float(values.std())],
        "count": [int(values.size)],
    }


def normalize_state(table: pa.Table, width: int) -> pa.Table:
    column = table[STATE_KEY].combine_chunks()
    if pa.types.is_fixed_size_list(column.type):
        if column.type.list_size != width:
            raise ValueError(f"Parquet state width {column.type.list_size} differs from metadata width {width}")
        if column.type.value_type == pa.float32():
            return table
    rows = column.to_pylist()
    if any(value is None or len(value) != width for value in rows):
        raise ValueError(f"Parquet contains an observation.state row whose width is not {width}")
    values = np.asarray(rows, dtype=np.float32)
    fixed = pa.FixedSizeListArray.from_arrays(pa.array(values.ravel(), type=pa.float32()), width)
    return table.set_column(table.schema.get_field_index(STATE_KEY), STATE_KEY, fixed)


def replace_scalar_column(table: pa.Table, name: str, values: np.ndarray) -> pa.Table:
    index = table.schema.get_field_index(name)
    if index < 0:
        raise ValueError(f"Parquet is missing required column {name!r}")
    return table.set_column(index, name, pa.array(values, type=table.schema.field(index).type))


def normalize_schema_metadata(table: pa.Table, info: dict[str, Any]) -> pa.Table:
    features: dict[str, Any] = {}
    for name in table.column_names:
        feature = info["features"].get(name)
        if not isinstance(feature, dict):
            continue
        if name == STATE_KEY:
            features[name] = {
                "feature": {"dtype": "float32", "_type": "Value"},
                "length": int(feature["shape"][0]),
                "_type": "Sequence",
            }
        else:
            features[name] = {"dtype": str(feature["dtype"]), "_type": "Value"}
    metadata = {
        b"huggingface": json.dumps({"info": {"features": features}}, separators=(",", ":")).encode("utf-8")
    }
    return table.replace_schema_metadata(metadata)


def merge_stats(items: list[dict[str, Any]]) -> dict[str, Any]:
    counts = np.asarray([item["count"][0] for item in items], dtype=np.float64)
    means = np.asarray([item["mean"] for item in items], dtype=np.float64)
    variances = np.square(np.asarray([item["std"] for item in items], dtype=np.float64))
    total = float(counts.sum())
    weights = counts.reshape((len(counts),) + (1,) * (means.ndim - 1))
    mean = np.sum(means * weights, axis=0) / total
    second = np.sum((variances + means * means) * weights, axis=0) / total
    return {
        "min": np.min(np.asarray([item["min"] for item in items]), axis=0).tolist(),
        "max": np.max(np.asarray([item["max"] for item in items]), axis=0).tolist(),
        "mean": mean.tolist(),
        "std": np.sqrt(np.maximum(second - mean * mean, 0)).tolist(),
        "count": [int(total)],
    }


def validate_dataset(root: Path) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if not root.is_dir():
        raise FileNotFoundError(f"Input dataset does not exist: {root}")
    info = read_json(root / "meta/info.json")
    if info.get("codebase_version") != "v2.1":
        raise ValueError(f"Only LeRobot v2.1 is supported: {root}")
    episodes = read_jsonl(root / "meta/episodes.jsonl")
    stats = read_jsonl(root / "meta/episodes_stats.jsonl")
    expected = list(range(len(episodes)))
    if [int(row["episode_index"]) for row in episodes] != expected:
        raise ValueError(f"Episode indices must be contiguous from zero: {root}")
    if int(info.get("total_episodes", -1)) != len(episodes):
        raise ValueError(f"meta/info.json episode count disagrees in {root}")
    if len(stats) != len(episodes) or [int(row["episode_index"]) for row in stats] != expected:
        raise ValueError(f"Episode statistics do not match episodes in {root}")
    chunks_size = int(info["chunks_size"])
    video_keys = sorted(
        key for key, feature in info["features"].items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    )
    frames = 0
    for episode in episodes:
        index, length = int(episode["episode_index"]), int(episode["length"])
        parquet = dataset_path(root, info["data_path"], index, chunks_size)
        if not parquet.is_file():
            raise FileNotFoundError(parquet)
        metadata = pq.read_metadata(parquet)
        if metadata.num_rows != length:
            raise ValueError(f"Parquet row count differs from episode metadata: {parquet}")
        for key in video_keys:
            video = dataset_path(root, info["video_path"], index, chunks_size, video_key=key)
            if not video.is_file():
                raise FileNotFoundError(video)
        frames += length
    if frames != int(info.get("total_frames", -1)):
        raise ValueError(f"meta/info.json frame count disagrees in {root}")
    if int(info.get("total_videos", -1)) != len(episodes) * len(video_keys):
        raise ValueError(f"meta/info.json video count disagrees in {root}")
    find_acquisition_config(root)
    return info, episodes, stats


def _publish(stage: Path, output: Path, overwrite: bool) -> None:
    if not output.exists():
        os.replace(stage, output)
        return
    if not overwrite:
        raise FileExistsError(f"Output already exists and overwrite is false: {output}")
    backup = output.with_name(f".{output.name}.backup-{os.getpid()}")
    os.replace(output, backup)
    try:
        os.replace(stage, output)
    except BaseException:
        os.replace(backup, output)
        raise
    shutil.rmtree(backup)


def merge_datasets(inputs: list[Path], output: Path, *, overwrite: bool = False) -> Path:
    inputs = [path.expanduser().resolve() for path in inputs]
    output = output.expanduser().resolve()
    if output in inputs or any(source in output.parents or output in source.parents for source in inputs):
        raise ValueError("Merge output must be separate from every input dataset")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists and overwrite is false: {output}")
    validated = [validate_dataset(path) for path in inputs]
    base_info, _, _ = validated[0]
    base_tasks = read_jsonl(inputs[0] / "meta/tasks.jsonl")
    base_assets = asset_manifest(inputs[0])
    for source, (info, _, _) in zip(inputs[1:], validated[1:]):
        if compatible_info(info) != compatible_info(base_info):
            raise ValueError(f"Dataset metadata/schema is incompatible: {source}")
        if read_jsonl(source / "meta/tasks.jsonl") != base_tasks:
            raise ValueError(f"Task metadata is incompatible: {source}")
        if asset_manifest(source) != base_assets:
            raise ValueError(f"Archived acquisition assets are incompatible: {source}")

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    merged_episodes: list[dict[str, Any]] = []
    merged_stats: list[dict[str, Any]] = []
    chunks_size = int(base_info["chunks_size"])
    state_width = int(base_info["features"][STATE_KEY]["shape"][0])
    video_keys = sorted(
        key for key, feature in base_info["features"].items()
        if isinstance(feature, dict) and feature.get("dtype") == "video"
    )
    global_index = 0
    next_episode = 0
    try:
        shutil.copytree(
            inputs[0] / "assets",
            stage / "assets",
            ignore=shutil.ignore_patterns("*.rrd"),
        )
        write_jsonl(stage / "meta/tasks.jsonl", base_tasks)
        for source, (info, episodes, stats) in zip(inputs, validated):
            source_chunks = int(info["chunks_size"])
            for episode, stats_row in zip(episodes, stats):
                old_index = int(episode["episode_index"])
                length = int(episode["length"])
                source_parquet = dataset_path(source, info["data_path"], old_index, source_chunks)
                target_parquet = dataset_path(stage, base_info["data_path"], next_episode, chunks_size)
                table = normalize_state(pq.read_table(source_parquet), state_width)
                table = replace_scalar_column(table, "episode_index", np.full(length, next_episode, dtype=np.int64))
                indices = np.arange(global_index, global_index + length, dtype=np.int64)
                table = replace_scalar_column(table, "index", indices)
                table = normalize_schema_metadata(table, base_info)
                target_parquet.parent.mkdir(parents=True, exist_ok=True)
                pq.write_table(table, target_parquet, compression="snappy")
                for key in video_keys:
                    source_video = dataset_path(source, info["video_path"], old_index, source_chunks, video_key=key)
                    target_video = dataset_path(stage, base_info["video_path"], next_episode, chunks_size, video_key=key)
                    target_video.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source_video, target_video)

                new_episode = copy.deepcopy(episode)
                new_episode["episode_index"] = next_episode
                merged_episodes.append(new_episode)
                new_stats = copy.deepcopy(stats_row)
                new_stats["episode_index"] = next_episode
                new_stats["stats"]["episode_index"] = scalar_stats(np.full(length, next_episode, dtype=np.int64))
                new_stats["stats"]["index"] = scalar_stats(indices)
                merged_stats.append(new_stats)
                global_index += length
                next_episode += 1
                print(f"[{next_episode}] {source.name}/episode_{old_index:06d}", flush=True)

        info = copy.deepcopy(base_info)
        info["total_episodes"] = next_episode
        info["total_frames"] = global_index
        info["total_tasks"] = len(base_tasks)
        info["total_videos"] = next_episode * len(video_keys)
        info["total_chunks"] = (next_episode + chunks_size - 1) // chunks_size
        info["splits"] = {"train": f"0:{next_episode}"}
        write_json(stage / "meta/info.json", info)
        write_jsonl(stage / "meta/episodes.jsonl", merged_episodes)
        write_jsonl(stage / "meta/episodes_stats.jsonl", merged_stats)
        stat_keys = sorted({key for row in merged_stats for key in row["stats"]})
        write_json(
            stage / "meta/stats.json",
            {
                key: merge_stats([row["stats"][key] for row in merged_stats if key in row["stats"]])
                for key in stat_keys
            },
        )
        _publish(stage, output, overwrite)
        print(f"Merged {len(inputs)} datasets into {output}: {next_episode} episodes, {global_index} frames")
        return output
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Pipeline YAML containing preprocessing.merge")
    parser.add_argument("--input", type=Path, action="append", dest="inputs")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.config:
        if args.inputs or args.output is not None or args.overwrite:
            parser.error("--config cannot be combined with --input, --output, or --overwrite")
        with args.config.expanduser().open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        if not isinstance(config, dict):
            raise ValueError("Pipeline config must contain a YAML mapping")
        preprocessing = config.get("preprocessing", {})
        if not isinstance(preprocessing, dict):
            raise ValueError("preprocessing must be a mapping")
        settings = preprocessing.get("merge", {})
        if not isinstance(settings, dict):
            raise ValueError("preprocessing.merge must be a mapping")
        enabled = settings.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError("preprocessing.merge.enabled must be true or false")
        if not enabled:
            print("Skipping dataset merge: preprocessing.merge.enabled is false")
            return
        for key in ("output", "overwrite"):
            if key not in settings:
                raise ValueError(f"preprocessing.merge.{key} is required")
        if not isinstance(settings["overwrite"], bool):
            raise ValueError("preprocessing.merge.overwrite must be true or false")
        merge_datasets(resolve_inputs(settings), Path(settings["output"]), overwrite=settings["overwrite"])
    else:
        if not args.inputs or args.output is None:
            parser.error("provide --config, or at least two --input values and --output")
        merge_datasets(args.inputs, args.output, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
