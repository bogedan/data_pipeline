#!/usr/bin/env python3
"""Detect gripper widths from archived MVS videos in a LeRobot v2.1 dataset."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

STATE_KEY = "observation.state"
ARM_ORDER = ("right", "left")
POSE_COMPONENTS = ("x", "y", "z", "qx", "qy", "qz", "qw")


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


def find_acquisition_config(root: Path) -> Path:
    candidates: list[Path] = []
    for suffix in ("*.yaml", "*.yml"):
        for path in sorted((root / "assets").glob(suffix)):
            value = load_yaml(path)
            if (
                isinstance(value.get("robot"), dict)
                and isinstance(value.get("dataset"), dict)
                and isinstance(value.get("gripper_detect_online"), dict)
            ):
                candidates.append(path)
    if len(candidates) != 1:
        raise ValueError(
            f"Expected one acquisition YAML under {root / 'assets'}, found {[path.name for path in candidates]}"
        )
    return candidates[0]


def publish_directory(stage: Path, output: Path, overwrite: bool) -> None:
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


@dataclass(frozen=True)
class ArmBinding:
    role: str
    pose_indices: tuple[int, ...]
    gripper_id: str
    mvs_name: str
    mvs_serial: str
    video_key: str


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a YAML mapping: {path}")
    return value


def resolve_bindings(config: dict[str, Any], info: dict[str, Any]) -> list[ArmBinding]:
    robot = config.get("robot")
    arms = robot.get("arms") if isinstance(robot, dict) else None
    cameras = robot.get("mvs") if isinstance(robot, dict) else None
    if not isinstance(arms, dict) or "right" not in arms or not isinstance(cameras, dict):
        raise ValueError("Acquisition YAML must define robot.arms.right and robot.mvs")
    unknown = set(arms) - set(ARM_ORDER)
    if unknown:
        raise ValueError(f"Unsupported arm roles: {sorted(unknown)}")
    names = info.get("features", {}).get(STATE_KEY, {}).get("names")
    if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
        raise ValueError("observation.state metadata must contain string names")
    result: list[ArmBinding] = []
    pose_sources: list[str] = []
    for role in ARM_ORDER:
        if role not in arms:
            continue
        arm = arms[role]
        if not isinstance(arm, dict):
            raise ValueError(f"robot.arms.{role} must be a mapping")
        pose_source = arm.get("pose_source")
        gripper_id = arm.get("gripper")
        mvs_name = arm.get("mvs")
        if not isinstance(pose_source, str) or not pose_source:
            raise ValueError(f"robot.arms.{role}.pose_source must be configured")
        if not isinstance(gripper_id, str) or not gripper_id:
            raise ValueError(f"robot.arms.{role}.gripper must be configured")
        if not isinstance(mvs_name, str) or not mvs_name:
            raise ValueError(f"robot.arms.{role}.mvs must be configured")
        if mvs_name not in cameras or not isinstance(cameras[mvs_name], dict):
            raise ValueError(f"robot.arms.{role}.mvs does not reference robot.mvs")
        serial = cameras[mvs_name].get("serial")
        if not isinstance(serial, str) or not serial:
            raise ValueError(f"robot.mvs.{mvs_name}.serial must be configured")
        indices: list[int] = []
        for component in POSE_COMPONENTS:
            matches = [
                index for index, name in enumerate(names)
                if name.startswith(f"{pose_source}_") and name.endswith(f".{component}")
            ]
            if len(matches) != 1:
                raise ValueError(f"Expected one {pose_source}.{component} state field, found {len(matches)}")
            indices.append(matches[0])
        result.append(ArmBinding(role, tuple(indices), gripper_id, mvs_name, serial, f"observation.images.mvs_{serial}"))
        pose_sources.append(pose_source)
    if len(pose_sources) != len(set(pose_sources)):
        raise ValueError("Multiple arms cannot share one pose_source")
    for attribute in ("gripper_id", "mvs_name"):
        values = [getattr(item, attribute) for item in result]
        if len(values) != len(set(values)):
            raise ValueError(f"Multiple arms cannot share one {attribute}")
    return result


def marker_translation(corners: Any, camera: np.ndarray, distortion: np.ndarray, object_points: np.ndarray) -> np.ndarray | None:
    pixels = np.asarray(corners, dtype=np.float64).reshape(4, 1, 2)
    outline = pixels.reshape(4, 2)
    sides = np.linalg.norm(outline - np.roll(outline, -1, axis=0), axis=1)
    if not np.isfinite(pixels).all() or sides.min() < 8:
        return None
    undistorted = cv2.fisheye.undistortPoints(pixels, camera, distortion, P=camera)
    count, rotations, translations, _ = cv2.solvePnPGeneric(
        object_points, undistorted, camera, None, flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    best: tuple[np.ndarray, float] | None = None
    for rotation, translation in zip(rotations[:count], translations[:count]):
        translation = np.asarray(translation, dtype=np.float64).reshape(3, 1)
        if not np.isfinite(translation).all() or translation[2, 0] <= 0:
            continue
        projected, _ = cv2.fisheye.projectPoints(
            object_points.reshape(-1, 1, 3), np.asarray(rotation).reshape(3, 1), translation, camera, distortion
        )
        error = float(np.sqrt(np.mean(np.sum((projected.reshape(4, 2) - outline) ** 2, axis=1))))
        if np.isfinite(error) and (best is None or error < best[1]):
            best = translation.reshape(3), error
    return best[0] if best is not None and best[1] <= 2 else None


class WidthDetector:
    def __init__(self, camera: np.ndarray, distortion: np.ndarray, markers: tuple[str, int, int, float]):
        dictionary, self.left_id, self.right_id, side = markers
        if not hasattr(cv2.aruco, dictionary):
            raise ValueError(f"Unknown ArUco dictionary: {dictionary}")
        self.detector = cv2.aruco.ArucoDetector(
            cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary)),
            cv2.aruco.DetectorParameters(),
        )
        half = side / 2
        self.object_points = np.asarray(
            [[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], dtype=np.float64
        )
        self.camera = camera
        self.distortion = distortion

    def measure(self, rgb: np.ndarray) -> float | None:
        corners, ids, _ = self.detector.detectMarkers(cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY))
        if ids is None:
            return None
        found = ids.reshape(-1).tolist()
        if found.count(self.left_id) != 1 or found.count(self.right_id) != 1:
            return None
        positions = [
            marker_translation(corners[found.index(marker)], self.camera, self.distortion, self.object_points)
            for marker in (self.left_id, self.right_id)
        ]
        if any(value is None for value in positions):
            return None
        return float(np.linalg.norm(positions[0] - positions[1]))


def validate_roi(value: Any, frame_size: tuple[int, int]) -> tuple[int, int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4 or any(type(item) is not int for item in value):
        raise ValueError("roi_xywh must contain four integers")
    x, y, width, height = value
    frame_width, frame_height = frame_size
    if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > frame_width or y + height > frame_height:
        raise ValueError(f"ROI {value} lies outside {frame_width}x{frame_height}")
    return x, y, width, height


def load_detector(root: Path, config: dict[str, Any], info: dict[str, Any], binding: ArmBinding) -> tuple[tuple[int, int, int, int], WidthDetector]:
    online = config.get("gripper_detect_online")
    if not isinstance(online, dict) or not isinstance(online.get(binding.gripper_id), dict):
        raise ValueError(f"Missing gripper_detect_online.{binding.gripper_id}")
    feature = info["features"].get(binding.video_key)
    if not isinstance(feature, dict) or feature.get("dtype") != "video":
        raise ValueError(f"Missing video feature {binding.video_key}")
    frame_height, frame_width = map(int, feature["shape"][:2])
    roi = validate_roi(online[binding.gripper_id].get("roi_xywh"), (frame_width, frame_height))

    camera_path = root / "assets/calibration" / f"mvs_{binding.mvs_serial}.yaml"
    gripper_path = root / "assets/calibration" / f"{binding.gripper_id}.yaml"
    camera_yaml, gripper_yaml = load_yaml(camera_path), load_yaml(gripper_path)
    if camera_yaml.get("device", {}).get("type") != "mvs" or str(camera_yaml.get("device", {}).get("serial")) != binding.mvs_serial:
        raise ValueError(f"Camera calibration identity mismatch: {camera_path}")
    if gripper_yaml.get("gripper_id") != binding.gripper_id:
        raise ValueError(f"Gripper calibration identity mismatch: {gripper_path}")
    if gripper_yaml.get("measurement") != "marker_center_distance" or gripper_yaml.get("units") != "m":
        raise ValueError(f"Gripper calibration must measure marker_center_distance in meters: {gripper_path}")
    if str(gripper_yaml.get("source", {}).get("camera_serial")) != binding.mvs_serial:
        raise ValueError(f"Gripper and MVS calibration mismatch: {gripper_path}")
    if camera_yaml.get("camera_model", {}).get("type") != "opencv_fisheye":
        raise ValueError(f"Camera calibration is not opencv_fisheye: {camera_path}")
    configured_rotation = int(config["robot"]["mvs"][binding.mvs_name].get("rotation", camera_yaml["image"]["rotation"]))
    if configured_rotation != int(camera_yaml["image"]["rotation"]):
        raise ValueError(f"Acquisition camera rotation differs from calibration: {camera_path}")
    camera = np.asarray(camera_yaml["camera_matrix"], dtype=np.float64)
    distortion = np.asarray(camera_yaml["dist_coeffs"], dtype=np.float64).reshape(-1, 1)
    if camera.shape != (3, 3) or distortion.shape != (4, 1) or not np.isfinite(camera).all() or not np.isfinite(distortion).all():
        raise ValueError(f"Invalid camera matrix or distortion: {camera_path}")
    crop = camera_yaml.get("fisheye_crop")
    if crop is None:
        crop_x, crop_y = 0, 0
    else:
        if (int(crop["output_width"]), int(crop["output_height"])) != (frame_width, frame_height):
            raise ValueError(f"Recorded frame size differs from camera crop: {camera_path}")
        crop_x = round(float(crop["center_x"])) - int(crop["output_width"]) // 2
        crop_y = round(float(crop["center_y"])) - int(crop["output_height"]) // 2
    local_camera = camera.copy()
    local_camera[0, 2] -= crop_x + roi[0]
    local_camera[1, 2] -= crop_y + roi[1]
    markers = gripper_yaml.get("markers", {})
    marker_config = (
        str(markers["dictionary"]), int(markers["left_id"]), int(markers["right_id"]),
        float(markers["black_square_side_m"]),
    )
    if marker_config[1] == marker_config[2] or not np.isfinite(marker_config[3]) or marker_config[3] <= 0:
        raise ValueError(f"Invalid marker configuration: {gripper_path}")
    return roi, WidthDetector(local_camera, distortion, marker_config)


def measure_video(path: Path, length: int, frame_size: tuple[int, int], roi: tuple[int, int, int, int], detector: WidthDetector) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot decode video: {path}")
    values = np.full(length, np.nan, dtype=np.float32)
    count = 0
    x, y, width, height = roi
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if count >= length:
                raise ValueError(f"Video has more frames than Parquet: {path}")
            if (frame.shape[1], frame.shape[0]) != frame_size:
                raise ValueError(f"Video frame size differs from metadata: {path}")
            value = detector.measure(cv2.cvtColor(frame[y:y + height, x:x + width], cv2.COLOR_BGR2RGB))
            if value is not None:
                values[count] = value
            count += 1
    finally:
        capture.release()
    if count != length:
        raise ValueError(f"Video has {count} frames but Parquet has {length}: {path}")
    return values


def interpolate(values: np.ndarray, episode_index: int, gripper_id: str) -> tuple[np.ndarray, int]:
    valid = np.flatnonzero(np.isfinite(values))
    if valid.size == 0:
        raise ValueError(f"Episode {episode_index} has no valid {gripper_id} measurements")
    missing = len(values) - len(valid)
    if missing:
        values = np.interp(np.arange(len(values)), valid, values[valid]).astype(np.float32)
    return values, missing


def state_array(table: pa.Table, width: int) -> np.ndarray:
    rows = table[STATE_KEY].combine_chunks().to_pylist()
    if any(value is None or len(value) != width for value in rows):
        raise ValueError(f"Parquet observation.state width differs from metadata width {width}")
    return np.asarray(rows, dtype=np.float32)


def replace_state(table: pa.Table, values: np.ndarray) -> pa.Table:
    column = pa.FixedSizeListArray.from_arrays(pa.array(values.ravel(), type=pa.float32()), values.shape[1])
    table = table.set_column(table.schema.get_field_index(STATE_KEY), STATE_KEY, column)
    metadata = dict(table.schema.metadata or {})
    if b"huggingface" in metadata:
        value = json.loads(metadata[b"huggingface"].decode("utf-8"))
        value["info"]["features"][STATE_KEY]["length"] = values.shape[1]
        metadata[b"huggingface"] = json.dumps(value).encode("utf-8")
        table = table.replace_schema_metadata(metadata)
    return table


def output_names(names: list[str], bindings: list[ArmBinding]) -> list[str]:
    pose_indices = {index for binding in bindings for index in binding.pose_indices}
    result: list[str] = []
    for binding in bindings:
        result.extend(names[index] for index in binding.pose_indices)
        result.append(f"{binding.gripper_id}.width_m")
    result.extend(name for index, name in enumerate(names) if index not in pose_indices)
    return result


def add_or_overwrite(state: np.ndarray, widths: dict[str, np.ndarray], bindings: list[ArmBinding], names: list[str], mode: str) -> np.ndarray:
    if mode == "overwrite":
        result = state.copy()
        for binding in bindings:
            name = f"{binding.gripper_id}.width_m"
            matches = [index for index, value in enumerate(names) if value == name]
            if len(matches) != 1:
                raise ValueError(f"mode=overwrite requires exactly one state field {name!r}; found {len(matches)}")
            result[:, matches[0]] = widths[binding.gripper_id]
        return result
    if mode != "add":
        raise ValueError("preprocessing.add_gripper_width.mode must be add or overwrite")
    pose_indices = {index for binding in bindings for index in binding.pose_indices}
    columns: list[np.ndarray] = []
    for binding in bindings:
        columns.extend(state[:, index] for index in binding.pose_indices)
        columns.append(widths[binding.gripper_id])
    columns.extend(state[:, index] for index in range(state.shape[1]) if index not in pose_indices)
    return np.column_stack(columns).astype(np.float32)


def state_stats(values: np.ndarray) -> dict[str, list[Any]]:
    return {
        "min": values.min(axis=0).tolist(), "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0, dtype=np.float64).tolist(),
        "std": values.std(axis=0, dtype=np.float64).tolist(), "count": [len(values)],
    }


def merge_stats(items: list[dict[str, list[Any]]]) -> dict[str, list[Any]]:
    counts = np.asarray([item["count"][0] for item in items], dtype=np.float64)
    means = np.asarray([item["mean"] for item in items], dtype=np.float64)
    variances = np.square(np.asarray([item["std"] for item in items], dtype=np.float64))
    weights = counts[:, None]
    total = float(counts.sum())
    mean = np.sum(means * weights, axis=0) / total
    second = np.sum((variances + means * means) * weights, axis=0) / total
    return {
        "min": np.min([item["min"] for item in items], axis=0).tolist(),
        "max": np.max([item["max"] for item in items], axis=0).tolist(),
        "mean": mean.tolist(), "std": np.sqrt(np.maximum(second - mean * mean, 0)).tolist(),
        "count": [int(total)],
    }


def convert(source: Path, output: Path, *, mode: str, overwrite: bool = False) -> Path:
    source, output = source.expanduser().resolve(), output.expanduser().resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Input dataset does not exist: {source}")
    if output == source or source in output.parents or output in source.parents:
        raise ValueError("Width output must be separate from the input dataset")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Output already exists and overwrite is false: {output}")
    info = read_json(source / "meta/info.json")
    if info.get("codebase_version") != "v2.1":
        raise ValueError("Only LeRobot v2.1 datasets are supported")
    config_path = find_acquisition_config(source)
    config = load_yaml(config_path)
    bindings = resolve_bindings(config, info)
    names = info["features"][STATE_KEY]["names"]
    width = int(info["features"][STATE_KEY]["shape"][0])
    if not isinstance(names, list) or len(names) != width:
        raise ValueError("observation.state names and shape disagree")
    if mode not in {"add", "overwrite"}:
        raise ValueError("preprocessing.add_gripper_width.mode must be add or overwrite")
    for binding in bindings:
        count = names.count(f"{binding.gripper_id}.width_m")
        if mode == "overwrite" and count != 1:
            raise ValueError(f"mode=overwrite requires one {binding.gripper_id}.width_m field; found {count}")
        if mode == "add" and count:
            raise ValueError(f"mode=add cannot add existing {binding.gripper_id}.width_m field")
    if mode == "add" and any(name.endswith(".width_m") for name in names):
        raise ValueError("mode=add does not support an input that already contains gripper width fields")
    detectors = {binding.gripper_id: load_detector(source, config, info, binding) for binding in bindings}
    episodes = read_jsonl(source / "meta/episodes.jsonl")
    stats = read_jsonl(source / "meta/episodes_stats.jsonl")
    stats_by_episode = {int(row["episode_index"]): row for row in stats}
    if len(episodes) != int(info["total_episodes"]) or len(stats_by_episode) != len(episodes):
        raise ValueError("Episode metadata counts disagree")

    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent))
    report: list[dict[str, Any]] = []
    try:
        shutil.copytree(source, stage, dirs_exist_ok=True)
        for number, episode in enumerate(episodes, start=1):
            index, length = int(episode["episode_index"]), int(episode["length"])
            parquet = dataset_path(stage, info["data_path"], index, int(info["chunks_size"]))
            table = pq.read_table(parquet)
            if table.num_rows != length or table["frame_index"].to_pylist() != list(range(length)):
                raise ValueError(f"Episode {index} Parquet rows/frame_index disagree")
            widths: dict[str, np.ndarray] = {}
            episode_report: dict[str, Any] = {"episode_index": index, "grippers": {}}
            for binding in bindings:
                feature = info["features"][binding.video_key]
                frame_size = (int(feature["shape"][1]), int(feature["shape"][0]))
                video = dataset_path(source, info["video_path"], index, int(info["chunks_size"]), video_key=binding.video_key)
                measured = measure_video(video, length, frame_size, *detectors[binding.gripper_id])
                widths[binding.gripper_id], missing = interpolate(measured, index, binding.gripper_id)
                episode_report["grippers"][binding.gripper_id] = {
                    "detected": length - missing, "interpolated": missing, "frames": length,
                }
            state = state_array(table, width)
            updated_state = add_or_overwrite(state, widths, bindings, names, mode)
            pq.write_table(replace_state(table, updated_state), parquet, compression="snappy")
            stats_by_episode[index]["stats"][STATE_KEY] = state_stats(updated_state)
            report.append(episode_report)
            print(f"[{number}/{len(episodes)}] width episode {index:06d}", flush=True)

        final_names = names if mode == "overwrite" else output_names(names, bindings)
        new_info = copy.deepcopy(info)
        new_info["features"][STATE_KEY]["shape"] = [len(final_names)]
        new_info["features"][STATE_KEY]["names"] = final_names
        write_json(stage / "meta/info.json", new_info)
        write_jsonl(stage / "meta/episodes_stats.jsonl", stats)
        global_stats_path = stage / "meta/stats.json"
        if global_stats_path.is_file():
            global_stats = read_json(global_stats_path)
            global_stats[STATE_KEY] = merge_stats([row["stats"][STATE_KEY] for row in stats])
            write_json(global_stats_path, global_stats)
        write_json(stage / "meta/gripper_width_report.json", {
            "source": str(source), "mode": mode, "acquisition_config": str(config_path.relative_to(source)),
            "episodes": report,
        })
        publish_directory(stage, output, overwrite)
        print(f"Added gripper widths to {output}: mode={mode}, episodes={len(episodes)}")
        return output
    finally:
        if stage.exists():
            shutil.rmtree(stage)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=("add", "overwrite"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--config", type=Path, help="Pipeline YAML containing preprocessing.add_gripper_width")
    args = parser.parse_args()
    if args.config:
        if args.input is not None or args.output is not None or args.mode is not None or args.overwrite:
            parser.error("--config cannot be combined with --input, --output, --mode, or --overwrite")
        with args.config.expanduser().open(encoding="utf-8") as stream:
            config = yaml.safe_load(stream)
        if not isinstance(config, dict):
            raise ValueError("Pipeline config must contain a YAML mapping")
        preprocessing = config.get("preprocessing", {})
        if not isinstance(preprocessing, dict):
            raise ValueError("preprocessing must be a mapping")
        settings = preprocessing.get("add_gripper_width", {})
        if not isinstance(settings, dict):
            raise ValueError("preprocessing.add_gripper_width must be a mapping")
        enabled = settings.get("enabled", False)
        if not isinstance(enabled, bool):
            raise ValueError("preprocessing.add_gripper_width.enabled must be true or false")
        if not enabled:
            print("Skipping gripper width detection: preprocessing.add_gripper_width.enabled is false")
            return
        for key in ("mode", "output", "overwrite"):
            if key not in settings:
                raise ValueError(f"preprocessing.add_gripper_width.{key} is required")
        if settings["mode"] not in {"add", "overwrite"}:
            raise ValueError("preprocessing.add_gripper_width.mode must be add or overwrite")
        if not isinstance(settings["overwrite"], bool):
            raise ValueError("preprocessing.add_gripper_width.overwrite must be true or false")
        merge = preprocessing.get("merge", {})
        if isinstance(merge, dict) and merge.get("enabled") is True:
            source = merge.get("output")
        else:
            source = config.get("source", {}).get("root")
        if not source:
            raise ValueError("source.root is required when preprocessing.merge is disabled")
        convert(Path(source), Path(settings["output"]), mode=settings["mode"], overwrite=settings["overwrite"])
    else:
        if args.input is None or args.output is None or args.mode is None:
            parser.error("provide --config, or --input, --output and --mode")
        convert(args.input, args.output, mode=args.mode, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
