#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.metadata
import json
import shutil
import subprocess
import sys
import uuid
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dataset_tools.tools.xense_runtime import RuntimeCatalog
from dataset_tools.tools.xense_runtime import sha256_file
from pipeline_common import PIPELINE_MARKER
from pipeline_common import episode_path
from pipeline_common import find_task_config
from pipeline_common import load_config
from pipeline_common import load_frame_selection_manifests
from pipeline_common import output_root
from pipeline_common import read_json
from pipeline_common import read_jsonl
from pipeline_common import selected_source_frame_indices
from pipeline_common import source_root
from pipeline_common import work_root
from pipeline_common import write_json
from pipeline_common import write_jsonl


@dataclass(frozen=True)
class XenseSensor:
    name: str
    serial: str
    source_video_key: str

    @property
    def difference_key(self) -> str:
        return f"observation.images.xense_{self.name}_difference"

    @property
    def depth_key(self) -> str:
        return f"observation.images.xense_{self.name}_depth"

    @property
    def force_video_key(self) -> str:
        return f"observation.images.xense_{self.name}_force"

    @property
    def force_key(self) -> str:
        return f"observation.xense.{self.name}.force"

    @property
    def force_resultant_key(self) -> str:
        return f"observation.xense.{self.name}.force_resultant"


def discover_xense_sensors(config: dict[str, Any], source_info: dict[str, Any]) -> list[XenseSensor]:
    task_path = find_task_config(config)
    with task_path.open(encoding="utf-8") as stream:
        task = yaml.safe_load(stream)
    configured = task.get("robot", {}).get("xense", {})
    if not isinstance(configured, dict) or not configured:
        raise ValueError(f"Task config has no robot.xense sensors: {task_path}")
    sensors: list[XenseSensor] = []
    for name, serial in configured.items():
        if not isinstance(name, str) or not name or not isinstance(serial, str) or not serial:
            raise ValueError(f"Invalid robot.xense entry in {task_path}: {name!r}: {serial!r}")
        video_key = f"observation.images.xense_{serial}"
        feature = source_info.get("features", {}).get(video_key)
        if not isinstance(feature, dict) or feature.get("dtype") != "video":
            raise ValueError(f"Source dataset has no Xense Rectify video feature {video_key!r}")
        sensors.append(XenseSensor(name, serial, video_key))
    if len({sensor.serial for sensor in sensors}) != len(sensors):
        raise ValueError("robot.xense must not assign the same serial to multiple sensor roles")
    return sensors


@cache
def load_xense_sensor() -> Any:
    import onnxruntime as ort

    ort.preload_dlls(directory="")
    from xensesdk import Sensor

    return Sensor


def sdk_version() -> str:
    for distribution in ("XenseSDK", "xensesdk"):
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            pass
    return "unknown"


def fixed_tensor_array(values: np.ndarray) -> pa.Array:
    values = np.asarray(values, dtype=np.float32)
    data_type: pa.DataType = pa.float32()
    for size in reversed(values.shape[1:]):
        data_type = pa.list_(data_type, int(size))
    return pa.array(values.tolist(), type=data_type)


def update_huggingface_metadata(
    table: pa.Table, fields: dict[str, tuple[str, tuple[int, ...]]]
) -> pa.Table:
    metadata = dict(table.schema.metadata or {})
    raw = metadata.get(b"huggingface")
    document = json.loads(raw.decode("utf-8")) if raw else {"info": {"features": {}}}
    features = document.setdefault("info", {}).setdefault("features", {})
    for name, (dtype, shape) in fields.items():
        feature: dict[str, Any] = {"dtype": dtype, "_type": "Value"}
        for size in reversed(shape):
            feature = {"feature": feature, "length": int(size), "_type": "Sequence"}
        features[name] = feature
    metadata[b"huggingface"] = json.dumps(document, separators=(",", ":")).encode("utf-8")
    return table.replace_schema_metadata(metadata)


def tensor_stats(values: np.ndarray) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 2 or values.shape[0] == 0:
        raise ValueError(f"Expected a non-empty tensor sequence, got {values.shape}")
    return {
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "count": [int(values.shape[0])],
    }


def depth_to_gray_rgb(
    depth: np.ndarray,
    minimum: float,
    maximum: float,
    out_of_range: str,
) -> tuple[np.ndarray, dict[str, float | int]]:
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2 or not np.all(np.isfinite(depth)):
        raise ValueError(f"Depth must be a finite 2-D float array, got {depth.shape}")
    if not np.isfinite(minimum) or not np.isfinite(maximum) or maximum <= minimum:
        raise ValueError(f"Invalid depth encoding range [{minimum}, {maximum}]")
    below = int(np.count_nonzero(depth < minimum))
    above = int(np.count_nonzero(depth > maximum))
    if (below or above) and out_of_range == "error":
        raise ValueError(
            f"Depth range [{float(np.min(depth)):g}, {float(np.max(depth)):g}] contains "
            f"{below} values below {minimum:g} and {above} values above {maximum:g}"
        )
    if out_of_range not in {"error", "clip"}:
        raise ValueError("xense.depth.out_of_range must be error or clip")
    normalized = (np.clip(depth, minimum, maximum) - minimum) / (maximum - minimum)
    gray = np.rint(normalized * 255.0).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2), {
        "min": float(np.min(depth)),
        "max": float(np.max(depth)),
        "below_range": below,
        "above_range": above,
    }


def force_to_rgb(force: np.ndarray, minimum: np.ndarray, maximum: np.ndarray) -> np.ndarray:
    force = np.asarray(force, dtype=np.float32)
    minimum = np.asarray(minimum, dtype=np.float32)
    maximum = np.asarray(maximum, dtype=np.float32)
    if force.shape[-1:] != (3,) or minimum.shape != (3,) or maximum.shape != (3,):
        raise ValueError(
            f"Force encoding expects [...,3] values and three-channel ranges, got "
            f"{force.shape}, {minimum.shape}, {maximum.shape}"
        )
    if not np.all(np.isfinite(force)) or not np.all(np.isfinite(minimum)) or not np.all(np.isfinite(maximum)):
        raise ValueError("Force encoding inputs must be finite")
    span = maximum - minimum
    if np.any(span < 0):
        raise ValueError(f"Force encoding maximum is below minimum: {minimum} -> {maximum}")
    denominator = np.where(span > 0, span, np.float32(1.0))
    normalized = (force - minimum) / denominator
    normalized[..., span == 0] = 0.0
    return np.rint(np.clip(normalized, 0.0, 1.0) * 255.0).astype(np.uint8)


def pad_force_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(rgb, dtype=np.uint8)
    if rgb.shape != (35, 20, 3):
        raise ValueError(f"Force RGB frame must have shape [35,20,3], got {rgb.shape}")
    return np.concatenate((rgb, rgb[-1:, :, :]), axis=0)


class RawVideoWriter:
    def __init__(
        self,
        path: Path,
        *,
        width: int,
        height: int,
        fps: float,
        ffmpeg: str,
        encoder: str,
        preset: str,
        crf: int,
        pixel_format: str,
    ):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        command = [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-s:v", f"{width}x{height}",
            "-r", f"{fps:g}", "-i", "-", "-an", "-c:v", encoder,
            "-preset", preset, "-crf", str(crf), "-pix_fmt", pixel_format,
            "-movflags", "+faststart", "-f", "mp4", str(path),
        ]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

    def write(self, rgb: np.ndarray) -> None:
        frame = np.ascontiguousarray(rgb, dtype=np.uint8)
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg input pipe is closed")
        try:
            self.process.stdin.write(frame.tobytes())
        except BrokenPipeError as error:
            raise RuntimeError(self._error_message()) from error

    def _error_message(self) -> str:
        stderr = b"" if self.process.stderr is None else self.process.stderr.read()
        return f"ffmpeg failed for {self.path}:\n{stderr.decode(errors='replace')[-4000:]}"

    def close(self) -> None:
        if self.process.stdin is not None and not self.process.stdin.closed:
            self.process.stdin.close()
        return_code = self.process.wait()
        if return_code:
            raise RuntimeError(self._error_message())

    def abort(self) -> None:
        if self.process.poll() is None:
            self.process.kill()
            self.process.wait()
        self.path.unlink(missing_ok=True)


def create_video_writer(path: Path, shape: tuple[int, int], fps: float, settings: dict[str, Any], crf: int) -> RawVideoWriter:
    height, width = shape
    return RawVideoWriter(
        path,
        width=width,
        height=height,
        fps=fps,
        ffmpeg=str(settings.get("ffmpeg", "ffmpeg")),
        encoder=str(settings.get("encoder", "libx264")),
        preset=str(settings.get("preset", "slow")),
        crf=crf,
        pixel_format=str(settings.get("pixel_format", "yuv420p")),
    )


def process_sensor_episode(
    *,
    Sensor: Any,
    solver: Any,
    sensor: XenseSensor,
    video: Path,
    episode_index: int,
    source_length: int,
    selected_indices: list[int],
    difference_path: Path | None,
    depth_path: Path | None,
    fps: float,
    settings: dict[str, Any],
) -> tuple[dict[str, np.ndarray], dict[str, Any], np.ndarray | None]:
    output_names = ["Difference", "Depth", "Force", "ForceResultant"]
    output_types = [getattr(Sensor.OutputType, name) for name in output_names]
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open Xense Rectify video: {video}")
    selected = set(selected_indices)
    force_values: list[np.ndarray] = []
    resultant_values: list[np.ndarray] = []
    difference_writer: RawVideoWriter | None = None
    depth_writer: RawVideoWriter | None = None
    observed_shapes: dict[str, list[int]] = {}
    depth_summary = {
        "min": float("inf"), "max": float("-inf"),
        "below_range": 0, "above_range": 0,
    }
    depth_config = settings["depth"]
    depth_range = depth_config.get("range", [0.0, 1.0])
    minimum, maximum = float(depth_range[0]), float(depth_range[1])

    try:
        for frame_index in range(source_length):
            ok, frame = capture.read()
            if not ok:
                raise ValueError(
                    f"Episode {episode_index} {sensor.name} Rectify video ended at frame "
                    f"{frame_index}; expected {source_length}"
                )
            try:
                result = solver.selectSensorInfo(*output_types, rectify_image=frame)
            except Exception as error:
                raise RuntimeError(
                    f"Xense SDK failed at episode {episode_index}, sensor {sensor.name}, "
                    f"source frame {frame_index}"
                ) from error
            difference, depth, force, resultant = (np.asarray(value) for value in result)
            values = dict(zip(output_names, (difference, depth, force, resultant), strict=True))
            for name, value in values.items():
                if not np.all(np.isfinite(value)):
                    raise ValueError(
                        f"Non-finite Xense {name} at episode {episode_index}, "
                        f"sensor {sensor.name}, source frame {frame_index}"
                    )
                shape = list(value.shape)
                if name in observed_shapes and observed_shapes[name] != shape:
                    raise ValueError(
                        f"Xense {sensor.name} {name} shape changed from "
                        f"{observed_shapes[name]} to {shape}"
                    )
                observed_shapes.setdefault(name, shape)
            if difference.ndim != 3 or difference.shape[2] != 3 or difference.dtype != np.uint8:
                raise ValueError(f"Unexpected Difference shape/dtype: {difference.shape} {difference.dtype}")
            if force.shape != (35, 20, 3) or resultant.shape != (6,):
                raise ValueError(
                    f"Unexpected Xense force shapes: Force={force.shape}, ForceResultant={resultant.shape}"
                )
            if frame_index not in selected:
                continue

            if bool(settings["difference"].get("enabled", True)):
                if difference_writer is None:
                    if difference_path is None:
                        raise RuntimeError("Difference output path is missing")
                    difference_writer = create_video_writer(
                        difference_path,
                        difference.shape[:2],
                        fps,
                        settings["video"],
                        int(settings["difference"].get("crf", 18)),
                    )
                difference_writer.write(difference[..., ::-1])

            depth_rgb, summary = depth_to_gray_rgb(
                depth,
                minimum,
                maximum,
                str(depth_config.get("out_of_range", "clip")),
            )
            depth_summary["min"] = min(float(depth_summary["min"]), float(summary["min"]))
            depth_summary["max"] = max(float(depth_summary["max"]), float(summary["max"]))
            depth_summary["below_range"] += int(summary["below_range"])
            depth_summary["above_range"] += int(summary["above_range"])
            if bool(depth_config.get("enabled", True)):
                if depth_writer is None:
                    if depth_path is None:
                        raise RuntimeError("Depth output path is missing")
                    depth_writer = create_video_writer(
                        depth_path,
                        depth.shape,
                        fps,
                        settings["video"],
                        int(depth_config.get("crf", 0)),
                    )
                depth_writer.write(depth_rgb)

            if bool(settings["force"].get("parquet", True)) or bool(settings["force"].get("video", False)):
                force_values.append(force.astype(np.float32, copy=True))
            if bool(settings["force_resultant"].get("enabled", True)):
                resultant_values.append(resultant.astype(np.float32, copy=True))

        ok, _ = capture.read()
        if ok:
            raise ValueError(
                f"Episode {episode_index} {sensor.name} Rectify video has more than "
                f"the expected {source_length} frames"
            )
        for writer in (difference_writer, depth_writer):
            if writer is not None:
                writer.close()
    except BaseException:
        for writer in (difference_writer, depth_writer):
            if writer is not None:
                writer.abort()
        raise
    finally:
        capture.release()

    expected = len(selected_indices)
    arrays: dict[str, np.ndarray] = {}
    force_array = None
    if bool(settings["force"].get("parquet", True)) or bool(settings["force"].get("video", False)):
        if len(force_values) != expected:
            raise RuntimeError(f"Expected {expected} Force frames, got {len(force_values)}")
        force_array = np.stack(force_values)
        if bool(settings["force"].get("parquet", True)):
            arrays[sensor.force_key] = force_array
    if bool(settings["force_resultant"].get("enabled", True)):
        if len(resultant_values) != expected:
            raise RuntimeError(f"Expected {expected} ForceResultant frames, got {len(resultant_values)}")
        arrays[sensor.force_resultant_key] = np.stack(resultant_values)
    return arrays, {"shapes": observed_shapes, "depth": depth_summary}, force_array


def write_visualization_assets(
    output: Path,
    settings: dict[str, Any],
    force_ranges: dict[str, dict[str, Any]] | None = None,
) -> dict[str, str]:
    final = output / "assets" / "xense"
    temporary = output / "assets" / f".xense.tmp-{uuid.uuid4().hex}"
    temporary.mkdir(parents=True)
    depth_range = settings["depth"].get("range", [0.0, 1.0])
    encoding = {
        "schema_version": 1,
        "feature_encoding": "unorm8_gray_rgb",
        "source_dtype": "float32",
        "stored_dtype": "uint8",
        "range": [float(depth_range[0]), float(depth_range[1])],
        "formula": "code = round(clip((depth-min)/(max-min), 0, 1) * 255)",
        "approximate_decode": "depth = mean(decoded_rgb, axis=channels) / 255 * (max-min) + min",
        "base_quantization_max_error": (float(depth_range[1]) - float(depth_range[0])) / 510.0,
        "notes": "H.264 RGB-to-YUV conversion can add codec error beyond base quantization.",
    }
    write_json(temporary / "depth_encoding.json", encoding)

    if bool(settings.get("force", {}).get("video", False)):
        if not force_ranges:
            raise ValueError("Force video is enabled but no dataset Force ranges were collected")
        force_encoding = {
            "schema_version": 1,
            "feature_encoding": "dataset_minmax_uint8_rgb",
            "range_scope": "selected_frames_across_entire_output_dataset_per_sensor_channel",
            "source_dtype": "float32",
            "source_shape": [35, 20, 3],
            "stored_dtype": "uint8",
            "stored_shape": [36, 20, 3],
            "channels": {"R": "Force X", "G": "Force Y", "B": "Force Z"},
            "padding": {"axis": "height", "source_rows": 35, "stored_rows": 36, "rule": "replicate_last_row"},
            "formula": "code_c = round((force_c-min_c)/(max_c-min_c)*255)",
            "constant_channel_rule": "code=0 and decoded value=min=max",
            "approximate_decode": "force_c = code_c/255*(max_c-min_c)+min_c",
            "codec": {
                "encoder": str(settings["video"].get("encoder", "libx264")),
                "pixel_format": str(settings["video"].get("pixel_format", "yuv420p")),
                "crf": int(settings["force"].get("crf", 0)),
            },
            "notes": "H.264 RGB-to-YUV conversion and chroma subsampling add error beyond uint8 quantization.",
            "sensors": force_ranges,
        }
        write_json(temporary / "force_encoding.json", force_encoding)

    asset_settings = settings.get("visualization_assets", {})
    save_colormaps = bool(asset_settings.get("save_colormaps", True))
    save_preview = bool(asset_settings.get("save_preview_png", True))
    if save_colormaps or save_preview:
        codes = np.arange(256, dtype=np.uint8).reshape(256, 1)
        opencv_rgb = cv2.applyColorMap(codes, cv2.COLORMAP_TURBO)[:, 0, ::-1]
        # Xense's official viewer uses the canonical Turbo map; OpenCV 4.x ships the same 256-entry LUT.
        sdk_rgb = opencv_rgb.copy()
        if save_colormaps:
            colormaps = {
                "schema_version": 1,
                "index": "stored uint8 depth code",
                "channels": "RGB",
                "xense_sdk_turbo": sdk_rgb.tolist(),
                "opencv_turbo": opencv_rgb.tolist(),
                "web_adaptive_recipe": {
                    "input": "positive float32 depth values",
                    "normalization": "per-frame percentiles P1..P99",
                    "quantization": "uint8",
                    "colormap": "OpenCV COLORMAP_TURBO",
                    "numeric_recovery": False,
                },
                "opencv_version": cv2.__version__,
                "xense_sdk_version": sdk_version(),
            }
            write_json(temporary / "depth_colormaps.json", colormaps)
        if save_preview:
            strip = np.repeat(opencv_rgb[None, :, :], 48, axis=0)
            preview = np.zeros((128, 512, 3), dtype=np.uint8)
            preview[16:64, :256] = strip
            preview[80:128, :256] = strip
            cv2.putText(preview, "Xense SDK Turbo", (270, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            cv2.putText(preview, "OpenCV Turbo", (270, 112), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
            if not cv2.imwrite(str(temporary / "depth_colormaps_preview.png"), preview[..., ::-1]):
                raise RuntimeError("Could not write Xense depth colormap preview")

    if final.exists():
        shutil.rmtree(final)
    temporary.replace(final)
    return {
        path.name: sha256_file(path)
        for path in sorted(final.iterdir())
        if path.is_file()
    }


def validate_settings(config: dict[str, Any]) -> dict[str, Any]:
    settings = config.get("xense", {})
    if not isinstance(settings, dict):
        raise ValueError("xense must be a mapping")
    for name in ("video", "difference", "depth", "force", "force_resultant"):
        if not isinstance(settings.get(name), dict):
            raise ValueError(f"xense.{name} must be a mapping")
    if settings["depth"].get("encoding", "unorm8_gray_rgb") != "unorm8_gray_rgb":
        raise ValueError("Only xense.depth.encoding=unorm8_gray_rgb is supported")
    settings["depth"].setdefault("out_of_range", "clip")
    force = settings["force"]
    if "enabled" in force:
        raise ValueError("xense.force.enabled is no longer supported; use xense.force.parquet and xense.force.video")
    for key, default in (("parquet", True), ("video", True)):
        force.setdefault(key, default)
        if not isinstance(force[key], bool):
            raise ValueError(f"xense.force.{key} must be a boolean")
    force.setdefault("encoding", "dataset_minmax_uint8_rgb")
    if force["encoding"] != "dataset_minmax_uint8_rgb":
        raise ValueError("Only xense.force.encoding=dataset_minmax_uint8_rgb is supported")
    force.setdefault("dtype", "float32")
    if force["dtype"] != "float32":
        raise ValueError("xense.force.dtype must be float32")
    crf = force.setdefault("crf", 0)
    if not isinstance(crf, int) or isinstance(crf, bool) or not 0 <= crf <= 51:
        raise ValueError("xense.force.crf must be an integer from 0 to 51")
    if not any((
        bool(settings["difference"].get("enabled", True)),
        bool(settings["depth"].get("enabled", True)),
        bool(force["parquet"]),
        bool(force["video"]),
        bool(settings["force_resultant"].get("enabled", True)),
    )):
        raise ValueError("At least one Xense output must be enabled")
    return settings


def main() -> None:
    parser = argparse.ArgumentParser(description="Reconstruct selected Xense outputs after final frame selection.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    config = load_config(args.config)
    if not bool(config.get("xense", {}).get("enabled", False)):
        print("xense.enabled=false; skipping offline Xense processing")
        return
    settings = validate_settings(config)
    source, output, work = source_root(config), output_root(config), work_root(config)
    marker_path = output / PIPELINE_MARKER
    marker = read_json(marker_path)
    if marker.get("xense_status") in {"running", "complete"}:
        raise RuntimeError(
            "This output has already entered Xense processing. Rerun 03_convert_trajectories.py "
            "to rebuild it before applying stage 09 again."
        )
    marker["xense_status"] = "running"
    write_json(marker_path, marker)
    force_cache_root: Path | None = None

    try:
        source_info = read_json(source / "meta/info.json")
        source_episodes = {
            int(row["episode_index"]): row for row in read_jsonl(source / "meta/episodes.jsonl")
        }
        episodes = read_jsonl(output / "meta/episodes.jsonl")
        sensors = discover_xense_sensors(config, source_info)
        runtime_catalog = RuntimeCatalog(source, list(source_episodes.values()))
        trim_by_episode, sparsification_by_episode = load_frame_selection_manifests(config)
        Sensor = load_xense_sensor()
        source_chunks = int(source_info.get("chunks_size", 1000))
        output_chunks = int(config["output"].get("chunks_size", 1000))
        fps = float(source_info["fps"])
        manifest_rows: list[dict[str, Any]] = []
        field_shapes: dict[str, list[int]] = {}
        sdk_shapes: dict[str, list[int]] = {}
        force_ranges: dict[str, dict[str, Any]] = {}
        force_jobs: list[dict[str, Any]] = []
        if bool(settings["force"].get("video", False)):
            force_cache_root = work / f".09_force_cache-{uuid.uuid4().hex}"
            force_cache_root.mkdir(parents=True)

        for ordinal, episode in enumerate(episodes, start=1):
            episode_index = int(episode["episode_index"])
            source_row = source_episodes[episode_index]
            source_length = int(source_row["length"])
            selected_indices = selected_source_frame_indices(
                config,
                episode_index,
                source_length,
                trim_by_episode=trim_by_episode,
                sparsification_by_episode=sparsification_by_episode,
            )
            if len(selected_indices) != int(episode["length"]):
                raise ValueError(
                    f"Episode {episode_index} selects {len(selected_indices)} Xense frames, "
                    f"but final Parquet has {episode['length']} rows"
                )
            parquet = episode_path(output, source_info["data_path"], episode_index, output_chunks)
            table = pq.read_table(parquet)
            episode_stats: dict[str, Any] = {}
            sensor_rows: dict[str, Any] = {}
            for sensor in sensors:
                asset = runtime_catalog.resolve(source_row, sensor.name, sensor.serial)
                source_video = episode_path(
                    source,
                    source_info["video_path"],
                    episode_index,
                    source_chunks,
                    video_key=sensor.source_video_key,
                )
                difference_path = episode_path(
                    output,
                    source_info["video_path"],
                    episode_index,
                    output_chunks,
                    video_key=sensor.difference_key,
                )
                depth_path = episode_path(
                    output,
                    source_info["video_path"],
                    episode_index,
                    output_chunks,
                    video_key=sensor.depth_key,
                )
                force_video_path = episode_path(
                    output,
                    source_info["video_path"],
                    episode_index,
                    output_chunks,
                    video_key=sensor.force_video_key,
                )
                difference_temp = difference_path.with_suffix(".tmp.mp4")
                depth_temp = depth_path.with_suffix(".tmp.mp4")
                solver = Sensor.createSolver(asset.runtime)
                if not solver:
                    raise RuntimeError(f"Xense SDK could not create solver from {asset.runtime}")
                try:
                    arrays, details, force_frames = process_sensor_episode(
                        Sensor=Sensor,
                        solver=solver,
                        sensor=sensor,
                        video=source_video,
                        episode_index=episode_index,
                        source_length=source_length,
                        selected_indices=selected_indices,
                        difference_path=difference_temp if settings["difference"].get("enabled", True) else None,
                        depth_path=depth_temp if settings["depth"].get("enabled", True) else None,
                        fps=fps,
                        settings=settings,
                    )
                finally:
                    if hasattr(solver, "release"):
                        solver.release()
                if settings["difference"].get("enabled", True):
                    difference_temp.replace(difference_path)
                if settings["depth"].get("enabled", True):
                    depth_temp.replace(depth_path)
                if bool(settings["force"].get("video", False)):
                    if force_frames is None or force_cache_root is None:
                        raise RuntimeError(f"Xense Force frames were not collected for {sensor.name}")
                    channel_min = np.min(force_frames, axis=(0, 1, 2))
                    channel_max = np.max(force_frames, axis=(0, 1, 2))
                    collected = force_ranges.setdefault(sensor.name, {
                        "serial": sensor.serial,
                        "minimum": [float("inf")] * 3,
                        "maximum": [float("-inf")] * 3,
                    })
                    collected["minimum"] = np.minimum(
                        np.asarray(collected["minimum"], dtype=np.float64), channel_min,
                    ).tolist()
                    collected["maximum"] = np.maximum(
                        np.asarray(collected["maximum"], dtype=np.float64), channel_max,
                    ).tolist()
                    force_cache = force_cache_root / f"episode_{episode_index:06d}_{len(force_jobs):04d}.npy"
                    np.save(force_cache, force_frames, allow_pickle=False)
                for output_name, shape in details["shapes"].items():
                    shape_key = f"{sensor.name}:{output_name}"
                    if shape_key in sdk_shapes and sdk_shapes[shape_key] != shape:
                        raise ValueError(
                            f"Xense {shape_key} shape changed across episodes: "
                            f"{sdk_shapes[shape_key]} -> {shape}"
                        )
                    sdk_shapes.setdefault(shape_key, shape)
                depth_details = details["depth"]
                clipped = int(depth_details["below_range"]) + int(depth_details["above_range"])
                if clipped:
                    print(
                        f"[Xense Depth] episode {episode_index:06d} {sensor.name}: "
                        f"observed=[{depth_details['min']:.6g}, {depth_details['max']:.6g}], "
                        f"encoded=[{settings['depth']['range'][0]:g}, {settings['depth']['range'][1]:g}], "
                        f"clipped below={depth_details['below_range']}, "
                        f"above={depth_details['above_range']}",
                        flush=True,
                    )
                for field, values in arrays.items():
                    shape = list(values.shape[1:])
                    if field in field_shapes and field_shapes[field] != shape:
                        raise ValueError(
                            f"Xense field {field} shape changed across episodes: "
                            f"{field_shapes[field]} -> {shape}"
                        )
                    table = table.append_column(field, fixed_tensor_array(values))
                    field_shapes[field] = shape
                    episode_stats[field] = tensor_stats(values)
                sensor_row = {
                    "serial": sensor.serial,
                    "runtime_set_id": asset.runtime_set_id,
                    "runtime_sha256": sha256_file(asset.runtime),
                    "source_video": str(source_video.relative_to(source)),
                    "difference_video": str(difference_path.relative_to(output)) if difference_path.exists() else None,
                    "difference_bytes": difference_path.stat().st_size if difference_path.exists() else None,
                    "depth_video": str(depth_path.relative_to(output)) if depth_path.exists() else None,
                    "depth_bytes": depth_path.stat().st_size if depth_path.exists() else None,
                    "force_video": None,
                    "force_video_bytes": None,
                    **details,
                }
                sensor_rows[sensor.name] = sensor_row
                if bool(settings["force"].get("video", False)):
                    force_jobs.append({
                        "sensor": sensor,
                        "cache": force_cache,
                        "path": force_video_path,
                        "manifest": sensor_row,
                    })
            table = update_huggingface_metadata(
                table,
                {field: ("float32", tuple(shape)) for field, shape in field_shapes.items()},
            )
            temporary = parquet.with_suffix(".parquet.tmp")
            pq.write_table(table, temporary, compression="zstd")
            temporary.replace(parquet)
            manifest_rows.append({
                "episode_index": episode_index,
                "source_frames": source_length,
                "selected_frames": len(selected_indices),
                "selected_source_frame_indices": selected_indices,
                "sensors": sensor_rows,
                "stats": episode_stats,
            })
            print(
                f"[{ordinal}/{len(episodes)}] Xense episode {episode_index:06d}: "
                f"{source_length} source -> {len(selected_indices)} selected frames",
                flush=True,
            )

        if bool(settings["force"].get("video", False)):
            for sensor_name, values in force_ranges.items():
                minimum = np.asarray(values["minimum"], dtype=np.float32)
                maximum = np.asarray(values["maximum"], dtype=np.float32)
                values["minimum"] = minimum.tolist()
                values["maximum"] = maximum.tolist()
                values["quantization_step"] = ((maximum - minimum) / 255.0).tolist()
                values["base_quantization_max_error"] = ((maximum - minimum) / 510.0).tolist()
                values["constant_channels"] = [
                    channel for channel, span in zip(("x", "y", "z"), maximum - minimum, strict=True)
                    if span == 0
                ]
                values["feature"] = next(
                    sensor.force_video_key for sensor in sensors if sensor.name == sensor_name
                )
            for force_ordinal, job in enumerate(force_jobs, start=1):
                sensor = job["sensor"]
                values = force_ranges[sensor.name]
                minimum = np.asarray(values["minimum"], dtype=np.float32)
                maximum = np.asarray(values["maximum"], dtype=np.float32)
                force_video_path = job["path"]
                force_video_temp = force_video_path.with_suffix(".tmp.mp4")
                writer = create_video_writer(
                    force_video_temp,
                    (36, 20),
                    fps,
                    settings["video"],
                    int(settings["force"].get("crf", 0)),
                )
                try:
                    cached = np.load(job["cache"], mmap_mode="r", allow_pickle=False)
                    for force_frame in cached:
                        writer.write(pad_force_rgb(force_to_rgb(force_frame, minimum, maximum)))
                    writer.close()
                except BaseException:
                    writer.abort()
                    raise
                force_video_temp.replace(force_video_path)
                job["manifest"]["force_video"] = str(force_video_path.relative_to(output))
                job["manifest"]["force_video_bytes"] = force_video_path.stat().st_size
                print(
                    f"[Xense Force video {force_ordinal}/{len(force_jobs)}] "
                    f"{sensor.name}: {len(cached)} frames -> {force_video_path.relative_to(output)}",
                    flush=True,
                )

        asset_hashes = write_visualization_assets(output, settings, force_ranges)
        write_jsonl(work / "09_xense_episode_manifest.jsonl", manifest_rows)
        write_json(work / "09_xense_report.json", {
            "episodes": len(episodes),
            "frames": sum(int(row["selected_frames"]) for row in manifest_rows),
            "sensors": [{"name": sensor.name, "serial": sensor.serial} for sensor in sensors],
            "xense_sdk_version": sdk_version(),
            "opencv_version": cv2.__version__,
            "field_shapes": field_shapes,
            "sdk_shapes": sdk_shapes,
            "force_video_shape": [36, 20, 3] if settings["force"].get("video", False) else None,
            "assets": asset_hashes,
            "encoding": settings,
        })
        marker = read_json(marker_path)
        marker["xense_status"] = "complete"
        marker["xense_report"] = str(work / "09_xense_report.json")
        write_json(marker_path, marker)
        print(f"Processed Xense outputs into {output}")
    except BaseException as error:
        marker = read_json(marker_path)
        marker["xense_status"] = "failed"
        marker["xense_error"] = f"{type(error).__name__}: {error}"
        write_json(marker_path, marker)
        raise
    finally:
        if force_cache_root is not None:
            shutil.rmtree(force_cache_root, ignore_errors=True)


if __name__ == "__main__":
    main()
