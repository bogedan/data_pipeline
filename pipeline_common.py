from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import yaml


POSE_NAMES = [
    "x",
    "y",
    "z",
    "rot6d_0",
    "rot6d_1",
    "rot6d_2",
    "rot6d_3",
    "rot6d_4",
    "rot6d_5",
    "gripper_angle_rad",
]
PIPELINE_MARKER = ".openpi_data_pipeline.json"


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a YAML mapping: {config_path}")
    config["_config_path"] = str(config_path)
    return config


def source_root(config: dict[str, Any]) -> Path:
    return Path(config["source"]["root"]).expanduser().resolve()


def output_root(config: dict[str, Any]) -> Path:
    return Path(config["output"]["root"]).expanduser().resolve()


def work_root(config: dict[str, Any]) -> Path:
    return Path(config["work_dir"]).expanduser().resolve()


def read_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.replace(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            stream.write("\n")
    temporary.replace(path)


def episode_path(root: Path, template: str, episode_index: int, chunks_size: int, **extra: str) -> Path:
    values: dict[str, Any] = {
        "episode_index": episode_index,
        "episode_chunk": episode_index // chunks_size,
        **extra,
    }
    return root / template.format(**values)


def quaternion_to_matrix(quaternion: np.ndarray, order: str = "xyzw") -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.ndim != 2 or quaternion.shape[1] != 4:
        raise ValueError(f"Expected quaternion array [N,4], got {quaternion.shape}")
    if order == "xyzw":
        x, y, z, w = np.moveaxis(quaternion, -1, 0)
    elif order == "wxyz":
        w, x, y, z = np.moveaxis(quaternion, -1, 0)
    else:
        raise ValueError(f"Unsupported quaternion order: {order!r}")

    norm = np.sqrt(x * x + y * y + z * z + w * w)
    if np.any(~np.isfinite(norm)) or np.any(norm < 1e-8):
        bad = np.flatnonzero((~np.isfinite(norm)) | (norm < 1e-8))[:10].tolist()
        raise ValueError(f"Invalid quaternion norm at rows {bad}")
    x, y, z, w = x / norm, y / norm, z / norm, w / norm

    matrix = np.empty((quaternion.shape[0], 3, 3), dtype=np.float64)
    matrix[:, 0, 0] = 1 - 2 * (y * y + z * z)
    matrix[:, 0, 1] = 2 * (x * y - z * w)
    matrix[:, 0, 2] = 2 * (x * z + y * w)
    matrix[:, 1, 0] = 2 * (x * y + z * w)
    matrix[:, 1, 1] = 1 - 2 * (x * x + z * z)
    matrix[:, 1, 2] = 2 * (y * z - x * w)
    matrix[:, 2, 0] = 2 * (x * z - y * w)
    matrix[:, 2, 1] = 2 * (y * z + x * w)
    matrix[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return matrix.astype(np.float32)


def map_gripper(values: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    mapping = config["trajectory"]["gripper"]
    if mapping.get("mapping", "linear") != "linear":
        raise ValueError("Only trajectory.gripper.mapping=linear is currently supported")
    calibration_path = Path(mapping["calibration_file"]).expanduser().resolve()
    with calibration_path.open(encoding="utf-8") as stream:
        calibration = yaml.safe_load(stream)
    distance_section_name = str(mapping.get("distance_section", "distance"))
    radians_section_name = str(mapping.get("radians_section", "width_to_rad"))
    if distance_section_name not in calibration:
        raise KeyError(f"Missing {distance_section_name!r} in gripper calibration: {calibration_path}")
    if radians_section_name not in calibration:
        raise KeyError(f"Missing {radians_section_name!r} in gripper calibration: {calibration_path}")
    distance_section = calibration[distance_section_name]
    radians_section = calibration[radians_section_name]
    input_min = float(distance_section["observed_min_m"])
    input_max = float(distance_section["observed_max_m"])
    output_min = float(radians_section["min_rad"])
    output_max = float(radians_section["max_rad"])
    if not input_max > input_min:
        raise ValueError("trajectory.gripper.input_max must be greater than input_min")
    result = output_min + (values - input_min) * (output_max - output_min) / (input_max - input_min)
    if bool(mapping.get("clip", True)):
        result = np.clip(result, min(output_min, output_max), max(output_min, output_max))
    return result.astype(np.float32)


def _axis_rotation(axis: str, radians: float) -> np.ndarray:
    cosine, sine = math.cos(radians), math.sin(radians)
    if axis == "x":
        return np.asarray([[1, 0, 0], [0, cosine, -sine], [0, sine, cosine]], dtype=np.float32)
    if axis == "y":
        return np.asarray([[cosine, 0, sine], [0, 1, 0], [-sine, 0, cosine]], dtype=np.float32)
    if axis == "z":
        return np.asarray([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]], dtype=np.float32)
    raise ValueError(f"Unknown rotation axis: {axis}")


def apply_pose_transform(
    position: np.ndarray, rotation: np.ndarray, transform: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """Apply T_world_target = T_world_tracker @ T_tracker_target.

    intrinsic_xyz means rotations about the moving local X/Y/Z axes, hence
    R_tracker_target = Rx @ Ry @ Rz. With rotated_local translation, the
    transform's translation column is R_tracker_target @ translation_m.
    """
    enabled = transform.get("enabled")
    if enabled is None:
        raise ValueError(
            "trajectory.pose_transform.enabled must be explicitly set to true or false after confirming pose semantics"
        )
    if not bool(enabled):
        return position, rotation
    if transform.get("rotation_order", "intrinsic_xyz") != "intrinsic_xyz":
        raise ValueError("Only pose_transform.rotation_order=intrinsic_xyz is supported")
    degrees = transform["rotation_deg"]
    angles = {axis: math.radians(float(degrees[axis])) for axis in "xyz"}
    offset_rotation = (
        _axis_rotation("x", angles["x"])
        @ _axis_rotation("y", angles["y"])
        @ _axis_rotation("z", angles["z"])
    )
    translation = np.asarray(transform["translation_m"], dtype=np.float32)
    if translation.shape != (3,):
        raise ValueError("pose_transform.translation_m must contain exactly three values")
    translation_frame = transform.get("translation_frame", "rotated_local")
    if translation_frame == "rotated_local":
        offset_translation = offset_rotation @ translation
    elif translation_frame == "tracker":
        offset_translation = translation
    else:
        raise ValueError(f"Unsupported pose_transform.translation_frame: {translation_frame!r}")
    transformed_position = position + np.einsum("nij,j->ni", rotation, offset_translation)
    transformed_rotation = rotation @ offset_rotation
    return transformed_position.astype(np.float32), transformed_rotation.astype(np.float32)


def state8_to_pose10(state: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    state = np.asarray(state, dtype=np.float32)
    trajectory = config["trajectory"]
    position_indices = trajectory["position_indices"]
    quaternion_indices = trajectory["quaternion_indices"]
    gripper_index = int(trajectory["gripper_index"])
    needed = [*position_indices, *quaternion_indices, gripper_index]
    if state.ndim != 2 or state.shape[1] <= max(needed):
        raise ValueError(f"State shape {state.shape} cannot satisfy configured indices {needed}")

    position = state[:, position_indices]
    rotation = quaternion_to_matrix(state[:, quaternion_indices], trajectory.get("quaternion_order", "xyzw"))
    position, rotation = apply_pose_transform(position, rotation, trajectory.get("pose_transform", {}))
    # UMI/openpi uses the first two *columns* of R, flattened column by column.
    rot6d = np.concatenate([rotation[:, :, 0], rotation[:, :, 1]], axis=1)
    gripper = map_gripper(state[:, gripper_index], config)[:, None]
    pose10 = np.concatenate([position, rot6d, gripper], axis=1)
    if not np.all(np.isfinite(pose10)):
        raise ValueError("Converted pose contains NaN or infinity")
    return pose10.astype(np.float32)


def shifted_actions(pose10: np.ndarray, offset: int, tail_policy: str) -> np.ndarray:
    if offset < 0:
        raise ValueError("trajectory.action_offset_frames must be >= 0")
    if len(pose10) == 0:
        raise ValueError("Cannot create actions for an empty episode")
    if offset >= len(pose10):
        raise ValueError(f"Action offset {offset} must be smaller than episode length {len(pose10)}")
    if tail_policy != "repeat_last":
        raise ValueError("Only trajectory.tail_policy=repeat_last preserves video/data synchronization")
    indices = np.minimum(np.arange(len(pose10)) + offset, len(pose10) - 1)
    return pose10[indices].copy()


def pose10_rotation_matrix(pose10: np.ndarray) -> np.ndarray:
    """Recover an orthonormal rotation matrix from column-convention rot6d."""
    pose10 = np.asarray(pose10, dtype=np.float32)
    if pose10.ndim != 2 or pose10.shape[1] < 9:
        raise ValueError(f"Expected pose array [N,>=9], got {pose10.shape}")
    first = pose10[:, 3:6]
    second = pose10[:, 6:9]
    first = first / np.maximum(np.linalg.norm(first, axis=1, keepdims=True), 1e-8)
    second = second - np.sum(first * second, axis=1, keepdims=True) * first
    second = second / np.maximum(np.linalg.norm(second, axis=1, keepdims=True), 1e-8)
    third = np.cross(first, second, axis=1)
    return np.stack([first, second, third], axis=-1)


def rotation_matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    """Convert rotation matrices to normalized xyzw quaternions."""
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.ndim != 3 or rotation.shape[1:] != (3, 3):
        raise ValueError(f"Expected rotation array [N,3,3], got {rotation.shape}")
    result = np.empty((len(rotation), 4), dtype=np.float64)
    for index, matrix in enumerate(rotation):
        trace = float(np.trace(matrix))
        if trace > 0.0:
            scale = math.sqrt(trace + 1.0) * 2.0
            x = (matrix[2, 1] - matrix[1, 2]) / scale
            y = (matrix[0, 2] - matrix[2, 0]) / scale
            z = (matrix[1, 0] - matrix[0, 1]) / scale
            w = 0.25 * scale
        elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
            scale = math.sqrt(max(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2], 0.0)) * 2.0
            x = 0.25 * scale
            y = (matrix[0, 1] + matrix[1, 0]) / scale
            z = (matrix[0, 2] + matrix[2, 0]) / scale
            w = (matrix[2, 1] - matrix[1, 2]) / scale
        elif matrix[1, 1] > matrix[2, 2]:
            scale = math.sqrt(max(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2], 0.0)) * 2.0
            x = (matrix[0, 1] + matrix[1, 0]) / scale
            y = 0.25 * scale
            z = (matrix[1, 2] + matrix[2, 1]) / scale
            w = (matrix[0, 2] - matrix[2, 0]) / scale
        else:
            scale = math.sqrt(max(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1], 0.0)) * 2.0
            x = (matrix[0, 2] + matrix[2, 0]) / scale
            y = (matrix[1, 2] + matrix[2, 1]) / scale
            z = 0.25 * scale
            w = (matrix[1, 0] - matrix[0, 1]) / scale
        result[index] = [x, y, z, w]
    norms = np.linalg.norm(result, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms < 1e-12):
        raise ValueError("Could not convert rotation matrices to finite quaternions")
    return result / norms[:, None]


def quaternion_angles_deg(quaternion: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion, dtype=np.float64)
    if quaternion.ndim != 2 or quaternion.shape[1] != 4:
        raise ValueError(f"Expected quaternion array [N,4], got {quaternion.shape}")
    if len(quaternion) < 2:
        return np.empty(0, dtype=np.float64)
    normalized = quaternion / np.linalg.norm(quaternion, axis=1, keepdims=True)
    dot = np.sum(normalized[:-1] * normalized[1:], axis=1)
    return np.degrees(2.0 * np.arccos(np.clip(np.abs(dot), 0.0, 1.0)))


def slerp_quaternions(left: np.ndarray, right: np.ndarray, fraction: np.ndarray) -> np.ndarray:
    """Interpolate matching xyzw quaternion rows along the shortest arc."""
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    fraction = np.asarray(fraction, dtype=np.float64)
    dot = np.sum(left * right, axis=1)
    right = np.where((dot < 0.0)[:, None], -right, right)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    theta = np.arccos(dot)
    sine = np.sin(theta)
    result = np.empty_like(left)
    small = sine < 1e-8
    result[small] = (
        (1.0 - fraction[small, None]) * left[small]
        + fraction[small, None] * right[small]
    )
    large = ~small
    if np.any(large):
        left_scale = np.sin((1.0 - fraction[large]) * theta[large]) / sine[large]
        right_scale = np.sin(fraction[large] * theta[large]) / sine[large]
        result[large] = left_scale[:, None] * left[large] + right_scale[:, None] * right[large]
    return result / np.linalg.norm(result, axis=1, keepdims=True)


def pose10_step_metrics(pose10: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return adjacent TCP translation in metres and SO(3) angle in degrees."""
    pose10 = np.asarray(pose10, dtype=np.float32)
    if pose10.ndim != 2 or pose10.shape[1] < 10:
        raise ValueError(f"Expected pose array [N,>=10], got {pose10.shape}")
    position_steps = np.linalg.norm(np.diff(pose10[:, :3].astype(np.float64), axis=0), axis=1)
    quaternion = rotation_matrix_to_quaternion(pose10_rotation_matrix(pose10))
    return position_steps, quaternion_angles_deg(quaternion)


def limit_pose10_steps(
    pose10: np.ndarray,
    *,
    position_limit_m: float,
    rotation_limit_deg: float,
    preserve_endpoint_frames: int = 1,
    max_sweeps: int = 3000,
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Project adjacent TCP steps under fixed translation and rotation limits.

    Position and orientation are constrained independently. The configured
    endpoint frames never move, and gripper values are copied exactly.
    """
    pose10 = np.asarray(pose10, dtype=np.float32)
    if pose10.ndim != 2 or pose10.shape[1] < 10:
        raise ValueError(f"Expected pose array [N,>=10], got {pose10.shape}")
    if not math.isfinite(position_limit_m) or position_limit_m <= 0:
        raise ValueError("position_limit_m must be finite and positive")
    if not math.isfinite(rotation_limit_deg) or not 0 < rotation_limit_deg <= 180:
        raise ValueError("rotation_limit_deg must be finite, positive, and at most 180")
    if preserve_endpoint_frames < 1:
        raise ValueError("preserve_endpoint_frames must be at least 1")
    if max_sweeps < 1:
        raise ValueError("max_sweeps must be at least 1")
    if len(pose10) < 2:
        return pose10.copy(), {
            "position_sweeps": 0,
            "rotation_sweeps": 0,
            "max_position_step_m": 0.0,
            "max_rotation_step_deg": 0.0,
        }

    preserve = min(preserve_endpoint_frames, len(pose10) // 2)
    fixed = np.zeros(len(pose10), dtype=bool)
    fixed[:preserve] = True
    fixed[-preserve:] = True

    position = pose10[:, :3].astype(np.float64).copy()
    position_sweeps = 0
    for sweep in range(max_sweeps):
        for parity in (0, 1):
            indices = np.arange(parity, len(position) - 1, 2)
            delta = position[indices + 1] - position[indices]
            distance = np.linalg.norm(delta, axis=1)
            active = distance > position_limit_m
            indices, delta, distance = indices[active], delta[active], distance[active]
            if not len(indices):
                continue
            both_fixed = fixed[indices] & fixed[indices + 1]
            if np.any(both_fixed):
                pair = int(indices[np.flatnonzero(both_fixed)[0]])
                raise ValueError(
                    f"Fixed endpoint frames {pair} and {pair + 1} exceed the position step limit"
                )
            correction = delta * ((distance - position_limit_m) / distance)[:, None]
            left_weight = np.where(fixed[indices], 0.0, np.where(fixed[indices + 1], 1.0, 0.5))
            right_weight = 1.0 - left_weight
            position[indices] += correction * left_weight[:, None]
            position[indices + 1] -= correction * right_weight[:, None]
        position_sweeps = sweep + 1
        if np.max(np.linalg.norm(np.diff(position, axis=0), axis=1)) <= position_limit_m + 1e-10:
            break
    else:
        raise RuntimeError(f"Position step limiting did not converge after {max_sweeps} sweeps")

    quaternion = rotation_matrix_to_quaternion(pose10_rotation_matrix(pose10))
    rotation_sweeps = 0
    for sweep in range(max_sweeps):
        for parity in (0, 1):
            indices = np.arange(parity, len(quaternion) - 1, 2)
            left = quaternion[indices].copy()
            right = quaternion[indices + 1].copy()
            dot = np.sum(left * right, axis=1)
            angle = np.degrees(2.0 * np.arccos(np.clip(np.abs(dot), 0.0, 1.0)))
            active = angle > rotation_limit_deg
            indices, left, right, angle = indices[active], left[active], right[active], angle[active]
            if not len(indices):
                continue
            both_fixed = fixed[indices] & fixed[indices + 1]
            if np.any(both_fixed):
                pair = int(indices[np.flatnonzero(both_fixed)[0]])
                raise ValueError(
                    f"Fixed endpoint frames {pair} and {pair + 1} exceed the rotation step limit"
                )
            excess_fraction = (angle - rotation_limit_deg) / angle
            left_weight = np.where(fixed[indices], 0.0, np.where(fixed[indices + 1], 1.0, 0.5))
            right_weight = 1.0 - left_weight
            quaternion[indices] = slerp_quaternions(left, right, excess_fraction * left_weight)
            quaternion[indices + 1] = slerp_quaternions(right, left, excess_fraction * right_weight)
        rotation_sweeps = sweep + 1
        if np.max(quaternion_angles_deg(quaternion), initial=0.0) <= rotation_limit_deg + 1e-8:
            break
    else:
        raise RuntimeError(f"Rotation step limiting did not converge after {max_sweeps} sweeps")

    rotation = quaternion_to_matrix(quaternion)
    result = pose10.copy()
    result[:, :3] = position.astype(np.float32)
    result[:, 3:9] = np.concatenate([rotation[:, :, 0], rotation[:, :, 1]], axis=1)
    result[:, 9] = pose10[:, 9]
    result[fixed] = pose10[fixed]
    position_steps, rotation_steps = pose10_step_metrics(result)
    if np.max(position_steps, initial=0.0) > position_limit_m + 1e-7:
        raise RuntimeError("Float32 position output exceeds the configured step limit")
    if np.max(rotation_steps, initial=0.0) > rotation_limit_deg + 1e-5:
        raise RuntimeError("Float32 rotation output exceeds the configured step limit")
    return result, {
        "position_sweeps": position_sweeps,
        "rotation_sweeps": rotation_sweeps,
        "max_position_step_m": float(np.max(position_steps, initial=0.0)),
        "max_rotation_step_deg": float(np.max(rotation_steps, initial=0.0)),
    }


def smooth_pose10(
    pose10: np.ndarray,
    *,
    window_size: int = 5,
    passes: int = 1,
    preserve_endpoint_frames: int = 2,
) -> np.ndarray:
    """Smooth TCP translation and rotation without changing frame count or gripper.

    Translation uses a centered binomial filter. Rotation matrices are averaged
    with the same weights and projected back onto SO(3) with SVD, so the output
    cannot contain invalid rot6d orientations. Gripper values are copied exactly.
    """
    pose10 = np.asarray(pose10, dtype=np.float32)
    if pose10.ndim != 2 or pose10.shape[1] < 10:
        raise ValueError(f"Expected pose array [N,>=10], got {pose10.shape}")
    if window_size < 3 or window_size % 2 == 0:
        raise ValueError("window_size must be an odd integer >= 3")
    if passes < 1:
        raise ValueError("passes must be >= 1")
    if preserve_endpoint_frames < 0:
        raise ValueError("preserve_endpoint_frames must be non-negative")
    if len(pose10) < window_size:
        return pose10.copy()

    kernel = np.asarray([math.comb(window_size - 1, index) for index in range(window_size)], dtype=np.float64)
    kernel /= np.sum(kernel)
    radius = window_size // 2
    position = pose10[:, :3].astype(np.float64)
    rotation = pose10_rotation_matrix(pose10).astype(np.float64)

    for _ in range(passes):
        padded_position = np.pad(position, ((radius, radius), (0, 0)), mode="edge")
        position = np.stack(
            [np.sum(padded_position[index : index + window_size] * kernel[:, None], axis=0) for index in range(len(position))]
        )
        padded_rotation = np.pad(rotation, ((radius, radius), (0, 0), (0, 0)), mode="edge")
        averaged = np.stack(
            [np.sum(padded_rotation[index : index + window_size] * kernel[:, None, None], axis=0) for index in range(len(rotation))]
        )
        left, _, right = np.linalg.svd(averaged)
        rotation = left @ right
        negative = np.linalg.det(rotation) < 0
        if np.any(negative):
            left[negative, :, -1] *= -1
            rotation[negative] = left[negative] @ right[negative]

    result = pose10.copy()
    result[:, :3] = position.astype(np.float32)
    result[:, 3:9] = np.concatenate([rotation[:, :, 0], rotation[:, :, 1]], axis=1).astype(np.float32)
    # Never blur grasp/release timing.
    result[:, 9] = pose10[:, 9]
    preserve = min(preserve_endpoint_frames, len(result) // 2)
    if preserve:
        result[:preserve] = pose10[:preserve]
        result[-preserve:] = pose10[-preserve:]
    if not np.all(np.isfinite(result)):
        raise ValueError("Smoothed pose contains NaN or infinity")
    return result


def sample_spatial_prefix(
    pose10: np.ndarray,
    *,
    prefix_frames: int,
    min_spacing_m: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Spatially sample the beginning of a trajectory while preserving its boundary.

    Frame zero and the first frame after the sampled prefix are retained. Inside
    the prefix, a frame is retained only after the TCP has moved at least
    ``min_spacing_m`` from the last retained frame. The final retained prefix
    point is removed when necessary to keep the same minimum distance to the
    boundary frame. Frames after the boundary are left untouched.
    """
    pose10 = np.asarray(pose10, dtype=np.float32)
    if pose10.ndim != 2 or pose10.shape[1] < 10:
        raise ValueError(f"Expected pose array [N,>=10], got {pose10.shape}")
    if prefix_frames < 1:
        raise ValueError("prefix_frames must be >= 1")
    if min_spacing_m <= 0:
        raise ValueError("min_spacing_m must be > 0")
    if len(pose10) <= 1:
        return np.arange(len(pose10), dtype=np.int64), {
            "prefix_input_frames": len(pose10),
            "prefix_kept_frames": len(pose10),
            "prefix_removed_indices": [],
            "prefix_min_resulting_step_m": 0.0,
        }

    position = pose10[:, :3].astype(np.float64)
    prefix_stop = min(prefix_frames, len(pose10))
    boundary = prefix_stop if prefix_stop < len(pose10) else len(pose10) - 1
    prefix_kept = [0]
    for index in range(1, boundary):
        if np.linalg.norm(position[index] - position[prefix_kept[-1]]) >= min_spacing_m:
            prefix_kept.append(index)

    # Avoid leaving a sub-threshold interval immediately before the untouched
    # part of the trajectory.
    while len(prefix_kept) > 1 and np.linalg.norm(position[boundary] - position[prefix_kept[-1]]) < min_spacing_m:
        prefix_kept.pop()
    if np.linalg.norm(position[boundary] - position[prefix_kept[-1]]) < min_spacing_m:
        raise ValueError(
            f"Cannot keep a {min_spacing_m:g} m TCP spacing across frames 0..{boundary}; "
            "the boundary is too close to frame 0"
        )

    keep = np.asarray(prefix_kept + list(range(boundary, len(pose10))), dtype=np.int64)
    sampled_region = set(range(prefix_stop))
    removed = sorted(sampled_region.difference(prefix_kept))
    steps = np.linalg.norm(np.diff(position[keep], axis=0), axis=1)
    checked_steps = steps[keep[:-1] < prefix_stop]
    minimum = float(np.min(checked_steps, initial=np.inf))
    if minimum + 1e-9 < min_spacing_m:
        raise RuntimeError(f"Prefix sampling produced a {minimum:g} m step below {min_spacing_m:g} m")
    return keep, {
        "prefix_input_frames": prefix_stop,
        "prefix_kept_frames": prefix_stop - len(removed),
        "prefix_removed_indices": removed,
        "prefix_min_resulting_step_m": minimum,
    }


def sparsify_dense_turns(
    pose10: np.ndarray,
    *,
    turn_angle_threshold_deg: float,
    turn_context_frames: int,
    min_turn_context_distance_m: float,
    dense_step_threshold_m: float,
    max_kept_step_m: float,
    max_position_error_m: float,
    max_rotation_error_deg: float,
    gripper_change_threshold_rad: float,
    max_consecutive_removed: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Remove only overly dense samples inside turns, with bounded SE(3) error.

    Straight regions are never candidates. The highest-curvature point in each
    turn is protected. A candidate is removed only if interpolation between the
    neighboring kept poses stays within position/rotation tolerances, gripper is
    effectively unchanged, and the resulting TCP step remains bounded.
    """
    pose10 = np.asarray(pose10, dtype=np.float32)
    if pose10.ndim != 2 or pose10.shape[1] < 10:
        raise ValueError(f"Expected pose array [N,>=10], got {pose10.shape}")
    if turn_context_frames < 1:
        raise ValueError("turn_context_frames must be >= 1")
    if max_consecutive_removed < 1:
        raise ValueError("max_consecutive_removed must be >= 1")
    thresholds = (
        turn_angle_threshold_deg,
        min_turn_context_distance_m,
        dense_step_threshold_m,
        max_kept_step_m,
        max_position_error_m,
        max_rotation_error_deg,
        gripper_change_threshold_rad,
    )
    if min(thresholds) < 0:
        raise ValueError("Turn sparsification thresholds must be non-negative")
    if len(pose10) < 2 * turn_context_frames + 1:
        return np.arange(len(pose10), dtype=np.int64), {
            "turn_frames": 0,
            "turn_regions": 0,
            "removed_indices": [],
            "protected_turn_indices": [],
        }

    position = pose10[:, :3].astype(np.float64)
    rotation = pose10_rotation_matrix(pose10).astype(np.float64)
    context = turn_context_frames
    turn_angles = np.zeros(len(pose10), dtype=np.float64)
    before = position[context:-context] - position[: -2 * context]
    after = position[2 * context :] - position[context:-context]
    before_norm = np.linalg.norm(before, axis=1)
    after_norm = np.linalg.norm(after, axis=1)
    valid = (before_norm >= min_turn_context_distance_m) & (after_norm >= min_turn_context_distance_m)
    cosine = np.ones(len(before), dtype=np.float64)
    cosine[valid] = np.sum(before[valid] * after[valid], axis=1) / (before_norm[valid] * after_norm[valid])
    turn_angles[context:-context] = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    turn_mask = turn_angles >= turn_angle_threshold_deg

    protected: list[int] = []
    region_count = 0
    index = 0
    while index < len(turn_mask):
        if not turn_mask[index]:
            index += 1
            continue
        end = index + 1
        while end < len(turn_mask) and turn_mask[end]:
            end += 1
        protected.append(index + int(np.argmax(turn_angles[index:end])))
        region_count += 1
        index = end

    adjacent_step = np.linalg.norm(np.diff(position, axis=0), axis=1)
    local_step = np.full(len(pose10), np.inf, dtype=np.float64)
    local_step[1:-1] = np.maximum(adjacent_step[:-1], adjacent_step[1:])
    candidate_mask = turn_mask & (local_step <= dense_step_threshold_m)
    candidate_mask[protected] = False
    candidates = np.flatnonzero(candidate_mask)
    candidates = candidates[np.argsort(local_step[candidates])]
    keep = np.ones(len(pose10), dtype=bool)
    removed: list[int] = []

    def projected_rotation(left: np.ndarray, right: np.ndarray, alpha: float) -> np.ndarray:
        average = (1.0 - alpha) * left + alpha * right
        u, _, vh = np.linalg.svd(average)
        result = u @ vh
        if np.linalg.det(result) < 0:
            u[:, -1] *= -1
            result = u @ vh
        return result

    for candidate in candidates:
        kept_before = np.flatnonzero(keep[:candidate])
        kept_after = np.flatnonzero(keep[candidate + 1 :])
        if len(kept_before) == 0 or len(kept_after) == 0:
            continue
        previous = int(kept_before[-1])
        following = int(candidate + 1 + kept_after[0])
        if following - previous - 1 > max_consecutive_removed:
            continue
        if np.linalg.norm(position[following] - position[previous]) > max_kept_step_m:
            continue
        interval = np.arange(previous + 1, following)
        if len(interval) == 0:
            continue
        if np.max(pose10[previous : following + 1, 9]) - np.min(pose10[previous : following + 1, 9]) > gripper_change_threshold_rad:
            continue
        alphas = (interval - previous) / (following - previous)
        interpolated_position = position[previous] + alphas[:, None] * (position[following] - position[previous])
        if np.max(np.linalg.norm(position[interval] - interpolated_position, axis=1)) > max_position_error_m:
            continue
        rotation_errors = []
        for frame, alpha in zip(interval, alphas, strict=True):
            estimate = projected_rotation(rotation[previous], rotation[following], float(alpha))
            relative = rotation[frame].T @ estimate
            value = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
            rotation_errors.append(float(np.degrees(np.arccos(value))))
        if max(rotation_errors, default=0.0) > max_rotation_error_deg:
            continue
        keep[candidate] = False
        removed.append(int(candidate))

    indices = np.flatnonzero(keep).astype(np.int64)
    details: dict[str, Any] = {
        "turn_frames": int(np.sum(turn_mask)),
        "turn_regions": region_count,
        "removed_indices": sorted(removed),
        "protected_turn_indices": protected,
        "max_turn_angle_deg": float(np.max(turn_angles, initial=0.0)),
    }
    return indices, details


def find_static_trim_bounds(
    pose10: np.ndarray,
    *,
    position_threshold_m: float,
    rotation_threshold_deg: float,
    gripper_threshold_rad: float,
    pre_roll_frames: int,
    post_roll_frames: int,
    min_episode_frames: int,
    min_active_frames: int = 1,
) -> tuple[int, int, dict[str, float | int]]:
    """Find ``[start, end)`` after removing long initial/final pose plateaus.

    The first active frame begins a sustained run in which every frame differs
    from the initial pose by at least one configured threshold. The last active
    frame is found symmetrically against the final pose. Requiring a sustained
    run prevents isolated tracking jitter from extending the retained interval.
    Small pre/post-roll regions are retained around those boundaries.
    """
    pose10 = np.asarray(pose10, dtype=np.float32)
    if pose10.ndim != 2 or pose10.shape[1] < 10:
        raise ValueError(f"Expected pose array [N,>=10], got {pose10.shape}")
    if len(pose10) < min_episode_frames:
        raise ValueError(f"Episode has {len(pose10)} frames, fewer than min_episode_frames={min_episode_frames}")
    if min(position_threshold_m, rotation_threshold_deg, gripper_threshold_rad) < 0:
        raise ValueError("Trim thresholds must be non-negative")
    if min(pre_roll_frames, post_roll_frames) < 0:
        raise ValueError("pre_roll_frames and post_roll_frames must be non-negative")
    if min_active_frames < 1:
        raise ValueError("min_active_frames must be >= 1")

    rotations = pose10_rotation_matrix(pose10)

    def differences(reference_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        position = np.linalg.norm(pose10[:, :3] - pose10[reference_index, :3], axis=1)
        relative = np.einsum("ji,njk->nik", rotations[reference_index], rotations)
        cosine = np.clip((np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
        rotation_deg = np.degrees(np.arccos(cosine))
        gripper = np.abs(pose10[:, 9] - pose10[reference_index, 9])
        return position, rotation_deg, gripper

    from_start = differences(0)
    from_end = differences(-1)
    active_from_start = (
        (from_start[0] > position_threshold_m)
        | (from_start[1] > rotation_threshold_deg)
        | (from_start[2] > gripper_threshold_rad)
    )
    active_from_end = (
        (from_end[0] > position_threshold_m)
        | (from_end[1] > rotation_threshold_deg)
        | (from_end[2] > gripper_threshold_rad)
    )
    def sustained_runs(active: np.ndarray) -> np.ndarray:
        if len(active) < min_active_frames:
            return np.empty(0, dtype=np.int64)
        counts = np.convolve(
            active.astype(np.int32),
            np.ones(min_active_frames, dtype=np.int32),
            mode="valid",
        )
        return np.flatnonzero(counts == min_active_frames)

    start_runs = sustained_runs(active_from_start)
    end_runs = sustained_runs(active_from_end)
    if len(start_runs) == 0 or len(end_runs) == 0:
        raise ValueError(
            "Episode never leaves its initial/final static plateau for "
            f"min_active_frames={min_active_frames} under the configured thresholds"
        )

    first_active = int(start_runs[0])
    last_active = int(end_runs[-1] + min_active_frames - 1)
    start = max(0, first_active - pre_roll_frames)
    end = min(len(pose10), last_active + 1 + post_roll_frames)
    if end <= start:
        raise ValueError(f"Invalid trim bounds [{start}, {end}) for {len(pose10)} frames")

    missing = min_episode_frames - (end - start)
    if missing > 0:
        grow_before = min(start, (missing + 1) // 2)
        start -= grow_before
        missing -= grow_before
        grow_after = min(len(pose10) - end, missing)
        end += grow_after
        missing -= grow_after
        if missing > 0:
            grow_before = min(start, missing)
            start -= grow_before
            missing -= grow_before
    if end - start < min_episode_frames:
        raise ValueError(
            f"Trimmed episode would contain {end - start} frames, fewer than min_episode_frames={min_episode_frames}"
        )

    details: dict[str, float | int] = {
        "original_length": len(pose10),
        "first_active_frame": first_active,
        "last_active_frame": last_active,
        "min_active_frames": min_active_frames,
        "start_frame": start,
        "end_frame": end,
        "trimmed_length": end - start,
        "trimmed_head_frames": start,
        "trimmed_tail_frames": len(pose10) - end,
    }
    return start, end, details


def array_stats(values: np.ndarray) -> dict[str, list[Any]]:
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError(f"Expected a non-empty [N,D] array for stats, got {values.shape}")
    return {
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "mean": np.mean(values, axis=0, dtype=np.float64).tolist(),
        "std": np.std(values, axis=0, dtype=np.float64).tolist(),
        "count": [int(values.shape[0])],
    }


def merge_stats(items: list[dict[str, list[Any]]]) -> dict[str, list[Any]]:
    if not items:
        raise ValueError("Cannot merge an empty stats list")
    counts = np.asarray([int(item["count"][0]) for item in items], dtype=np.float64)
    means = np.asarray([item["mean"] for item in items], dtype=np.float64)
    variances = np.square(np.asarray([item["std"] for item in items], dtype=np.float64))
    total = float(np.sum(counts))
    weights = counts.reshape((len(counts),) + (1,) * (means.ndim - 1))
    mean = np.sum(means * weights, axis=0) / total
    second_moment = np.sum((variances + means * means) * weights, axis=0) / total
    std = np.sqrt(np.maximum(second_moment - mean * mean, 0.0))
    return {
        "min": np.min(np.asarray([item["min"] for item in items]), axis=0).tolist(),
        "max": np.max(np.asarray([item["max"] for item in items]), axis=0).tolist(),
        "mean": mean.tolist(),
        "std": std.tolist(),
        "count": [int(total)],
    }


def assert_safe_output(config: dict[str, Any]) -> None:
    source = source_root(config)
    output = output_root(config)
    if output == source:
        raise ValueError("output.root must differ from source.root")
    if output == Path("/") or len(output.parts) < 4:
        raise ValueError(f"Refusing unsafe output path: {output}")


def rotation_quality(pose10: np.ndarray) -> tuple[float, float]:
    first = pose10[:, 3:6]
    second = pose10[:, 6:9]
    first = first / np.maximum(np.linalg.norm(first, axis=1, keepdims=True), 1e-12)
    second = second - np.sum(first * second, axis=1, keepdims=True) * first
    second = second / np.maximum(np.linalg.norm(second, axis=1, keepdims=True), 1e-12)
    third = np.cross(first, second)
    rotation = np.stack([first, second, third], axis=-1)
    identity = np.eye(3, dtype=np.float32)
    orthogonality_error = float(np.max(np.abs(np.swapaxes(rotation, 1, 2) @ rotation - identity)))
    determinant_error = float(np.max(np.abs(np.linalg.det(rotation) - 1.0)))
    return orthogonality_error, determinant_error


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))
