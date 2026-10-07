import numpy as np

from pipeline_common import quaternion_to_matrix
from pipeline_common import apply_pose_transform
from pipeline_common import shifted_actions
from pipeline_common import find_static_trim_bounds
from pipeline_common import rotation_quality
from pipeline_common import sample_spatial_prefix
from pipeline_common import smooth_pose10
from pipeline_common import sparsify_dense_turns


def test_identity_quaternion_uses_matrix_columns() -> None:
    matrix = quaternion_to_matrix(np.asarray([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32))
    rot6d = np.concatenate([matrix[:, :, 0], matrix[:, :, 1]], axis=1)
    np.testing.assert_allclose(rot6d, [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]], atol=1e-6)


def test_action_shift_repeats_tail() -> None:
    pose = np.arange(30, dtype=np.float32).reshape(3, 10)
    action = shifted_actions(pose, offset=1, tail_policy="repeat_last")
    np.testing.assert_array_equal(action[0], pose[1])
    np.testing.assert_array_equal(action[1], pose[2])
    np.testing.assert_array_equal(action[2], pose[2])


def test_disabled_pose_transform_is_identity() -> None:
    position = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
    rotation = np.eye(3, dtype=np.float32)[None]
    result_position, result_rotation = apply_pose_transform(position, rotation, {"enabled": False})
    np.testing.assert_array_equal(result_position, position)
    np.testing.assert_array_equal(result_rotation, rotation)


def test_find_static_trim_bounds_keeps_roll_and_minimum_length() -> None:
    pose = np.zeros((100, 10), dtype=np.float32)
    pose[:, 3] = 1.0
    pose[:, 7] = 1.0
    pose[20:80, 0] = np.linspace(0.0, 0.1, 60)
    pose[80:, 0] = 0.1
    start, end, details = find_static_trim_bounds(
        pose,
        position_threshold_m=0.005,
        rotation_threshold_deg=3.0,
        gripper_threshold_rad=0.1,
        pre_roll_frames=3,
        post_roll_frames=3,
        min_episode_frames=32,
    )
    assert start == 20
    assert end == 80
    assert details["trimmed_head_frames"] == 20
    assert details["trimmed_tail_frames"] == 20


def test_find_static_trim_bounds_rejects_all_static_episode() -> None:
    pose = np.zeros((40, 10), dtype=np.float32)
    pose[:, 3] = 1.0
    pose[:, 7] = 1.0
    try:
        find_static_trim_bounds(
            pose,
            position_threshold_m=0.005,
            rotation_threshold_deg=3.0,
            gripper_threshold_rad=0.1,
            pre_roll_frames=3,
            post_roll_frames=3,
            min_episode_frames=32,
        )
    except ValueError as error:
        assert "never leaves" in str(error)
    else:
        raise AssertionError("Expected an all-static episode to be rejected")


def test_find_static_trim_bounds_ignores_short_jitter_bursts() -> None:
    pose = np.zeros((100, 10), dtype=np.float32)
    pose[:, 3] = 1.0
    pose[:, 7] = 1.0
    pose[10, 0] = 0.006  # One-frame jitter relative to the initial plateau.
    pose[30:70, 0] = 0.1
    pose[70:, 0] = 0.2
    pose[90, 0] = 0.194  # One-frame jitter relative to the final plateau.

    start, end, details = find_static_trim_bounds(
        pose,
        position_threshold_m=0.005,
        rotation_threshold_deg=3.0,
        gripper_threshold_rad=0.1,
        pre_roll_frames=0,
        post_roll_frames=0,
        min_episode_frames=32,
        min_active_frames=5,
    )

    assert start == 30
    assert end == 70
    assert details["min_active_frames"] == 5


def test_smooth_pose10_preserves_shape_endpoints_and_gripper() -> None:
    pose = np.zeros((21, 10), dtype=np.float32)
    pose[:, 3] = 1.0
    pose[:, 7] = 1.0
    pose[:, 0] = np.linspace(0.0, 0.1, len(pose))
    pose[10, 1] = 0.01
    pose[:, 9] = np.linspace(0.0, 1.0, len(pose))
    result = smooth_pose10(pose, window_size=5, passes=1, preserve_endpoint_frames=2)
    assert result.shape == pose.shape
    np.testing.assert_array_equal(result[:2], pose[:2])
    np.testing.assert_array_equal(result[-2:], pose[-2:])
    np.testing.assert_array_equal(result[:, 9], pose[:, 9])
    assert abs(result[10, 1]) < abs(pose[10, 1])


def test_smooth_pose10_outputs_valid_rotations() -> None:
    pose = np.zeros((21, 10), dtype=np.float32)
    angles = np.linspace(-0.3, 0.3, len(pose))
    pose[:, 3] = np.cos(angles)
    pose[:, 4] = np.sin(angles)
    pose[:, 6] = -np.sin(angles)
    pose[:, 7] = np.cos(angles)
    pose[10, 3:9] += 0.02
    result = smooth_pose10(pose, window_size=5, passes=1, preserve_endpoint_frames=2)
    orthogonality, determinant = rotation_quality(result)
    assert orthogonality < 1e-5
    assert determinant < 1e-5


def _turn_sparsification_kwargs() -> dict:
    return {
        "turn_angle_threshold_deg": 3.0,
        "turn_context_frames": 2,
        "min_turn_context_distance_m": 0.0005,
        "dense_step_threshold_m": 0.0015,
        "max_kept_step_m": 0.003,
        "max_position_error_m": 0.0005,
        "max_rotation_error_deg": 0.3,
        "gripper_change_threshold_rad": 0.02,
        "max_consecutive_removed": 1,
    }


def _identity_pose(length: int) -> np.ndarray:
    pose = np.zeros((length, 10), dtype=np.float32)
    pose[:, 3] = 1.0
    pose[:, 7] = 1.0
    return pose


def test_sample_spatial_prefix_enforces_one_mm_and_keeps_suffix() -> None:
    pose = _identity_pose(80)
    pose[:, 0] = np.arange(len(pose)) * 0.00025
    keep, details = sample_spatial_prefix(pose, prefix_frames=50, min_spacing_m=0.001)
    prefix_pairs = np.linalg.norm(np.diff(pose[keep, :3], axis=0), axis=1)[keep[:-1] < 50]
    assert np.min(prefix_pairs) >= 0.001 - 1e-9
    np.testing.assert_array_equal(keep[keep >= 50], np.arange(50, len(pose)))
    assert details["prefix_removed_indices"]


def test_sample_spatial_prefix_repairs_short_boundary_gap() -> None:
    pose = _identity_pose(60)
    pose[:, 0] = np.arange(len(pose)) * 0.0006
    keep, _ = sample_spatial_prefix(pose, prefix_frames=50, min_spacing_m=0.001)
    boundary_position = int(np.flatnonzero(keep == 50)[0])
    previous = keep[boundary_position - 1]
    assert np.linalg.norm(pose[50, :3] - pose[previous, :3]) >= 0.001 - 1e-9


def test_turn_sparsification_never_removes_straight_motion() -> None:
    pose = _identity_pose(30)
    pose[:, 0] = np.arange(len(pose)) * 0.0005
    keep, details = sparsify_dense_turns(pose, **_turn_sparsification_kwargs())
    np.testing.assert_array_equal(keep, np.arange(len(pose)))
    assert details["removed_indices"] == []


def test_turn_sparsification_reduces_dense_curve_with_bounded_gap() -> None:
    pose = _identity_pose(21)
    angle = np.linspace(0.0, np.pi / 2.0, len(pose))
    pose[:, 0] = 0.01 * np.cos(angle)
    pose[:, 1] = 0.01 * np.sin(angle)
    keep, details = sparsify_dense_turns(pose, **_turn_sparsification_kwargs())
    assert 0 < len(details["removed_indices"]) < len(pose) - 2
    assert keep[0] == 0 and keep[-1] == len(pose) - 1
    resulting_step = np.linalg.norm(np.diff(pose[keep, :3], axis=0), axis=1)
    assert np.max(resulting_step) <= 0.003


def test_turn_sparsification_preserves_gripper_change() -> None:
    pose = _identity_pose(21)
    angle = np.linspace(0.0, np.pi / 2.0, len(pose))
    pose[:, 0] = 0.01 * np.cos(angle)
    pose[:, 1] = 0.01 * np.sin(angle)
    pose[:, 9] = np.linspace(0.0, 1.0, len(pose))
    keep, details = sparsify_dense_turns(pose, **_turn_sparsification_kwargs())
    np.testing.assert_array_equal(keep, np.arange(len(pose)))
    assert details["removed_indices"] == []
