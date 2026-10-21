import numpy as np
import yaml

from pipeline_common import quaternion_to_matrix
from pipeline_common import apply_pose_transform
from pipeline_common import shifted_actions
from pipeline_common import find_static_trim_bounds
from pipeline_common import limit_pose10_steps
from pipeline_common import pose10_step_metrics
from pipeline_common import rotation_quality
from pipeline_common import sample_spatial_prefix
from pipeline_common import smooth_pose10
from pipeline_common import sparsify_dense_turns
from pipeline_common import ArmLayout
from pipeline_common import find_trajectory_trim_bounds
from pipeline_common import output_state_names
from pipeline_common import resolve_arm_layouts
from pipeline_common import sample_trajectory_prefix
from pipeline_common import source_state_to_trajectory
from pipeline_common import sparsify_trajectory_turns
from pipeline_common import step_limit_tolerances


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


def test_limit_pose10_steps_enforces_position_and_rotation_limits() -> None:
    nine = 9
    pose = _identity_pose(nine)
    pose[:, 0] = np.asarray([0.0, 0.05, 0.10, 0.60, 0.20, 0.25, 0.20, 0.10, 0.0])
    angles = np.radians([0.0, 5.0, 10.0, 70.0, 20.0, 15.0, 10.0, 5.0, 0.0])
    pose[:, 3] = np.cos(angles)
    pose[:, 4] = np.sin(angles)
    pose[:, 6] = -np.sin(angles)
    pose[:, 7] = np.cos(angles)
    pose[:, 9] = np.linspace(0.0, 1.0, nine)

    result, details = limit_pose10_steps(
        pose,
        position_limit_m=0.1,
        rotation_limit_deg=12.0,
        preserve_endpoint_frames=1,
    )

    position_steps, rotation_steps = pose10_step_metrics(result)
    assert np.max(position_steps) <= 0.1 + 1e-7
    assert np.max(rotation_steps) <= 12.0 + 1e-5
    np.testing.assert_array_equal(result[[0, -1]], pose[[0, -1]])
    np.testing.assert_array_equal(result[:, 9], pose[:, 9])
    assert details["position_sweeps"] > 0
    assert details["rotation_sweeps"] > 0


def test_limit_pose10_steps_rejects_violating_fixed_frames() -> None:
    pose = _identity_pose(2)
    pose[1, 0] = 1.0
    try:
        limit_pose10_steps(
            pose,
            position_limit_m=0.1,
            rotation_limit_deg=10.0,
            preserve_endpoint_frames=1,
        )
    except ValueError as error:
        assert "Fixed endpoint frames" in str(error)
    else:
        raise AssertionError("Expected an infeasible fixed endpoint pair to be rejected")


def test_limit_pose10_steps_accepts_sub_nanometre_convergence_residual() -> None:
    rng = np.random.default_rng(1)
    pose = _identity_pose(20)
    pose[:, :3] = np.cumsum(rng.normal(size=(20, 3)) * 0.015, axis=0)

    result, details = limit_pose10_steps(
        pose,
        position_limit_m=0.02,
        rotation_limit_deg=1.0,
        preserve_endpoint_frames=1,
        max_sweeps=14,
    )

    position_steps, _ = pose10_step_metrics(result)
    assert np.max(position_steps) <= 0.02 + 1e-7
    assert details["position_sweeps"] == 14


def test_limit_pose10_steps_uses_configured_position_tolerance() -> None:
    rng = np.random.default_rng(1)
    pose = _identity_pose(20)
    pose[:, :3] = np.cumsum(rng.normal(size=(20, 3)) * 0.015, axis=0)

    result, details = limit_pose10_steps(
        pose,
        position_limit_m=0.02,
        rotation_limit_deg=1.0,
        position_tolerance_m=1e-5,
        preserve_endpoint_frames=1,
        max_sweeps=14,
    )

    position_steps, _ = pose10_step_metrics(result)
    assert np.max(position_steps) <= 0.02 + 1e-5 + 5e-7
    assert details["position_sweeps"] < 14


def test_limit_pose10_steps_uses_configured_rotation_tolerance() -> None:
    pose = _identity_pose(20)
    angles_deg = np.linspace(0.0, 18.0, len(pose))
    angles_deg[1:-1:2] += 0.2
    angles = np.deg2rad(angles_deg)
    pose[:, 3] = np.cos(angles)
    pose[:, 4] = np.sin(angles)
    pose[:, 6] = -np.sin(angles)
    pose[:, 7] = np.cos(angles)

    result, details = limit_pose10_steps(
        pose,
        position_limit_m=0.02,
        rotation_limit_deg=1.0,
        rotation_tolerance_deg=0.01,
        preserve_endpoint_frames=1,
        max_sweeps=100,
    )

    _, rotation_steps = pose10_step_metrics(result)
    assert np.max(rotation_steps) <= 1.0 + 0.01 + 1e-5
    assert details["rotation_sweeps"] < 100


def test_step_limit_tolerances_convert_millimetres_to_metres() -> None:
    position_m, rotation_deg = step_limit_tolerances(
        {"step_limit": {"position_tolerance_mm": 0.01, "rotation_tolerance_deg": 0.02}}
    )

    assert position_m == 1e-5
    assert rotation_deg == 0.02


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


def _dual_pose(length: int) -> np.ndarray:
    return np.concatenate([_identity_pose(length), _identity_pose(length)], axis=1)


def test_resolve_dual_arm_layout_is_right_then_left(tmp_path) -> None:
    root = tmp_path / "dataset"
    (root / "assets").mkdir(parents=True)
    task = {
        "robot": {
            "arms": {
                "left": {"pose_source": "tcp_left", "gripper": "gripper_left", "mvs": "left_hand"},
                "right": {"pose_source": "tcp_right", "gripper": "gripper_right", "mvs": "right_hand"},
            },
            "mvs": {"right_hand": {"serial": "R"}, "left_hand": {"serial": "L"}},
        },
        "dataset": {"name": "test"},
    }
    (root / "assets" / "task.yaml").write_text(yaml.safe_dump(task), encoding="utf-8")
    names = [f"tcp_right_S.{x}" for x in ("x", "y", "z", "qx", "qy", "qz", "qw")] + ["gripper_right.width_m"]
    names += [f"tcp_left_S.{x}" for x in ("x", "y", "z", "qx", "qy", "qz", "qw")] + ["gripper_left.width_m"]
    info = {"features": {"observation.state": {"names": names}, "observation.images.mvs_R": {}, "observation.images.mvs_L": {}}}
    layouts = resolve_arm_layouts({"source": {"root": str(root)}}, info)
    assert [layout.role for layout in layouts] == ["right", "left"]
    assert [layout.output_video_key for layout in layouts] == [
        "observation.images.fisheye_img_right", "observation.images.fisheye_img_left"
    ]
    assert output_state_names(layouts)[0] == "right.x"
    assert output_state_names(layouts)[10] == "left.x"


def test_source_state_conversion_maps_each_arm_and_preserves_minus_one(tmp_path) -> None:
    calibration_dir = tmp_path / "assets" / "calibration"
    calibration_dir.mkdir(parents=True)
    calibration = {
        "distance": {"observed_min_m": 0.0, "observed_max_m": 0.1},
        "width_to_rad": {"min_rad": 0.0, "max_rad": 1.0},
    }
    for gripper in ("gripper_right", "gripper_left"):
        (calibration_dir / f"{gripper}.yaml").write_text(yaml.safe_dump(calibration), encoding="utf-8")
    layouts = [
        ArmLayout("right", "tcp_right", tuple(range(7)), "gripper_right", 7, "right", "R", "source_r", "output_r"),
        ArmLayout("left", "tcp_left", tuple(range(8, 15)), "gripper_left", 15, "left", "L", "source_l", "output_l"),
    ]
    state = np.zeros((1, 16), dtype=np.float32)
    state[0, [6, 14]] = 1.0
    state[0, 7] = 0.05
    state[0, 15] = -1.0
    config = {
        "source": {"root": str(tmp_path)},
        "trajectory": {
            "pose_transform": {"enabled": False},
            "gripper": {"mapping": "linear", "distance_section": "distance", "radians_section": "width_to_rad", "clip": True},
        },
    }
    result = source_state_to_trajectory(state, config, layouts)
    assert result.shape == (1, 20)
    np.testing.assert_allclose(result[0, 9], 0.5)
    np.testing.assert_allclose(result[0, 19], -1.0)


def test_dual_trim_combines_arm_activity_and_ignores_disabled_gripper() -> None:
    trajectory = _dual_pose(100)
    trajectory[:, [9, 19]] = -1.0
    trajectory[20:, 0] = 0.1
    trajectory[80:, 10] = 0.1
    start, end, _ = find_trajectory_trim_bounds(
        trajectory, position_threshold_m=0.01, rotation_threshold_deg=3.0,
        gripper_threshold_rad=0.1, pre_roll_frames=0, post_roll_frames=0,
        min_episode_frames=20, min_active_frames=3,
    )
    assert start == 20
    assert end == 80


def test_dual_trim_keeps_gripper_only_activity() -> None:
    trajectory = _dual_pose(100)
    trajectory[20:80, 19] = 0.5
    start, end, _ = find_trajectory_trim_bounds(
        trajectory, position_threshold_m=0.01, rotation_threshold_deg=3.0,
        gripper_threshold_rad=0.1, pre_roll_frames=0, post_roll_frames=0,
        min_episode_frames=20, min_active_frames=3,
    )
    assert start == 20
    assert end == 80


def test_joint_prefix_keeps_frame_when_either_arm_moves() -> None:
    trajectory = _dual_pose(8)
    trajectory[:, 10] = np.arange(8) * 0.0011
    keep, _ = sample_trajectory_prefix(trajectory, prefix_frames=6, min_spacing_m=0.001)
    np.testing.assert_array_equal(keep[:6], np.arange(6))


def test_joint_sparsification_static_arm_does_not_block_moving_turn() -> None:
    trajectory = _dual_pose(21)
    angle = np.linspace(0.0, np.pi / 2.0, len(trajectory))
    trajectory[:, 0] = 0.01 * np.cos(angle)
    trajectory[:, 1] = 0.01 * np.sin(angle)
    keep, details = sparsify_trajectory_turns(trajectory, **_turn_sparsification_kwargs())
    assert len(keep) < len(trajectory)
    assert details["removed_indices"]


def test_joint_sparsification_gripper_change_on_either_arm_protects_frames() -> None:
    trajectory = _dual_pose(21)
    angle = np.linspace(0.0, np.pi / 2.0, len(trajectory))
    trajectory[:, 0] = 0.01 * np.cos(angle)
    trajectory[:, 1] = 0.01 * np.sin(angle)
    trajectory[:, 19] = np.linspace(0.0, 1.0, len(trajectory))
    keep, _ = sparsify_trajectory_turns(trajectory, **_turn_sparsification_kwargs())
    np.testing.assert_array_equal(keep, np.arange(len(trajectory)))


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
