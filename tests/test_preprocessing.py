from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import yaml

from pipeline_common import load_config
from pipeline_common import source_root

merge_module = importlib.import_module("00_merge_datasets")
width_module = importlib.import_module("01_add_gripper_width")
ArmBinding = width_module.ArmBinding
add_or_overwrite = width_module.add_or_overwrite
output_names = width_module.output_names
resolve_bindings = width_module.resolve_bindings
discover_inputs = merge_module.discover_inputs
merge_datasets = merge_module.merge_datasets
read_json = merge_module.read_json
read_jsonl = merge_module.read_jsonl


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def stats(values: np.ndarray) -> dict:
    values = np.asarray(values)
    if values.ndim == 1:
        values = values[:, None]
    return {
        "min": values.min(axis=0).tolist(), "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(), "std": values.std(axis=0).tolist(),
        "count": [len(values)],
    }


def make_dataset(root: Path, state_rows: list[list[float]], *, fixed_state: bool, task: str = "test") -> None:
    names = [f"tcp_right_S.{name}" for name in ("x", "y", "z", "qx", "qy", "qz", "qw")]
    task_yaml = {
        "robot": {
            "arms": {"right": {"pose_source": "tcp_right", "gripper": "gripper_right", "mvs": "hand"}},
            "mvs": {"hand": {"serial": "R", "rotation": 0}},
        },
        "dataset": {"name": "test"},
        "gripper_detect_online": {"gripper_right": {"enabled": False, "roi_xywh": [0, 0, 1, 1]}},
    }
    (root / "assets").mkdir(parents=True)
    (root / "assets/task.yaml").write_text(yaml.safe_dump(task_yaml), encoding="utf-8")
    length = len(state_rows)
    feature = {"dtype": "float32", "shape": [7], "names": names}
    video_key = "observation.images.mvs_R"
    info = {
        "codebase_version": "v2.1", "robot_type": "test", "total_episodes": 1,
        "total_frames": length, "total_tasks": 1, "total_videos": 1,
        "total_chunks": 1, "chunks_size": 1000, "fps": 30, "splits": {"train": "0:1"},
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
        "features": {
            "observation.state": feature,
            video_key: {"dtype": "video", "shape": [1, 1, 3]},
            "timestamp": {"dtype": "float32", "shape": [1]},
            "frame_index": {"dtype": "int64", "shape": [1]},
            "episode_index": {"dtype": "int64", "shape": [1]},
            "index": {"dtype": "int64", "shape": [1]},
            "task_index": {"dtype": "int64", "shape": [1]},
        },
    }
    write_json(root / "meta/info.json", info)
    write_jsonl(root / "meta/tasks.jsonl", [{"task_index": 0, "task": task}])
    write_jsonl(root / "meta/episodes.jsonl", [{"episode_index": 0, "tasks": [task], "length": length}])
    state = np.asarray(state_rows, dtype=np.float32)
    episode_index = np.zeros(length, dtype=np.int64)
    index = np.arange(length, dtype=np.int64)
    episode_stats = {
        "observation.state": stats(state), "timestamp": stats(np.arange(length) / 30),
        "frame_index": stats(index), "episode_index": stats(episode_index),
        "index": stats(index), "task_index": stats(episode_index),
        video_key: {"min": [[[0]]], "max": [[[0]]], "mean": [[[0]]], "std": [[[0]]], "count": [length]},
    }
    write_jsonl(root / "meta/episodes_stats.jsonl", [{"episode_index": 0, "stats": episode_stats}])
    if fixed_state:
        state_column = pa.FixedSizeListArray.from_arrays(pa.array(state.ravel(), type=pa.float32()), 7)
    else:
        state_column = pa.array(state_rows, type=pa.list_(pa.float32()))
    table = pa.table({
        "observation.state": state_column,
        "timestamp": pa.array(np.arange(length) / 30, type=pa.float32()),
        "frame_index": pa.array(index), "episode_index": pa.array(episode_index),
        "index": pa.array(index), "task_index": pa.array(episode_index),
    })
    parquet = root / "data/chunk-000/episode_000000.parquet"
    parquet.parent.mkdir(parents=True)
    pq.write_table(table, parquet)
    video = root / f"videos/chunk-000/{video_key}/episode_000000.mp4"
    video.parent.mkdir(parents=True)
    video.write_bytes(b"video")


def test_discovery_only_accepts_exact_raw_collection_names(tmp_path: Path) -> None:
    accepted = ["task_20261019_123819", "task_20261019_130050"]
    rejected = ["task_20261019_merged", "task_20261019_123819_width", "other_20261019_141617"]
    for name in accepted + rejected:
        (tmp_path / name).mkdir()
    result = discover_inputs({"directory": str(tmp_path), "dataset_name": "task", "date": "20261019"})
    assert [path.name for path in result] == accepted


def test_merge_reindexes_and_normalizes_state_schema(tmp_path: Path) -> None:
    first, second = tmp_path / "task_20261019_100000", tmp_path / "task_20261019_110000"
    make_dataset(first, [[0, 1, 2, 3, 4, 5, 6], [1, 2, 3, 4, 5, 6, 7]], fixed_state=False)
    make_dataset(second, [[2, 3, 4, 5, 6, 7, 8]], fixed_state=True)
    (first / "assets/trajectory.rrd").write_bytes(b"first dataset trajectory")
    (second / "assets/trajectory.rrd").write_bytes(b"second dataset trajectory")
    output = tmp_path / "merged"
    merge_datasets([first, second], output)

    info = read_json(output / "meta/info.json")
    assert (info["total_episodes"], info["total_frames"], info["total_videos"]) == (2, 3, 2)
    assert [row["episode_index"] for row in read_jsonl(output / "meta/episodes.jsonl")] == [0, 1]
    second_table = pq.read_table(output / "data/chunk-000/episode_000001.parquet")
    assert pa.types.is_fixed_size_list(second_table["observation.state"].type)
    assert b"huggingface" in second_table.schema.metadata
    assert second_table["episode_index"].to_pylist() == [1]
    assert second_table["index"].to_pylist() == [2]
    assert (output / "videos/chunk-000/observation.images.mvs_R/episode_000001.mp4").read_bytes() == b"video"
    assert not (output / "assets/trajectory.rrd").exists()
    assert read_json(output / "meta/stats.json")["index"]["count"] == [3]


def test_merge_rejects_existing_output_before_processing(tmp_path: Path) -> None:
    output = tmp_path / "merged"
    output.mkdir()
    with pytest.raises(FileExistsError, match="overwrite is false"):
        merge_datasets([tmp_path / "missing_1", tmp_path / "missing_2"], output)


def test_add_and_overwrite_width_modes() -> None:
    binding = ArmBinding("right", tuple(range(7)), "gripper_right", "hand", "R", "video")
    names = [f"pose.{index}" for index in range(7)]
    state = np.arange(14, dtype=np.float32).reshape(2, 7)
    widths = {"gripper_right": np.asarray([0.1, 0.2], dtype=np.float32)}
    added = add_or_overwrite(state, widths, [binding], names, "add")
    assert output_names(names, [binding])[-1] == "gripper_right.width_m"
    np.testing.assert_allclose(added[:, -1], widths["gripper_right"])

    overwritten = add_or_overwrite(added, {"gripper_right": np.asarray([0.3, 0.4])}, [binding], output_names(names, [binding]), "overwrite")
    assert overwritten.shape == added.shape
    np.testing.assert_allclose(overwritten[:, -1], [0.3, 0.4])


def test_dual_arm_widths_follow_right_then_left_tcp() -> None:
    components = ("x", "y", "z", "qx", "qy", "qz", "qw")
    left_names = [f"tcp_left_L.{component}" for component in components]
    right_names = [f"tcp_right_R.{component}" for component in components]
    names = left_names + ["object_O.x"] + right_names
    config = {
        "robot": {
            "arms": {
                "left": {"pose_source": "tcp_left", "gripper": "gripper_left", "mvs": "left_hand"},
                "right": {"pose_source": "tcp_right", "gripper": "gripper_right", "mvs": "right_hand"},
            },
            "mvs": {
                "left_hand": {"serial": "L"},
                "right_hand": {"serial": "R"},
            },
        }
    }
    info = {"features": {"observation.state": {"names": names}}}
    bindings = resolve_bindings(config, info)
    state = np.arange(30, dtype=np.float32).reshape(2, 15)
    widths = {
        "gripper_right": np.asarray([100.0, 101.0], dtype=np.float32),
        "gripper_left": np.asarray([200.0, 201.0], dtype=np.float32),
    }

    result = add_or_overwrite(state, widths, bindings, names, "add")

    assert [binding.role for binding in bindings] == ["right", "left"]
    np.testing.assert_array_equal(result[:, :7], state[:, 8:15])
    np.testing.assert_array_equal(result[:, 7], widths["gripper_right"])
    np.testing.assert_array_equal(result[:, 8:15], state[:, :7])
    np.testing.assert_array_equal(result[:, 15], widths["gripper_left"])
    np.testing.assert_array_equal(result[:, 16], state[:, 7])


def test_dual_arms_cannot_share_pose_source() -> None:
    components = ("x", "y", "z", "qx", "qy", "qz", "qw")
    names = [f"tcp_shared_S.{component}" for component in components]
    config = {
        "robot": {
            "arms": {
                "right": {"pose_source": "tcp_shared", "gripper": "gripper_right", "mvs": "right_hand"},
                "left": {"pose_source": "tcp_shared", "gripper": "gripper_left", "mvs": "left_hand"},
            },
            "mvs": {
                "right_hand": {"serial": "R"},
                "left_hand": {"serial": "L"},
            },
        }
    }
    info = {"features": {"observation.state": {"names": names}}}

    with pytest.raises(ValueError, match="share one pose_source"):
        resolve_bindings(config, info)


def test_last_preprocessing_output_becomes_pipeline_source(tmp_path: Path) -> None:
    merged, width = tmp_path / "merged", tmp_path / "merged_width"
    config = {
        "preprocessing": {
            "merge": {"enabled": True, "inputs": ["a", "b"], "output": str(merged), "overwrite": False},
            "add_gripper_width": {"enabled": True, "mode": "overwrite", "overwrite": False},
        },
        "source": {"root": None},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    assert source_root(load_config(config_path)) == width.resolve()


def test_disabled_preprocessing_stages_exit_successfully(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({
            "preprocessing": {
                "merge": {"enabled": False},
                "add_gripper_width": {"enabled": False},
            }
        }),
        encoding="utf-8",
    )
    monkeypatch.setattr(sys, "argv", ["00_merge_datasets.py", "--config", str(config_path)])
    merge_module.main()
    monkeypatch.setattr(sys, "argv", ["01_add_gripper_width.py", "--config", str(config_path)])
    width_module.main()
    output = capsys.readouterr().out
    assert "Skipping dataset merge" in output
    assert "Skipping gripper width detection" in output


def test_width_config_uses_merge_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    merged, output = tmp_path / "merged", tmp_path / "merged_width"
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump({
            "preprocessing": {
                "merge": {"enabled": True, "output": str(merged)},
                "add_gripper_width": {
                    "enabled": True, "mode": "overwrite", "overwrite": False,
                },
            },
            "source": {"root": None},
        }),
        encoding="utf-8",
    )
    received = {}

    def fake_convert(source, target, **kwargs):
        received.update(source=source, target=target, **kwargs)

    monkeypatch.setattr(width_module, "convert", fake_convert)
    monkeypatch.setattr(sys, "argv", ["01_add_gripper_width.py", "--config", str(config_path)])
    width_module.main()
    assert received == {"source": merged, "target": output, "mode": "overwrite", "overwrite": False}
