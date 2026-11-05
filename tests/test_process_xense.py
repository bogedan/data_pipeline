from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


MODULE = importlib.import_module("09_process_xense")
VALIDATE = importlib.import_module("validate_openpi")


def test_depth_to_gray_rgb_uses_fixed_unorm8_mapping() -> None:
    depth = np.asarray([[0.0, 0.5, 1.0]], dtype=np.float32)

    rgb, summary = MODULE.depth_to_gray_rgb(depth, 0.0, 1.0, "error")

    assert rgb.shape == (1, 3, 3)
    np.testing.assert_array_equal(rgb[0, :, 0], [0, 128, 255])
    np.testing.assert_array_equal(rgb[..., 0], rgb[..., 1])
    np.testing.assert_array_equal(rgb[..., 1], rgb[..., 2])
    assert summary == {"min": 0.0, "max": 1.0, "below_range": 0, "above_range": 0}


def test_depth_to_gray_rgb_rejects_out_of_range_values() -> None:
    with pytest.raises(ValueError, match="above 1"):
        MODULE.depth_to_gray_rgb(np.asarray([[1.01]], dtype=np.float32), 0.0, 1.0, "error")


def test_validator_uses_each_video_features_dimensions() -> None:
    feature = {
        "dtype": "video",
        "shape": [700, 400, 3],
        "info": {"video.width": 400, "video.height": 700},
    }

    assert VALIDATE.expected_video_size(feature) == (400, 700)


def test_visualization_asset_switches_keep_numeric_encoding(tmp_path: Path) -> None:
    settings = {
        "depth": {"range": [0.0, 1.0]},
        "visualization_assets": {"save_colormaps": False, "save_preview_png": False},
    }

    hashes = MODULE.write_visualization_assets(tmp_path, settings)

    assert set(hashes) == {"depth_encoding.json"}
    assert not (tmp_path / "assets/xense/depth_colormaps.json").exists()
    assert not (tmp_path / "assets/xense/depth_colormaps_preview.png").exists()


class FakeCapture:
    def __init__(self, frames: list[np.ndarray]):
        self.frames = frames
        self.index = 0

    def isOpened(self) -> bool:
        return True

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame

    def release(self) -> None:
        pass


class FakeWriter:
    def __init__(self):
        self.frames: list[np.ndarray] = []
        self.closed = False

    def write(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())

    def close(self) -> None:
        self.closed = True

    def abort(self) -> None:
        raise AssertionError("writer must not abort")


def test_process_sensor_feeds_all_frames_and_emits_only_selected(monkeypatch, tmp_path: Path) -> None:
    frames = [np.full((2, 3, 3), value, dtype=np.uint8) for value in range(5)]
    capture = FakeCapture(frames)
    monkeypatch.setattr(MODULE.cv2, "VideoCapture", lambda _path: capture)
    writers: list[FakeWriter] = []

    def writer_factory(*_args, **_kwargs):
        writer = FakeWriter()
        writers.append(writer)
        return writer

    monkeypatch.setattr(MODULE, "create_video_writer", writer_factory)

    class Solver:
        def __init__(self):
            self.inputs: list[int] = []

        def selectSensorInfo(self, *_outputs, rectify_image):
            value = int(rectify_image[0, 0, 0])
            self.inputs.append(value)
            difference = np.dstack([
                np.full((2, 3), value + 10, dtype=np.uint8),
                np.full((2, 3), value + 20, dtype=np.uint8),
                np.full((2, 3), value + 30, dtype=np.uint8),
            ])
            depth = np.full((2, 3), value / 10.0, dtype=np.float32)
            force = np.full((35, 20, 3), value, dtype=np.float32)
            resultant = np.full(6, value, dtype=np.float32)
            return difference, depth, force, resultant

    output_type = SimpleNamespace(
        Difference="Difference", Depth="Depth", Force="Force", ForceResultant="ForceResultant"
    )
    sensor_type = SimpleNamespace(OutputType=output_type)
    sensor = MODULE.XenseSensor("finger", "OG000001", "source")
    settings = {
        "video": {},
        "difference": {"enabled": True, "crf": 18},
        "depth": {"enabled": True, "crf": 0, "range": [0.0, 1.0], "out_of_range": "error"},
        "force": {"enabled": True},
        "force_resultant": {"enabled": True},
    }
    solver = Solver()

    arrays, details = MODULE.process_sensor_episode(
        Sensor=sensor_type,
        solver=solver,
        sensor=sensor,
        video=tmp_path / "input.mp4",
        episode_index=0,
        source_length=5,
        selected_indices=[1, 4],
        difference_path=tmp_path / "difference.mp4",
        depth_path=tmp_path / "depth.mp4",
        fps=30,
        settings=settings,
    )

    assert solver.inputs == [0, 1, 2, 3, 4]
    assert len(writers) == 2
    assert [len(writer.frames) for writer in writers] == [2, 2]
    # Difference is returned by the SDK as BGR and written to ffmpeg as RGB.
    np.testing.assert_array_equal(writers[0].frames[0][0, 0], [31, 21, 11])
    np.testing.assert_array_equal(arrays[sensor.force_key][:, 0, 0, 0], [1, 4])
    np.testing.assert_array_equal(arrays[sensor.force_resultant_key][:, 0], [1, 4])
    assert details["depth"]["min"] == pytest.approx(0.1)
    assert details["depth"]["max"] == pytest.approx(0.4)
