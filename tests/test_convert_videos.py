from __future__ import annotations

import importlib

import pytest


convert_videos = importlib.import_module("08_convert_videos")
resolve_crop_mode = convert_videos.resolve_crop_mode
video_filter = convert_videos.video_filter


@pytest.mark.parametrize(
    ("crop_enabled", "expected_mode", "expected_filter"),
    [
        (
            False,
            "stretch",
            "scale=240:240:flags=lanczos:force_original_aspect_ratio=disable,setsar=1",
        ),
        (
            True,
            "center_square",
            "crop='min(iw,ih)':'min(iw,ih)',scale=240:240:flags=lanczos:"
            "force_original_aspect_ratio=disable,setsar=1",
        ),
    ],
)
def test_auto_crop_mode_follows_archived_acquisition_config(
    crop_enabled: bool,
    expected_mode: str,
    expected_filter: str,
) -> None:
    config = {"video": {"width": 240, "height": 240, "crop_mode": "auto"}}
    task = {"robot": {"mvs": {"right_hand": {"crop": crop_enabled}}}}

    mode = resolve_crop_mode(config, task, "right_hand")

    assert mode == expected_mode
    assert video_filter(config, mode) == expected_filter


def test_explicit_crop_mode_overrides_acquisition_config() -> None:
    config = {"video": {"width": 240, "height": 240, "crop_mode": "stretch"}}
    task = {"robot": {"mvs": {"right_hand": {"crop": True}}}}

    assert resolve_crop_mode(config, task, "right_hand") == "stretch"


def test_auto_crop_mode_uses_camera_default_when_crop_is_omitted() -> None:
    config = {"video": {"width": 240, "height": 240, "crop_mode": "auto"}}
    task = {"robot": {"mvs": {"right_hand": {}}}}

    assert resolve_crop_mode(config, task, "right_hand") == "center_square"
