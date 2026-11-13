from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path


def _trim_module():
    path = Path(__file__).resolve().parents[1] / "06_trim_static_segments.py"
    spec = spec_from_file_location("trim_static_segments", path)
    assert spec and spec.loader
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_short_episode_trim_plan_keeps_the_entire_episode() -> None:
    trim = _trim_module()

    start, end, details = trim.short_episode_trim_plan(31, 32)

    assert (start, end) == (0, 31)
    assert details == {
        "original_length": 31,
        "start_frame": 0,
        "end_frame": 31,
        "trimmed_length": 31,
        "trimmed_head_frames": 0,
        "trimmed_tail_frames": 0,
        "skipped": "short_episode",
        "minimum_episode_frames": 32,
    }
