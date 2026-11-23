from __future__ import annotations

from video_statistics import RgbStatistics
from video_statistics import sample_indices


def test_rgb_statistics_and_sampling() -> None:
    stats = RgbStatistics()
    stats.update([[[0, 128, 255]]])
    result = stats.result()

    assert sample_indices(5, 3) == [0, 2, 4]
    assert result["min"] == [[[0.0]], [[128 / 255]], [[1.0]]]
    assert result["max"] == result["min"]
    assert result["count"] == [1]
