"""Streaming RGB statistics stored alongside generated videos."""

from __future__ import annotations

import numpy as np


def sample_indices(frame_count: int, requested_frames: int) -> list[int]:
    if frame_count <= 0:
        raise ValueError("Video frame count must be positive")
    return np.linspace(0, frame_count - 1, min(max(requested_frames, 1), frame_count), dtype=np.int64).tolist()


class RgbStatistics:
    def __init__(self) -> None:
        self.minimum = np.full(3, np.inf, dtype=np.float64)
        self.maximum = np.full(3, -np.inf, dtype=np.float64)
        self.total = np.zeros(3, dtype=np.float64)
        self.total_square = np.zeros(3, dtype=np.float64)
        self.pixel_count = 0
        self.frames = 0

    def update(self, rgb: np.ndarray) -> None:
        pixels = np.asarray(rgb, dtype=np.float64).reshape(-1, 3) / 255.0
        self.minimum = np.minimum(self.minimum, np.min(pixels, axis=0))
        self.maximum = np.maximum(self.maximum, np.max(pixels, axis=0))
        self.total += np.sum(pixels, axis=0)
        self.total_square += np.sum(pixels * pixels, axis=0)
        self.pixel_count += len(pixels)
        self.frames += 1

    def result(self) -> dict[str, list]:
        if not self.pixel_count:
            raise ValueError("No video frames were sampled")
        mean = self.total / self.pixel_count
        std = np.sqrt(np.maximum(self.total_square / self.pixel_count - mean * mean, 0.0))
        wrap = lambda values: [[[float(value)]] for value in values]
        return {"min": wrap(self.minimum), "max": wrap(self.maximum), "mean": wrap(mean), "std": wrap(std), "count": [self.frames]}
