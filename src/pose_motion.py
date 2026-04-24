from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class MotionEstimate:
    vx_px_s: float
    vy_px_s: float
    speed_px_s: float


@dataclass
class TrajectoryPoint:
    x: int
    y: int


class PointMotionTracker:
    def __init__(self, history_size: int = 12, velocity_window: int = 4) -> None:
        self.history = deque(maxlen=max(2, history_size))
        self.velocity_window = max(2, velocity_window)

    def add_sample(self, timestamp_s: float, point: tuple[float, float] | None) -> None:
        if point is None:
            return
        self.history.append((timestamp_s, float(point[0]), float(point[1])))

    def estimate_velocity(self) -> MotionEstimate | None:
        if len(self.history) < 2:
            return None

        samples = list(self.history)[-self.velocity_window :]
        if len(samples) < 2:
            return None

        start_t, start_x, start_y = samples[0]
        end_t, end_x, end_y = samples[-1]
        dt = end_t - start_t
        if dt <= 1e-6:
            return None

        vx_px_s = (end_x - start_x) / dt
        vy_px_s = (end_y - start_y) / dt
        speed_px_s = float(np.hypot(vx_px_s, vy_px_s))
        return MotionEstimate(
            vx_px_s=float(vx_px_s),
            vy_px_s=float(vy_px_s),
            speed_px_s=speed_px_s,
        )


class MultiPointMotionTracker:
    def __init__(self, point_names: list[str], history_size: int = 12, velocity_window: int = 4) -> None:
        self.trackers = {
            point_name: PointMotionTracker(history_size=history_size, velocity_window=velocity_window)
            for point_name in point_names
        }

    def add_samples(self, timestamp_s: float, points: dict[str, tuple[float, float] | None]) -> None:
        for point_name, tracker in self.trackers.items():
            tracker.add_sample(timestamp_s, points.get(point_name))

    def estimate_velocity(self, point_name: str) -> MotionEstimate | None:
        tracker = self.trackers.get(point_name)
        if tracker is None:
            return None
        return tracker.estimate_velocity()


def midpoint(
    point_a: tuple[float, float] | None,
    point_b: tuple[float, float] | None,
) -> tuple[float, float] | None:
    if point_a is None or point_b is None:
        return None
    return (
        (float(point_a[0]) + float(point_b[0])) / 2.0,
        (float(point_a[1]) + float(point_b[1])) / 2.0,
    )


class TrajectoryBuffer:
    def __init__(self, maxlen: int = 90) -> None:
        self.points = deque(maxlen=max(2, maxlen))

    def add_point(self, point: tuple[float, float] | None) -> None:
        if point is None:
            return
        self.points.append(TrajectoryPoint(x=int(round(point[0])), y=int(round(point[1]))))

    def as_polyline(self) -> list[tuple[int, int]]:
        return [(point.x, point.y) for point in self.points]
