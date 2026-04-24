from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

try:
    from .realtime_smooth import RealtimeTrajectorySmoother
except ImportError:  # pragma: no cover
    from realtime_smooth import RealtimeTrajectorySmoother


TRACK_COLOR = (0, 200, 0)
TRACK_OUTLINE_COLOR = (255, 255, 255)


def open_capture(source: str, width: int, height: int) -> cv2.VideoCapture:
    capture_source = int(source) if source.isdigit() else source
    capture = cv2.VideoCapture(capture_source)
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open camera/video source: {source}")
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return capture


def choose_ball_detection(
    result,
    ball_class_id: int,
    reference_point: tuple[float, float] | None = None,
    track_gate_px: float = 180.0,
) -> tuple[float | None, float | None, float | None, tuple[int, int, int, int] | None]:
    best_score = None
    best_conf = None
    best_center = None
    best_box = None

    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return None, None, None, None

    for box in boxes:
        class_id = int(box.cls.item())
        if class_id != ball_class_id:
            continue

        conf = float(box.conf.item())
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        score = conf
        if reference_point is not None:
            distance = float(np.hypot(cx - reference_point[0], cy - reference_point[1]))
            if distance > track_gate_px:
                continue
            score = conf - 0.0025 * distance
        if best_score is None or score > best_score:
            best_score = score
            best_conf = conf
            best_center = (cx, cy)
            best_box = (int(x1), int(y1), int(x2), int(y2))

    if best_center is None:
        return None, None, None, None
    return best_center[0], best_center[1], best_conf, best_box


class TrailRenderer:
    def __init__(self, thickness: int) -> None:
        self.thickness = thickness
        self.points: list[tuple[int, int]] = []

    def update(self, points: list[tuple[int, int]]) -> None:
        self.points = points


@dataclass
class BallTrackingConfig:
    track_conf: float = 0.12
    max_jump_px: float = 140.0
    ema_alpha: float = 0.45
    velocity_alpha: float = 0.35
    trail_length: int = 0
    max_missing_frames: int = 6
    trail_thickness: int = 7
    point_radius: int = 10


class BallTrackingOverlay:
    def __init__(self, config: BallTrackingConfig) -> None:
        self.config = config
        self.smoother = RealtimeTrajectorySmoother(
            conf_thresh=config.track_conf,
            max_jump_px=config.max_jump_px,
            ema_alpha=config.ema_alpha,
            velocity_alpha=config.velocity_alpha,
            max_history=config.trail_length,
            max_missing_frames=config.max_missing_frames,
        )
        self.trail_renderer = TrailRenderer(thickness=config.trail_thickness)

    def reference_point(self) -> tuple[float, float] | None:
        if self.smoother.state is not None and self.smoother.state.initialized:
            return (self.smoother.state.x, self.smoother.state.y)
        return None

    def update(self, cx: float | None, cy: float | None, conf: float | None) -> tuple[float | None, float | None]:
        return self.smoother.update(
            detected=cx is not None,
            conf=conf,
            cx=cx,
            cy=cy,
        )

    def draw(self, frame, smooth_x: float | None, smooth_y: float | None):
        history_points = [(int(x), int(y)) for x, y in self.smoother.history]
        self.trail_renderer.update(history_points)
        if len(self.trail_renderer.points) >= 2:
            polyline_points = np.array(self.trail_renderer.points, dtype=np.int32)
            cv2.polylines(
                frame,
                [polyline_points],
                isClosed=False,
                color=TRACK_OUTLINE_COLOR,
                thickness=self.config.trail_thickness + 2,
                lineType=cv2.LINE_8,
            )
            cv2.polylines(
                frame,
                [polyline_points],
                isClosed=False,
                color=TRACK_COLOR,
                thickness=self.config.trail_thickness,
                lineType=cv2.LINE_8,
            )
        if smooth_x is not None and smooth_y is not None:
            cv2.circle(
                frame,
                (int(smooth_x), int(smooth_y)),
                self.config.point_radius + 2,
                TRACK_OUTLINE_COLOR,
                -1,
            )
            cv2.circle(
                frame,
                (int(smooth_x), int(smooth_y)),
                self.config.point_radius,
                TRACK_COLOR,
                -1,
            )
        return frame
