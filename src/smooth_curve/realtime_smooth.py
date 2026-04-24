from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .data_io import load_detection_csv
except ImportError:  # pragma: no cover
    from data_io import load_detection_csv


@dataclass
class RealtimeState:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    initialized: bool = False


class RealtimeTrajectorySmoother:
    def __init__(
        self,
        conf_thresh: float = 0.15,
        max_jump_px: float = 120.0,
        ema_alpha: float = 0.45,
        velocity_alpha: float = 0.35,
        max_history: int = 30,
        max_missing_frames: int = 6,
    ) -> None:
        self.conf_thresh = conf_thresh
        self.max_jump_px = max_jump_px
        self.ema_alpha = ema_alpha
        self.velocity_alpha = velocity_alpha
        self.max_history = max_history
        self.max_missing_frames = max_missing_frames
        self.state: RealtimeState | None = None
        self.history: deque[tuple[float, float]] = deque(maxlen=max_history if max_history > 0 else None)
        self.missing_frames = 0

    def _accept_measurement(self, x: float, y: float) -> bool:
        if self.state is None or not self.state.initialized:
            return True
        jump = float(np.hypot(x - self.state.x, y - self.state.y))
        return jump <= self.max_jump_px

    def update(self, detected: bool, conf: float | None, cx: float | None, cy: float | None) -> tuple[float | None, float | None]:
        has_measurement = (
            detected
            and conf is not None
            and conf >= self.conf_thresh
            and cx is not None
            and cy is not None
        )

        if not has_measurement:
            if self.state is None or not self.state.initialized:
                return None, None
            self.missing_frames += 1
            if self.missing_frames > self.max_missing_frames:
                self.state = None
                self.history.clear()
                self.missing_frames = 0
                return None, None
            self.state.x = self.state.x + self.state.vx
            self.state.y = self.state.y + self.state.vy
            self.history.append((self.state.x, self.state.y))
            return self.state.x, self.state.y

        x = float(cx)
        y = float(cy)

        if not self._accept_measurement(x, y):
            if self.state is None or not self.state.initialized:
                return None, None
            self.missing_frames += 1
            if self.missing_frames > self.max_missing_frames:
                self.state = None
                self.history.clear()
                self.missing_frames = 0
                return None, None
            self.state.x = self.state.x + self.state.vx
            self.state.y = self.state.y + self.state.vy
            self.history.append((self.state.x, self.state.y))
            return self.state.x, self.state.y

        if self.state is None or not self.state.initialized:
            self.state = RealtimeState(x=x, y=y, initialized=True)
            self.missing_frames = 0
            self.history.append((x, y))
            return x, y

        predicted_x = self.state.x + self.state.vx
        predicted_y = self.state.y + self.state.vy

        smooth_x = self.ema_alpha * x + (1.0 - self.ema_alpha) * predicted_x
        smooth_y = self.ema_alpha * y + (1.0 - self.ema_alpha) * predicted_y

        measured_vx = smooth_x - self.state.x
        measured_vy = smooth_y - self.state.y

        self.state.vx = self.velocity_alpha * measured_vx + (1.0 - self.velocity_alpha) * self.state.vx
        self.state.vy = self.velocity_alpha * measured_vy + (1.0 - self.velocity_alpha) * self.state.vy
        self.state.x = smooth_x
        self.state.y = smooth_y
        self.state.initialized = True
        self.missing_frames = 0

        self.history.append((smooth_x, smooth_y))
        return smooth_x, smooth_y

def run_realtime_smoothing(
    input_path: Path,
    output_path: Path | None = None,
    conf_thresh: float = 0.15,
    max_jump_px: float = 120.0,
    ema_alpha: float = 0.45,
    velocity_alpha: float = 0.35,
    max_history: int = 30,
    max_missing_frames: int = 6,
) -> Path:
    df = load_detection_csv(input_path)
    smoother = RealtimeTrajectorySmoother(
        conf_thresh=conf_thresh,
        max_jump_px=max_jump_px,
        ema_alpha=ema_alpha,
        velocity_alpha=velocity_alpha,
        max_history=max_history,
        max_missing_frames=max_missing_frames,
    )

    smooth_x: list[float | None] = []
    smooth_y: list[float | None] = []
    history_size: list[int] = []

    for row in df.itertuples(index=False):
        x, y = smoother.update(
            detected=bool(row.detected),
            conf=None if pd.isna(row.conf) else float(row.conf),
            cx=None if pd.isna(row.cx) else float(row.cx),
            cy=None if pd.isna(row.cy) else float(row.cy),
        )
        smooth_x.append(x)
        smooth_y.append(y)
        history_size.append(len(smoother.history))

    result = df.copy()
    result["cx_realtime"] = smooth_x
    result["cy_realtime"] = smooth_y
    result["history_size"] = history_size

    output = output_path or input_path.with_name(f"{input_path.stem}_realtime.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"[DONE] Saved realtime trajectory CSV to {output}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run realtime-friendly smoothing on YOLO detection CSV output.")
    parser.add_argument("--input", type=Path, required=True, help="Detection CSV to smooth.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to <input_stem>_realtime.csv next to the input file.",
    )
    parser.add_argument("--conf-thresh", type=float, default=0.15, help="Minimum confidence retained.")
    parser.add_argument(
        "--max-jump-px",
        type=float,
        default=120.0,
        help="Ignore measurements whose jump from the current track exceeds this threshold.",
    )
    parser.add_argument("--ema-alpha", type=float, default=0.45, help="Measurement blending factor.")
    parser.add_argument("--velocity-alpha", type=float, default=0.35, help="Velocity update blending factor.")
    parser.add_argument("--max-history", type=int, default=30, help="Number of recent points kept in memory.")
    parser.add_argument(
        "--max-missing-frames",
        type=int,
        default=6,
        help="How many consecutive misses are tolerated before the active track is reset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_realtime_smoothing(
        input_path=args.input,
        output_path=args.output,
        conf_thresh=args.conf_thresh,
        max_jump_px=args.max_jump_px,
        ema_alpha=args.ema_alpha,
        velocity_alpha=args.velocity_alpha,
        max_history=args.max_history,
        max_missing_frames=args.max_missing_frames,
    )


if __name__ == "__main__":
    main()
