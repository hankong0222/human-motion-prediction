from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .data_io import load_detection_csv
except ImportError:  # pragma: no cover
    from data_io import load_detection_csv


def mark_invalid_points(df: pd.DataFrame, conf_thresh: float, max_step_px: float) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned["valid_raw"] = (cleaned["detected"] == 1) & cleaned["conf"].ge(conf_thresh)

    prev_x = None
    prev_y = None
    valid_mask: list[bool] = []

    for row in cleaned.itertuples(index=False):
        is_valid = bool(row.valid_raw) and pd.notna(row.cx) and pd.notna(row.cy)
        if is_valid and prev_x is not None and prev_y is not None:
            jump = float(np.hypot(row.cx - prev_x, row.cy - prev_y))
            if jump > max_step_px:
                is_valid = False
        valid_mask.append(is_valid)
        if is_valid:
            prev_x = row.cx
            prev_y = row.cy

    cleaned["valid_clean"] = valid_mask
    cleaned["cx_clean"] = cleaned["cx"].where(cleaned["valid_clean"])
    cleaned["cy_clean"] = cleaned["cy"].where(cleaned["valid_clean"])
    return cleaned


def interpolate_short_gaps(series: pd.Series, max_gap: int) -> pd.Series:
    return series.interpolate(limit=max_gap, limit_direction="both")


def split_trajectory(points: list[tuple[float, float]], max_distance: float) -> list[list[tuple[float, float]]]:
    if not points:
        return []

    segments: list[list[tuple[float, float]]] = [[points[0]]]
    for point in points[1:]:
        previous = segments[-1][-1]
        distance = float(np.hypot(point[0] - previous[0], point[1] - previous[1]))
        if distance <= max_distance:
            segments[-1].append(point)
        else:
            segments.append([point])
    return segments


def smooth_segment(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(points) < 3:
        return points

    t = np.arange(len(points), dtype=float)
    x = np.array([point[0] for point in points], dtype=float)
    y = np.array([point[1] for point in points], dtype=float)

    x_coeffs = np.polyfit(t, x, deg=1)
    y_coeffs = np.polyfit(t, y, deg=2 if len(points) >= 3 else 1)

    smooth_x = np.polyval(x_coeffs, t)
    smooth_y = np.polyval(y_coeffs, t)
    return list(zip(smooth_x.tolist(), smooth_y.tolist()))


def fit_trajectory(
    df: pd.DataFrame,
    rolling_window: int,
    max_gap: int,
    split_distance_px: float,
) -> pd.DataFrame:
    fitted = df.copy()

    fitted["cx_interp"] = interpolate_short_gaps(fitted["cx_clean"], max_gap=max_gap)
    fitted["cy_interp"] = interpolate_short_gaps(fitted["cy_clean"], max_gap=max_gap)

    fitted["cx_med"] = (
        fitted["cx_interp"].rolling(window=rolling_window, center=True, min_periods=1).median()
    )
    fitted["cy_med"] = (
        fitted["cy_interp"].rolling(window=rolling_window, center=True, min_periods=1).median()
    )

    valid_fit = fitted["cx_med"].notna() & fitted["cy_med"].notna()
    if valid_fit.sum() < 3:
        raise ValueError("Not enough valid points to fit a smooth curve.")

    points = list(
        zip(
            fitted.loc[valid_fit, "cx_med"].astype(float).tolist(),
            fitted.loc[valid_fit, "cy_med"].astype(float).tolist(),
        )
    )
    segments = split_trajectory(points, max_distance=split_distance_px)
    smoothed_points = [point for segment in segments for point in smooth_segment(segment)]

    smooth_df = fitted.loc[valid_fit, ["frame_idx", "timestamp_s"]].copy()
    smooth_df["cx_smooth"] = [point[0] for point in smoothed_points]
    smooth_df["cy_smooth"] = [point[1] for point in smoothed_points]

    fitted = fitted.merge(smooth_df, on=["frame_idx", "timestamp_s"], how="left")
    return fitted


def build_summary(df: pd.DataFrame) -> dict[str, float]:
    observed = int(df["detected"].sum())
    valid = int(df["valid_clean"].sum())
    interpolated = int(df["cx_interp"].notna().sum())
    smoothed = int(df["cx_smooth"].notna().sum())
    return {
        "frames_total": int(len(df)),
        "frames_detected": observed,
        "frames_clean_used": valid,
        "frames_after_interpolation": interpolated,
        "frames_smoothed": smoothed,
    }


def build_smooth_trajectory(
    input_path: Path,
    output_path: Path | None = None,
    conf_thresh: float = 0.15,
    max_step_px: float = 120.0,
    max_gap: int = 5,
    rolling_window: int = 5,
    split_distance_px: float = 60.0,
) -> Path:
    df = load_detection_csv(input_path)
    cleaned = mark_invalid_points(df, conf_thresh=conf_thresh, max_step_px=max_step_px)
    fitted = fit_trajectory(
        cleaned,
        rolling_window=max(1, rolling_window),
        max_gap=max(0, max_gap),
        split_distance_px=split_distance_px,
    )

    output = output_path or input_path.with_name(f"{input_path.stem}_smooth.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    fitted.to_csv(output, index=False)

    summary = build_summary(fitted)
    print(f"[DONE] Saved smooth trajectory CSV to {output}")
    for key, value in summary.items():
        print(f"[INFO] {key}={value}")
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a smooth 2D basketball trajectory from YOLO detection CSV output."
    )
    parser.add_argument("--input", type=Path, required=True, help="Detection CSV to smooth.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to <input_stem>_smooth.csv next to the input file.",
    )
    parser.add_argument("--conf-thresh", type=float, default=0.15, help="Minimum confidence retained.")
    parser.add_argument(
        "--max-step-px",
        type=float,
        default=120.0,
        help="Reject points whose frame-to-frame jump exceeds this threshold.",
    )
    parser.add_argument("--max-gap", type=int, default=5, help="Maximum missing run length to interpolate.")
    parser.add_argument(
        "--rolling-window",
        type=int,
        default=5,
        help="Centered rolling median window applied before segment smoothing.",
    )
    parser.add_argument(
        "--split-distance-px",
        type=float,
        default=60.0,
        help="Split trajectory into separate segments when adjacent points exceed this distance.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_smooth_trajectory(
        input_path=args.input,
        output_path=args.output,
        conf_thresh=args.conf_thresh,
        max_step_px=args.max_step_px,
        max_gap=args.max_gap,
        rolling_window=args.rolling_window,
        split_distance_px=args.split_distance_px,
    )


if __name__ == "__main__":
    main()
