from __future__ import annotations

import argparse
import time
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

from pose_motion import TrajectoryBuffer, midpoint
from smooth_curve.ball_tracking import BallTrackingConfig, BallTrackingOverlay, choose_ball_detection

# User numbering: 12 = left hip, 13 = right hip.
# Ultralytics COCO17 keypoints are zero-based, so these map to 11 and 12.
LEFT_HIP_INDEX = 11
RIGHT_HIP_INDEX = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Draw a motion trajectory using the midpoint between left and right hip keypoints.",
    )
    parser.add_argument("--video", required=True, help="Input video path.")
    parser.add_argument(
        "--model",
        default="yolo26n-pose.pt",
        help="YOLO pose model path. Default: yolo26n-pose.pt",
    )
    parser.add_argument(
        "--ball-model",
        default="exp.pt",
        help="Basketball detection model path. Default: exp.pt",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output video path. Default: outputs/<input>_hip_trajectory.mp4",
    )
    parser.add_argument("--conf", type=float, default=0.15, help="Detection confidence threshold.")
    parser.add_argument(
        "--kpt-conf",
        type=float,
        default=0.20,
        help="Minimum confidence required for left/right hip keypoints.",
    )
    parser.add_argument(
        "--trajectory-length",
        type=int,
        default=240,
        help="How many midpoint samples to keep in the displayed trajectory.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=7,
        help="Moving-average window size for smoothing each tracked trajectory.",
    )
    parser.add_argument("--line-thickness", type=int, default=3, help="Trajectory line thickness.")
    parser.add_argument("--ball-class-id", type=int, default=0, help="Basketball class id in the detection model.")
    parser.add_argument("--ball-detect-conf", type=float, default=0.05, help="Basketball detection confidence threshold.")
    parser.add_argument("--ball-track-conf", type=float, default=0.12, help="Minimum confidence used by the basketball smoother.")
    parser.add_argument("--ball-trail-length", type=int, default=0, help="Basketball trail history. 0 keeps full history.")
    parser.add_argument("--ball-trail-thickness", type=int, default=7, help="Basketball trail thickness.")
    parser.add_argument("--ball-point-radius", type=int, default=10, help="Basketball point radius.")
    parser.add_argument("--ball-track-gate-px", type=float, default=180.0, help="Basketball detection gating distance.")
    parser.add_argument("--ball-max-jump-px", type=float, default=140.0, help="Maximum accepted basketball jump between smoothed points.")
    parser.add_argument("--ball-max-missing-frames", type=int, default=6, help="How many consecutive basketball misses are tolerated before reset.")
    parser.add_argument("--ball-ema-alpha", type=float, default=0.45, help="Basketball smoother measurement blending factor.")
    parser.add_argument("--ball-velocity-alpha", type=float, default=0.35, help="Basketball smoother velocity blending factor.")
    parser.add_argument(
        "--no-skeleton",
        action="store_true",
        help="Disable drawing the YOLO pose skeleton overlay.",
    )
    return parser.parse_args()


def ensure_output_path(video_path: Path, output_path: str | None) -> Path:
    if output_path is not None:
        return Path(output_path)
    output_dir = video_path.parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{video_path.stem}_hip_trajectory.avi"


def open_video_writer(output_path: Path, fps: float, width: int, height: int) -> cv2.VideoWriter:
    suffix = output_path.suffix.lower()
    codec_candidates = {
        ".avi": ["XVID", "MJPG"],
        ".mp4": ["mp4v", "avc1"],
    }.get(suffix, ["XVID", "MJPG", "mp4v"])

    for codec in codec_candidates:
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (width, height),
        )
        if writer.isOpened():
            return writer
        writer.release()

    raise RuntimeError(
        f"Unable to open output video for writing: {output_path}. "
        f"Tried codecs: {', '.join(codec_candidates)}"
    )


def extract_point(
    keypoints_xy: np.ndarray,
    keypoints_conf: np.ndarray | None,
    person_index: int,
    keypoint_index: int,
    min_conf: float,
) -> tuple[float, float] | None:
    if person_index >= len(keypoints_xy):
        return None
    if keypoint_index >= keypoints_xy.shape[1]:
        return None

    if keypoints_conf is not None:
        if person_index >= len(keypoints_conf):
            return None
        conf = float(keypoints_conf[person_index, keypoint_index])
        if conf < min_conf:
            return None

    point = keypoints_xy[person_index, keypoint_index]
    return (float(point[0]), float(point[1]))


def track_color(track_id: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(track_id + 17)
    color = rng.integers(80, 256, size=3)
    return (int(color[0]), int(color[1]), int(color[2]))


def smooth_point(history: deque[tuple[float, float]]) -> tuple[float, float] | None:
    if not history:
        return None
    samples = np.array(history, dtype=np.float32)
    mean_xy = samples.mean(axis=0)
    return (float(mean_xy[0]), float(mean_xy[1]))


def draw_trajectory(
    frame: np.ndarray,
    trajectory: list[tuple[int, int]],
    line_thickness: int,
    color: tuple[int, int, int],
) -> None:
    if len(trajectory) >= 2:
        cv2.polylines(
            frame,
            [np.array(trajectory, dtype=np.int32)],
            False,
            color,
            thickness=line_thickness,
            lineType=cv2.LINE_AA,
        )
    if trajectory:
        cv2.circle(frame, trajectory[-1], 6, color, -1, lineType=cv2.LINE_AA)


def draw_label(frame: np.ndarray, tracked_count: int) -> None:
    cv2.putText(
        frame,
        f"hip midpoint trajectories | tracked={tracked_count}",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def draw_pose_overlay(frame: np.ndarray, result, draw_skeleton: bool) -> np.ndarray:
    if not draw_skeleton:
        return frame
    return result.plot(
        img=frame,
        boxes=False,
        labels=False,
        probs=False,
        conf=False,
    )


def main() -> None:
    args = parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"Input video not found: {video_path}")

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    ball_model_path = Path(args.ball_model)
    if not ball_model_path.exists():
        raise FileNotFoundError(f"Ball model not found: {ball_model_path}")

    output_path = ensure_output_path(video_path, args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))
    ball_model = YOLO(str(ball_model_path))
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    writer = open_video_writer(output_path, fps, width, height)

    trajectory_buffers: dict[int, TrajectoryBuffer] = defaultdict(
        lambda: TrajectoryBuffer(maxlen=args.trajectory_length)
    )
    ball_tracking = BallTrackingOverlay(
        BallTrackingConfig(
            track_conf=args.ball_track_conf,
            max_jump_px=args.ball_max_jump_px,
            ema_alpha=args.ball_ema_alpha,
            velocity_alpha=args.ball_velocity_alpha,
            trail_length=args.ball_trail_length,
            max_missing_frames=args.ball_max_missing_frames,
            trail_thickness=args.ball_trail_thickness,
            point_radius=args.ball_point_radius,
        )
    )
    raw_point_histories: dict[int, deque[tuple[float, float]]] = defaultdict(
        lambda: deque(maxlen=max(2, args.smooth_window))
    )
    processed_frames = 0
    start_time = time.perf_counter()

    try:
        with tqdm(
            total=total_frames if total_frames > 0 else None,
            desc="Rendering video",
            unit="frame",
        ) as progress:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break

                ball_result = ball_model.predict(frame, conf=args.ball_detect_conf, verbose=False)[0]
                result = model.track(frame, conf=args.conf, persist=True, verbose=False)[0]
                annotated_frame = draw_pose_overlay(frame, result, draw_skeleton=not args.no_skeleton)
                tracked_count = 0

                ball_cx, ball_cy, ball_conf, ball_box = choose_ball_detection(
                    ball_result,
                    ball_class_id=args.ball_class_id,
                    reference_point=ball_tracking.reference_point(),
                    track_gate_px=args.ball_track_gate_px,
                )
                smooth_ball_x, smooth_ball_y = ball_tracking.update(ball_cx, ball_cy, ball_conf)
                annotated_frame = ball_tracking.draw(annotated_frame, smooth_x=smooth_ball_x, smooth_y=smooth_ball_y)
                if ball_box is not None and ball_conf is not None:
                    x1, y1, x2, y2 = ball_box
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 200, 0), 2, cv2.LINE_AA)
                    cv2.putText(
                        annotated_frame,
                        f"ball {ball_conf:.2f}",
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 200, 0),
                        2,
                        cv2.LINE_AA,
                    )

                if (
                    result.boxes is not None
                    and result.boxes.id is not None
                    and result.keypoints is not None
                    and result.keypoints.xy is not None
                ):
                    track_ids = result.boxes.id.detach().cpu().numpy().astype(int)
                    keypoints_xy = result.keypoints.xy.detach().cpu().numpy()
                    keypoints_conf = None
                    if result.keypoints.conf is not None:
                        keypoints_conf = result.keypoints.conf.detach().cpu().numpy()

                    for person_index, track_id in enumerate(track_ids):
                        left_hip = extract_point(
                            keypoints_xy,
                            keypoints_conf,
                            person_index,
                            LEFT_HIP_INDEX,
                            args.kpt_conf,
                        )
                        right_hip = extract_point(
                            keypoints_xy,
                            keypoints_conf,
                            person_index,
                            RIGHT_HIP_INDEX,
                            args.kpt_conf,
                        )
                        hip_center = midpoint(left_hip, right_hip)
                        if hip_center is None:
                            continue

                        raw_point_histories[track_id].append(hip_center)
                        smoothed_center = smooth_point(raw_point_histories[track_id])
                        if smoothed_center is None:
                            continue

                        tracked_count += 1
                        trajectory_buffers[track_id].add_point(smoothed_center)
                        color = track_color(track_id)
                        center_xy = (int(round(smoothed_center[0])), int(round(smoothed_center[1])))
                        draw_trajectory(
                            annotated_frame,
                            trajectory_buffers[track_id].as_polyline(),
                            args.line_thickness,
                            color,
                        )
                        cv2.putText(
                            annotated_frame,
                            f"id={track_id}",
                            (center_xy[0] + 8, center_xy[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            2,
                            cv2.LINE_AA,
                        )

                draw_label(annotated_frame, tracked_count)
                writer.write(annotated_frame)
                processed_frames += 1
                progress.update(1)
    finally:
        capture.release()
        writer.release()

    elapsed_s = time.perf_counter() - start_time
    fps_processed = processed_frames / elapsed_s if elapsed_s > 1e-6 else 0.0
    print(f"Processed {processed_frames} frames in {elapsed_s:.2f}s ({fps_processed:.2f} frames/s)")
    print(f"Saved trajectory video to: {output_path}")


if __name__ == "__main__":
    main()
