from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import pandas as pd
from ultralytics import YOLO

try:
    from ..database import SQLiteRunWriter
    from .ball_tracking import (
        BallTrackingConfig,
        BallTrackingOverlay,
        choose_ball_detection,
        open_capture,
    )
    from .cli_args import add_ball_tracking_args, add_camera_args
    from .pose_utils import ball_near_wrists, extract_debug_landmarks, extract_pose_record, nearest_wrist_distance
    from ..prediction.release_prediction import ReleaseTrajectoryPredictor
except ImportError:  # pragma: no cover
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from database import SQLiteRunWriter
    from ball_tracking import BallTrackingConfig, BallTrackingOverlay, choose_ball_detection, open_capture
    from cli_args import add_ball_tracking_args, add_camera_args
    from pose_utils import ball_near_wrists, extract_debug_landmarks, extract_pose_record, nearest_wrist_distance
    from prediction.release_prediction import ReleaseTrajectoryPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run basketball detection and human pose estimation on the same live camera feed."
    )
    parser.add_argument("--ball-model", type=Path, default=Path("exp.pt"), help="Basketball detection weights.")
    parser.add_argument("--pose-model", type=Path, default=Path("yolo26n-pose.pt"), help="Pose model weights.")
    add_camera_args(parser)
    add_ball_tracking_args(parser)
    parser.add_argument(
        "--ball-stride",
        type=int,
        default=2,
        help="Run basketball detection every N frames. Cached detections are reused on the skipped frames.",
    )
    parser.add_argument("--pose-conf", type=float, default=0.25, help="Pose confidence threshold.")
    parser.add_argument(
        "--pose-stride",
        type=int,
        default=3,
        help="Run pose inference every N frames by default. Basketball detection still runs every frame.",
    )
    parser.add_argument(
        "--hand-radius-px",
        type=float,
        default=120.0,
        help="If the detected ball is within this distance of a wrist, temporarily prioritize pose updates.",
    )
    parser.add_argument(
        "--hand-pose-boost",
        type=int,
        default=6,
        help="After the ball is seen near a wrist, keep running pose every frame for this many frames.",
    )
    parser.add_argument(
        "--release-distance-px",
        type=float,
        default=160.0,
        help="Ball is considered released once its nearest wrist distance exceeds this threshold after being near a hand.",
    )
    parser.add_argument(
        "--release-hold-frames",
        type=int,
        default=2,
        help="Require this many consecutive far-from-hand frames before marking the ball as released.",
    )
    parser.add_argument(
        "--missing-ball-release-frames",
        type=int,
        default=1,
        help="If the ball disappears right after being near the hand, trigger release after this many missing frames.",
    )
    parser.add_argument(
        "--uncertain-distance-px",
        type=float,
        default=130.0,
        help="Distances between hand-radius and this threshold are treated as uncertain before release.",
    )
    parser.add_argument(
        "--export-csv",
        type=Path,
        default=Path("runs") / "combined_detection.csv",
        help="CSV path used to save per-frame ball and pose detections.",
    )
    parser.add_argument(
        "--sqlite-db",
        type=Path,
        default=None,
        help="Optional SQLite database path used to store run metadata and per-frame records.",
    )
    parser.add_argument(
        "--save-video",
        type=Path,
        default=None,
        help="Optional output video path used to save the rendered demo with overlays.",
    )
    parser.add_argument(
        "--save-video-fps",
        type=float,
        default=None,
        help="Optional FPS for the saved demo video. Defaults to source FPS for files and 30 for camera.",
    )
    parser.add_argument(
        "--prediction-history-size",
        type=int,
        default=12,
        help="Number of recent wrist samples kept for release-velocity estimation.",
    )
    parser.add_argument(
        "--prediction-velocity-window",
        type=int,
        default=4,
        help="How many recent wrist samples are used to estimate hand speed.",
    )
    parser.add_argument(
        "--prediction-gravity-px-s2",
        type=float,
        default=1600.0,
        help="Downward image-space gravity used by the simple projectile preview.",
    )
    parser.add_argument(
        "--prediction-horizon-s",
        type=float,
        default=1.4,
        help="How far into the future to simulate the predicted ball flight.",
    )
    parser.add_argument(
        "--prediction-step-s",
        type=float,
        default=0.04,
        help="Time step used when sampling the predicted trajectory.",
    )
    parser.add_argument(
        "--prediction-min-speed-px-s",
        type=float,
        default=120.0,
        help="Minimum wrist speed required before a release trajectory is shown.",
    )
    parser.add_argument(
        "--prediction-persist-frames",
        type=int,
        default=18,
        help="How many frames a predicted release path stays visible after release.",
    )
    parser.add_argument(
        "--prediction-fit-samples",
        type=int,
        default=4,
        help="How many early post-release ball detections are used to refine the fitted launch trajectory.",
    )
    parser.add_argument(
        "--prediction-speed-scale",
        type=float,
        default=1.0,
        help="Scale factor applied to the estimated launch speed to compensate for low measured pose velocity.",
    )
    parser.add_argument(
        "--release-speed-threshold-px-s",
        type=float,
        default=220.0,
        help="Minimum hand speed used by the pose-based release trigger.",
    )
    parser.add_argument(
        "--release-angle-threshold-deg",
        type=float,
        default=20.0,
        help="Minimum forearm angle used by the pose-based release trigger.",
    )
    parser.add_argument(
        "--pose-release-min-ball-distance-px",
        type=float,
        default=150.0,
        help="Pose-based release only fires after the ball is at least this far from the nearest wrist.",
    )
    parser.add_argument(
        "--release-speed-drop-ratio",
        type=float,
        default=0.82,
        help="Lock release once hand speed drops below this ratio of the recent peak after the shooting motion crests.",
    )
    parser.add_argument(
        "--show-pose-debug",
        action="store_true",
        help="Draw key pose landmarks and their availability for debugging release timing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ball_model = YOLO(str(args.ball_model))
    pose_model = YOLO(str(args.pose_model))
    capture = open_capture(args.source, width=args.camera_width, height=args.camera_height)
    is_camera_source = str(args.source).isdigit()
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if source_fps <= 1e-6 or source_fps > 240.0:
        source_fps = 30.0
    tracking = BallTrackingOverlay(
        BallTrackingConfig(
            track_conf=args.track_conf,
            max_jump_px=args.max_jump_px,
            ema_alpha=args.ema_alpha,
            velocity_alpha=args.velocity_alpha,
            trail_length=args.trail_length,
            max_missing_frames=args.max_missing_frames,
            trail_thickness=args.trail_thickness,
            point_radius=args.point_radius,
        )
    )
    predictor = ReleaseTrajectoryPredictor(
        pose_conf=args.pose_conf,
        history_size=args.prediction_history_size,
        velocity_window=args.prediction_velocity_window,
        gravity_px_s2=args.prediction_gravity_px_s2,
        horizon_s=args.prediction_horizon_s,
        step_s=args.prediction_step_s,
        min_speed_px_s=args.prediction_min_speed_px_s,
        persist_frames=args.prediction_persist_frames,
        speed_scale=args.prediction_speed_scale,
        fit_sample_count=args.prediction_fit_samples,
    )
    pose_boost_frames = 0
    frame_idx = 0
    cached_ball_box = None
    cached_ball_conf = None
    cached_ball_point = None
    cached_pose_result = None
    records: list[dict[str, float | int | str | None]] = []
    db_writer: SQLiteRunWriter | None = None
    video_writer: cv2.VideoWriter | None = None
    start_time = time.perf_counter()
    was_near_hand = False
    released = False
    release_counter = 0
    release_state = "unknown"
    last_wrist_speed = None

    print("[INFO] Press 'q' to quit the live view.")
    if args.sqlite_db is not None:
        db_writer = SQLiteRunWriter(
            args.sqlite_db,
            source_path=args.source,
            ball_model_path=args.ball_model,
            pose_model_path=args.pose_model,
            export_csv_path=args.export_csv,
            params=vars(args),
        )
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frame_for_models = frame.copy()

        run_ball_now = frame_idx % max(1, args.ball_stride) == 0 or cached_ball_point is None
        if run_ball_now:
            ball_result = ball_model.predict(
                source=frame_for_models,
                conf=args.detect_conf,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )[0]
            cx, cy, conf, box = choose_ball_detection(
                ball_result,
                ball_class_id=args.ball_class_id,
                reference_point=tracking.reference_point(),
                track_gate_px=args.track_gate_px,
            )
            cached_ball_box = box
            cached_ball_conf = conf
            cached_ball_point = (cx, cy) if cx is not None and cy is not None else None
        else:
            box = cached_ball_box
            conf = cached_ball_conf
            if cached_ball_point is not None:
                cx, cy = cached_ball_point
            else:
                cx, cy = None, None
        ball_point = (cx, cy) if cx is not None and cy is not None else None

        run_pose_now = frame_idx % max(1, args.pose_stride) == 0 or pose_boost_frames > 0
        if run_pose_now:
            pose_result = pose_model.predict(
                source=frame_for_models,
                conf=args.pose_conf,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )[0]
            cached_pose_result = pose_result
            frame = pose_result.plot()
            if ball_near_wrists(
                pose_result,
                ball_point=ball_point,
                pose_conf=args.pose_conf,
                hand_radius_px=args.hand_radius_px,
            ):
                pose_boost_frames = args.hand_pose_boost
            elif pose_boost_frames > 0:
                pose_boost_frames -= 1
        elif cached_pose_result is not None:
            pose_result = cached_pose_result
            frame = pose_result.plot(img=frame)
        elif pose_boost_frames > 0:
            pose_boost_frames -= 1
            pose_result = None
        else:
            pose_result = None

        smooth_x, smooth_y = tracking.update(cx=cx, cy=cy, conf=conf)

        nearest_hand_distance = nearest_wrist_distance(
            pose_result,
            ball_point=ball_point,
            pose_conf=args.pose_conf,
        ) if pose_result is not None else None
        ball_near_hand = (
            nearest_hand_distance is not None and nearest_hand_distance <= args.hand_radius_px
        )

        if ball_near_hand:
            was_near_hand = True
            released = False
            release_counter = 0
            release_state = "near_hand"
        elif (
            was_near_hand
            and nearest_hand_distance is not None
            and args.uncertain_distance_px <= nearest_hand_distance < args.release_distance_px
        ):
            released = False
            release_counter = 0
            release_state = "uncertain"
        elif was_near_hand and nearest_hand_distance is not None and nearest_hand_distance >= args.release_distance_px:
            release_counter += 1
            release_state = "uncertain"
            if release_counter >= max(1, args.release_hold_frames):
                released = True
                release_state = "released"
        elif ball_point is None:
            if was_near_hand:
                release_counter += 1
                release_state = "released_missing"
                if release_counter >= max(1, args.missing_ball_release_frames):
                    released = True
                    release_state = "released"
                else:
                    released = False
            else:
                was_near_hand = False
                released = False
                release_counter = 0
                release_state = "unknown"
        else:
            released = False
            release_counter = 0
            release_state = "unknown"
        if box is not None:
            x1, y1, x2, y2 = box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(
                frame,
                f"ball {conf:.2f}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 220, 0),
                2,
                cv2.LINE_AA,
            )

        frame = tracking.draw(frame, smooth_x=smooth_x, smooth_y=smooth_y)
        if is_camera_source:
            timestamp_s = time.perf_counter() - start_time
        else:
            pos_msec = float(capture.get(cv2.CAP_PROP_POS_MSEC))
            if pos_msec > 0:
                timestamp_s = pos_msec / 1000.0
            else:
                timestamp_s = frame_idx / source_fps
        release_hint = release_state in {"released", "released_missing", "released_pose"}
        prediction, arm_snapshot = predictor.update(
            pose_result=pose_result,
            ball_point=ball_point,
            ball_near_hand=ball_near_hand,
            released=released,
            release_hint=release_hint,
            timestamp_s=timestamp_s,
            frame_idx=frame_idx,
            frame_shape=frame.shape,
        )

        pose_release = False
        wrist_speed_now = arm_snapshot.wrist_speed_px_s if arm_snapshot is not None else None
        release_angle_now = arm_snapshot.release_angle_deg if arm_snapshot is not None else None
        if (
            not released
            and was_near_hand
            and release_state in {"uncertain", "released_missing", "released"}
            and arm_snapshot is not None
            and wrist_speed_now is not None
            and release_angle_now is not None
            and last_wrist_speed is not None
            and last_wrist_speed >= args.release_speed_threshold_px_s
            and release_angle_now >= args.release_angle_threshold_deg
            and (
                (nearest_hand_distance is not None and nearest_hand_distance >= args.pose_release_min_ball_distance_px)
                or ball_point is None
                or wrist_speed_now <= last_wrist_speed * args.release_speed_drop_ratio
            )
            and wrist_speed_now < last_wrist_speed
        ):
            pose_release = True

        if pose_release:
            released = True
            release_state = "released_pose"
            prediction, arm_snapshot = predictor.update(
                pose_result=pose_result,
                ball_point=ball_point,
                ball_near_hand=ball_near_hand,
                released=True,
                release_hint=True,
                timestamp_s=timestamp_s,
                frame_idx=frame_idx,
                frame_shape=frame.shape,
            )

        last_wrist_speed = wrist_speed_now
        frame = predictor.draw(frame, prediction=prediction, snapshot=arm_snapshot)

        if args.show_pose_debug:
            landmarks = extract_debug_landmarks(pose_result, pose_conf=args.pose_conf)
            landmark_colors = {
                "left_shoulder": (0, 255, 255),
                "right_shoulder": (0, 255, 255),
                "left_elbow": (0, 165, 255),
                "right_elbow": (0, 165, 255),
                "left_wrist": (0, 0, 255),
                "right_wrist": (0, 0, 255),
                "torso_center": (255, 255, 0),
            }
            for name, point in landmarks.items():
                if point is None or name not in landmark_colors:
                    continue
                px, py = int(point[0]), int(point[1])
                color = landmark_colors[name]
                cv2.circle(frame, (px, py), 8, (255, 255, 255), -1)
                cv2.circle(frame, (px, py), 5, color, -1)
                cv2.putText(
                    frame,
                    name,
                    (px + 8, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    color,
                    1,
                    cv2.LINE_AA,
                )

            debug_lines = [
                f"release_state={release_state}",
                f"left_wrist={'ok' if landmarks.get('left_wrist') is not None else 'lost'}",
                f"right_wrist={'ok' if landmarks.get('right_wrist') is not None else 'lost'}",
                f"left_elbow={'ok' if landmarks.get('left_elbow') is not None else 'lost'}",
                f"right_elbow={'ok' if landmarks.get('right_elbow') is not None else 'lost'}",
                f"torso={'ok' if landmarks.get('torso_center') is not None else 'lost'}",
            ]
            for idx, text in enumerate(debug_lines):
                cv2.putText(
                    frame,
                    text,
                    (16, frame.shape[0] - 140 + idx * 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )


        cv2.putText(
            frame,
            f"ball_stride={args.ball_stride} pose_stride={args.pose_stride}",
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        if args.save_video is not None:
            if video_writer is None:
                args.save_video.parent.mkdir(parents=True, exist_ok=True)
                output_fps = args.save_video_fps
                if output_fps is None:
                    output_fps = source_fps if not is_camera_source else 30.0
                if output_fps <= 1e-6:
                    output_fps = 30.0
                frame_height, frame_width = frame.shape[:2]
                video_writer = cv2.VideoWriter(
                    str(args.save_video),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    float(output_fps),
                    (frame_width, frame_height),
                )
            video_writer.write(frame)

        cv2.imshow("Basketball + Pose", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        record: dict[str, float | int | str | None] = {
            "frame_idx": frame_idx,
            "timestamp_s": timestamp_s,
            "ball_detected": int(ball_point is not None),
            "ball_source": "detect" if run_ball_now else "cached",
            "ball_conf": conf,
            "ball_x1": box[0] if box is not None else None,
            "ball_y1": box[1] if box is not None else None,
            "ball_x2": box[2] if box is not None else None,
            "ball_y2": box[3] if box is not None else None,
            "ball_cx": cx,
            "ball_cy": cy,
            "nearest_wrist_distance_px": nearest_hand_distance,
            "ball_near_hand": int(ball_near_hand),
            "ball_released": int(released),
            "release_state": release_state,
            "shooting_side": arm_snapshot.side if arm_snapshot is not None else None,
            "elbow_angle_deg": arm_snapshot.elbow_angle_deg if arm_snapshot is not None else None,
            "release_angle_deg": arm_snapshot.release_angle_deg if arm_snapshot is not None else None,
            "wrist_speed_px_s": arm_snapshot.wrist_speed_px_s if arm_snapshot is not None else None,
            "predicted_release_x": prediction.start_point[0] if prediction is not None else None,
            "predicted_release_y": prediction.start_point[1] if prediction is not None else None,
            "predicted_vx0_px_s": prediction.velocity_px_s[0] if prediction is not None else None,
            "predicted_vy0_px_s": prediction.velocity_px_s[1] if prediction is not None else None,
        }
        if pose_result is not None:
            record.update(extract_pose_record(pose_result, pose_conf=args.pose_conf))
        else:
            empty_pose = extract_pose_record(type("EmptyPose", (), {"keypoints": None})(), pose_conf=args.pose_conf)
            record.update(empty_pose)
        records.append(record)
        if db_writer is not None:
            db_writer.append_frame_record(record)
        frame_idx += 1

    capture.release()
    cv2.destroyAllWindows()
    if db_writer is not None:
        db_writer.finalize()
    if video_writer is not None:
        video_writer.release()
    if records:
        args.export_csv.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(records).to_csv(args.export_csv, index=False)
        print(f"[DONE] Saved detection CSV to {args.export_csv}")
        if db_writer is not None:
            print(f"[DONE] Saved run to SQLite session_id={db_writer.session_id} at {args.sqlite_db}")
        if video_writer is not None:
            print(f"[DONE] Saved overlay video to {args.save_video}")


if __name__ == "__main__":
    main()
