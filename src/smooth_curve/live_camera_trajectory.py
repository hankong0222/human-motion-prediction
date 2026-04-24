from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

try:
    from .ball_tracking import (
        BallTrackingConfig,
        BallTrackingOverlay,
        choose_ball_detection,
        open_capture,
    )
    from .cli_args import add_ball_tracking_args, add_camera_args
except ImportError:  # pragma: no cover
    from ball_tracking import BallTrackingConfig, BallTrackingOverlay, choose_ball_detection, open_capture
    from cli_args import add_ball_tracking_args, add_camera_args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run realtime basketball trajectory visualization on a live camera feed."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("exp.pt"),
        help="Path to the YOLO weights used for basketball detection.",
    )
    add_camera_args(parser)
    add_ball_tracking_args(parser)
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    model = YOLO(str(args.model))
    capture = open_capture(args.source, width=args.camera_width, height=args.camera_height)
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

    print("[INFO] Press 'q' to quit the live view.")
    while True:
        ok, frame = capture.read()
        if not ok:
            break

        result = model.predict(
            source=frame,
            conf=args.detect_conf,
            imgsz=args.imgsz,
            device=args.device,
            verbose=False,
        )[0]

        cx, cy, conf, box = choose_ball_detection(
            result,
            ball_class_id=args.ball_class_id,
            reference_point=tracking.reference_point(),
            track_gate_px=args.track_gate_px,
        )
        smooth_x, smooth_y = tracking.update(cx=cx, cy=cy, conf=conf)

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
        cv2.putText(
            frame,
            f"det={args.detect_conf:.2f} track={args.track_conf:.2f}",
            (16, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Basketball Trajectory", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
