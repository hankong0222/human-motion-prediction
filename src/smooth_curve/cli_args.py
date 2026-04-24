from __future__ import annotations

import argparse


def add_camera_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--source", default="0", help="Camera index or OpenCV video source.")
    parser.add_argument("--device", default="cpu", help="Inference device, e.g. cpu, 0, 0,1.")
    parser.add_argument("--imgsz", type=int, default=960, help="Inference image size.")
    parser.add_argument("--camera-width", type=int, default=1280, help="Requested camera width.")
    parser.add_argument("--camera-height", type=int, default=720, help="Requested camera height.")
    return parser


def add_ball_tracking_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--ball-class-id", type=int, default=0, help="Basketball class id in the detection model.")
    parser.add_argument("--detect-conf", type=float, default=0.05, help="Detection confidence threshold.")
    parser.add_argument("--track-conf", type=float, default=0.12, help="Minimum confidence used by the trajectory smoother.")
    parser.add_argument("--trail-length", type=int, default=0, help="Number of trail points kept. 0 keeps full history.")
    parser.add_argument("--trail-thickness", type=int, default=7, help="Base thickness of the trajectory line.")
    parser.add_argument("--point-radius", type=int, default=10, help="Radius of the smoothed ball point.")
    parser.add_argument("--track-gate-px", type=float, default=180.0, help="Candidate gating distance to the current trajectory.")
    parser.add_argument("--max-jump-px", type=float, default=140.0, help="Maximum accepted jump between smoothed points.")
    parser.add_argument("--max-missing-frames", type=int, default=6, help="How many consecutive misses are tolerated before resetting the active ball track.")
    parser.add_argument("--ema-alpha", type=float, default=0.45, help="Realtime smoother measurement blending factor.")
    parser.add_argument("--velocity-alpha", type=float, default=0.35, help="Realtime smoother velocity blending factor.")
    return parser
