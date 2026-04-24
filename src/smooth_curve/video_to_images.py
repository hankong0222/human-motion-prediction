from __future__ import annotations

from pathlib import Path

import cv2


def convert_to_images(video_path: Path, output_path: Path, video_stride: int = 1) -> None:
    if video_stride < 1:
        raise ValueError("Video stride must be positive.")

    output_path.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    current_frame = 0
    while capture.isOpened():
        ret, image = capture.read()
        if not ret:
            break

        if current_frame % video_stride == 0:
            cv2.imwrite(str(output_path / f"{current_frame:05d}.jpg"), image)

        current_frame += 1

    capture.release()
