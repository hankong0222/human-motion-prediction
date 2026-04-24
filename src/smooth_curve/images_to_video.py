from __future__ import annotations

from pathlib import Path

import cv2

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


def convert_to_video(images_path: Path, output_path: Path, fps: int = 30) -> None:
    filenames = sorted(images_path.glob("*.jpg"), key=lambda path: path.stem)
    if not filenames:
        raise FileNotFoundError(f"No JPG frames found in {images_path}")

    first_frame = cv2.imread(str(filenames[0]))
    if first_frame is None:
        raise ValueError(f"Unable to read first image frame: {filenames[0]}")

    height, width, _ = first_frame.shape
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    iterator = tqdm(filenames, desc="Making a video", unit="frame") if tqdm else filenames
    for filename in iterator:
        frame = cv2.imread(str(filename))
        if frame is None:
            continue
        writer.write(frame)

    writer.release()
