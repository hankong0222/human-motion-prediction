# smooth_curve

This folder contains the basketball trajectory smoothing, live camera overlay, and detection-export pipeline.

## File Roles

- `data_io.py`
  Loads detection CSV files used by offline smoothing scripts.

- `realtime_smooth.py`
  Implements the realtime trajectory smoother for frame-by-frame ball tracking.

- `ball_tracking.py`
  Shared basketball tracking utilities for live scripts:
  camera opening, ball candidate selection, trail rendering, and overlay state.

- `pose_utils.py`
  Shared pose helpers:
  wrist distance checks, near-hand logic, and full-keypoint export formatting.

- `cli_args.py`
  Shared argparse helpers for camera and ball-tracking parameters used by live scripts.

- `draw.py`
  Offline smoothing entrypoint for detection CSV files.
  Produces a `*_smooth.csv` file from raw detection outputs.

- `build_smooth_curve.py`
  Thin compatibility wrapper around `draw.py`.

- `live_camera_trajectory.py`
  Live camera mode for basketball-only detection plus trajectory overlay.

- `live_camera_combined.py`
  Live camera mode for basketball detection, pose estimation, release-state heuristics,
  per-frame detection export to CSV, and prediction overlays imported from `src/prediction`.

- `video_to_images.py`
  Converts a video file into image frames.

- `images_to_video.py`
  Converts an image sequence back into a video.

- `__init__.py`
  Re-exports the main shared functions and classes from this package.

## Common Commands

### 1. Ball-only live trajectory

```powershell
.\venv\Scripts\python.exe .\src\smooth_curve\live_camera_trajectory.py `
  --model .\exp.pt `
  --source 0 `
  --device cpu `
  --imgsz 960
```

### 2. Ball + pose live view with CSV export

```powershell
.\venv\Scripts\python.exe .\src\smooth_curve\live_camera_combined.py `
  --ball-model .\exp.pt `
  --pose-model .\yolo26n-pose.pt `
  --source 0 `
  --device cpu `
  --imgsz 960 `
  --ball-stride 2 `
  --pose-stride 3 `
  --export-csv .\runs\combined_detection.csv
```

This exports a time-series CSV with:
- per-frame ball detection
- all pose keypoints
- hand-distance fields
- release-state fields
- heuristic release-prediction fields

You can tune the pose-based release preview with:
- `--prediction-gravity-px-s2`
- `--prediction-horizon-s`
- `--prediction-step-s`
- `--prediction-min-speed-px-s`
- `--prediction-persist-frames`

### 3. Offline smooth curve from detection CSV

```powershell
.\venv\Scripts\python.exe .\src\smooth_curve\build_smooth_curve.py `
  --input .\data\hard_examples\detections\WIN_20260322_15_09_02_Pro.csv
```

### 4. Realtime smoothing from detection CSV

```powershell
.\venv\Scripts\python.exe .\src\smooth_curve\realtime_smooth.py `
  --input .\data\hard_examples\detections\WIN_20260322_15_09_02_Pro.csv
```

### 5. Video to frames

```python
from pathlib import Path
from src.smooth_curve import convert_to_images

convert_to_images(
    video_path=Path("input.mp4"),
    output_path=Path("frames"),
    video_stride=1,
)
```

### 6. Frames to video

```python
from pathlib import Path
from src.smooth_curve import convert_to_video

convert_to_video(
    images_path=Path("frames"),
    output_path=Path("output.mp4"),
    fps=30,
)
```

## Notes

- `live_camera_combined.py` is the main script to use when you need ball + body detections exported for later prediction.
- Pose-angle, motion, and release-trajectory prediction logic now lives under `src/prediction`.
- The current release preview is a 2D image-space approximation, not a calibrated 3D physics model.
- The exported CSV is intended to store detection signals only.
  The on-screen trajectory is for visualization and debugging, not for training labels.
- On CPU, the main performance knobs are:
  - `--imgsz`
  - `--ball-stride`
  - `--pose-stride`
  - `--trail-length`
