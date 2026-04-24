from __future__ import annotations

from pathlib import Path

import pandas as pd


DETECTION_COLUMNS = {"frame_idx", "timestamp_s", "detected", "conf", "cx", "cy"}


def load_detection_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = DETECTION_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Detection CSV is missing required columns: {sorted(missing)}")

    df = df.copy()
    df["detected"] = df["detected"].fillna(0).astype(int)
    for column in ["timestamp_s", "conf", "cx", "cy"]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    return df
