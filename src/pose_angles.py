from __future__ import annotations

from dataclasses import dataclass

import numpy as np

try:
    from smooth_curve.pose_utils import extract_arm_points
except ImportError:  # pragma: no cover
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1] / "smooth_curve"))
    from pose_utils import extract_arm_points


@dataclass
class ArmAngles:
    side: str
    shoulder: tuple[float, float] | None
    elbow: tuple[float, float] | None
    wrist: tuple[float, float] | None
    elbow_angle_deg: float | None
    forearm_angle_deg: float | None


def calculate_joint_angle(
    point_a: tuple[float, float] | None,
    point_b: tuple[float, float] | None,
    point_c: tuple[float, float] | None,
) -> float | None:
    if point_a is None or point_b is None or point_c is None:
        return None

    vector_ba = np.array(point_a, dtype=np.float32) - np.array(point_b, dtype=np.float32)
    vector_bc = np.array(point_c, dtype=np.float32) - np.array(point_b, dtype=np.float32)

    norm_ba = float(np.linalg.norm(vector_ba))
    norm_bc = float(np.linalg.norm(vector_bc))
    if norm_ba <= 1e-6 or norm_bc <= 1e-6:
        return None

    cosine = float(np.dot(vector_ba, vector_bc) / (norm_ba * norm_bc))
    cosine = float(np.clip(cosine, -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def calculate_segment_angle(
    start_point: tuple[float, float] | None,
    end_point: tuple[float, float] | None,
) -> float | None:
    if start_point is None or end_point is None:
        return None

    vector = np.array(end_point, dtype=np.float32) - np.array(start_point, dtype=np.float32)
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-6:
        return None

    return float(np.degrees(np.arctan2(-vector[1], vector[0])))


def calculate_elbow_angle(
    shoulder: tuple[float, float] | None,
    elbow: tuple[float, float] | None,
    wrist: tuple[float, float] | None,
) -> float | None:
    return calculate_joint_angle(shoulder, elbow, wrist)


def extract_arm_angles(
    pose_result,
    pose_conf: float = 0.25,
    side: str | None = None,
) -> dict[str, ArmAngles] | ArmAngles | None:
    arm_points = extract_arm_points(pose_result, pose_conf=pose_conf)
    if arm_points is None:
        return None

    angles_by_side = {
        "left": ArmAngles(
            side="left",
            shoulder=arm_points["left_shoulder"],
            elbow=arm_points["left_elbow"],
            wrist=arm_points["left_wrist"],
            elbow_angle_deg=calculate_elbow_angle(
                arm_points["left_shoulder"],
                arm_points["left_elbow"],
                arm_points["left_wrist"],
            ),
            forearm_angle_deg=calculate_segment_angle(
                arm_points["left_elbow"],
                arm_points["left_wrist"],
            ),
        ),
        "right": ArmAngles(
            side="right",
            shoulder=arm_points["right_shoulder"],
            elbow=arm_points["right_elbow"],
            wrist=arm_points["right_wrist"],
            elbow_angle_deg=calculate_elbow_angle(
                arm_points["right_shoulder"],
                arm_points["right_elbow"],
                arm_points["right_wrist"],
            ),
            forearm_angle_deg=calculate_segment_angle(
                arm_points["right_elbow"],
                arm_points["right_wrist"],
            ),
        ),
    }

    if side is None:
        return angles_by_side
    return angles_by_side.get(side)
