from __future__ import annotations

import numpy as np


KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def get_best_person_keypoints(pose_result):
    keypoints = getattr(pose_result, "keypoints", None)
    if keypoints is None or keypoints.xy is None or keypoints.conf is None:
        return None, None

    xy_batch = keypoints.xy.cpu().numpy()
    conf_batch = keypoints.conf.cpu().numpy()
    if len(xy_batch) == 0:
        return None, None

    best_idx = int(np.argmax(conf_batch.mean(axis=1)))
    return xy_batch[best_idx], conf_batch[best_idx]


def get_keypoint(
    person_xy: np.ndarray,
    person_conf: np.ndarray,
    idx: int,
    pose_conf: float,
) -> tuple[float, float] | None:
    if idx >= len(person_conf) or float(person_conf[idx]) < pose_conf:
        return None
    return float(person_xy[idx][0]), float(person_xy[idx][1])


def extract_arm_points(pose_result, pose_conf: float = 0.25) -> dict[str, tuple[float, float] | None] | None:
    person_xy, person_conf = get_best_person_keypoints(pose_result)
    if person_xy is None or person_conf is None:
        return None

    return {
        "left_shoulder": get_keypoint(person_xy, person_conf, 5, pose_conf),
        "right_shoulder": get_keypoint(person_xy, person_conf, 6, pose_conf),
        "left_elbow": get_keypoint(person_xy, person_conf, 7, pose_conf),
        "right_elbow": get_keypoint(person_xy, person_conf, 8, pose_conf),
        "left_wrist": get_keypoint(person_xy, person_conf, 9, pose_conf),
        "right_wrist": get_keypoint(person_xy, person_conf, 10, pose_conf),
    }


def extract_torso_points(pose_result, pose_conf: float = 0.25) -> dict[str, tuple[float, float] | None] | None:
    person_xy, person_conf = get_best_person_keypoints(pose_result)
    if person_xy is None or person_conf is None:
        return None

    return {
        "left_shoulder": get_keypoint(person_xy, person_conf, 5, pose_conf),
        "right_shoulder": get_keypoint(person_xy, person_conf, 6, pose_conf),
        "left_hip": get_keypoint(person_xy, person_conf, 11, pose_conf),
        "right_hip": get_keypoint(person_xy, person_conf, 12, pose_conf),
    }


def midpoint(
    point_a: tuple[float, float] | None,
    point_b: tuple[float, float] | None,
) -> tuple[float, float] | None:
    if point_a is None or point_b is None:
        return None
    return ((point_a[0] + point_b[0]) / 2.0, (point_a[1] + point_b[1]) / 2.0)


def extract_body_centers(pose_result, pose_conf: float = 0.25) -> dict[str, tuple[float, float] | None] | None:
    torso_points = extract_torso_points(pose_result, pose_conf=pose_conf)
    if torso_points is None:
        return None

    shoulder_center = midpoint(torso_points["left_shoulder"], torso_points["right_shoulder"])
    hip_center = midpoint(torso_points["left_hip"], torso_points["right_hip"])
    torso_center = midpoint(shoulder_center, hip_center)

    return {
        "shoulder_center": shoulder_center,
        "hip_center": hip_center,
        "torso_center": torso_center,
    }


def extract_debug_landmarks(pose_result, pose_conf: float = 0.25) -> dict[str, tuple[float, float] | None]:
    arm_points = extract_arm_points(pose_result, pose_conf=pose_conf) or {
        "left_shoulder": None,
        "right_shoulder": None,
        "left_elbow": None,
        "right_elbow": None,
        "left_wrist": None,
        "right_wrist": None,
    }
    body_centers = extract_body_centers(pose_result, pose_conf=pose_conf) or {
        "shoulder_center": None,
        "hip_center": None,
        "torso_center": None,
    }
    landmarks: dict[str, tuple[float, float] | None] = {}
    landmarks.update(arm_points)
    landmarks.update(body_centers)
    return landmarks


def nearest_wrist_distance(pose_result, ball_point: tuple[float, float] | None, pose_conf: float) -> float | None:
    if ball_point is None:
        return None

    keypoints = getattr(pose_result, "keypoints", None)
    if keypoints is None or keypoints.xy is None or keypoints.conf is None:
        return None

    xy_batch = keypoints.xy.cpu().numpy()
    conf_batch = keypoints.conf.cpu().numpy()
    best_distance = None

    for person_xy, person_conf in zip(xy_batch, conf_batch):
        for wrist_idx in (9, 10):
            wrist = get_keypoint(person_xy, person_conf, wrist_idx, pose_conf)
            if wrist is None:
                continue
            wrist_x, wrist_y = wrist
            distance = float(np.hypot(ball_point[0] - wrist_x, ball_point[1] - wrist_y))
            if best_distance is None or distance < best_distance:
                best_distance = distance
    return best_distance


def ball_near_wrists(
    pose_result,
    ball_point: tuple[float, float] | None,
    pose_conf: float,
    hand_radius_px: float,
) -> bool:
    nearest_distance = nearest_wrist_distance(pose_result, ball_point=ball_point, pose_conf=pose_conf)
    return nearest_distance is not None and nearest_distance <= hand_radius_px


def extract_pose_record(pose_result, pose_conf: float) -> dict[str, float | int | str | None]:
    record: dict[str, float | int | str | None] = {}
    for name in KEYPOINT_NAMES:
        record[f"{name}_x"] = None
        record[f"{name}_y"] = None
        record[f"{name}_conf"] = None

    keypoints = getattr(pose_result, "keypoints", None)
    if keypoints is None or keypoints.xy is None or keypoints.conf is None:
        record["pose_detected"] = 0
        return record

    person_xy, person_conf = get_best_person_keypoints(pose_result)
    if person_xy is None or person_conf is None:
        record["pose_detected"] = 0
        return record

    record["pose_detected"] = 1

    for idx, name in enumerate(KEYPOINT_NAMES):
        conf = float(person_conf[idx])
        record[f"{name}_conf"] = conf
        point = get_keypoint(person_xy, person_conf, idx, pose_conf)
        if point is not None:
            record[f"{name}_x"] = point[0]
            record[f"{name}_y"] = point[1]

    return record
