"""Regressions for bugs that only showed up on real footage.

The first real clip produced 461 degrees of separation, called contact deep in
the follow-through, and counted walking away as head movement. These tests
rebuild each situation with synthetic keypoints.
"""

from __future__ import annotations

import numpy as np

from swing_lab.metrics import _first_major_peak, compute
from swing_lab.pose import (
    L_HIP, L_SHOULDER, L_WRIST, NOSE, R_HIP, R_SHOULDER, R_WRIST, PoseSequence,
)


def _pose(n=200, shoulder_dx=None, hip_dx=None, hand_x=None, nose_x=None) -> PoseSequence:
    kp = np.zeros((n, 17, 3), dtype=np.float32)
    kp[:, :, 2] = 0.9
    shoulder_dx = np.full(n, 70.0) if shoulder_dx is None else shoulder_dx
    hip_dx = np.full(n, 55.0) if hip_dx is None else hip_dx
    hand_x = np.full(n, 300.0) if hand_x is None else hand_x
    nose_x = np.full(n, 320.0) if nose_x is None else nose_x
    for i in range(n):
        kp[i, L_SHOULDER, :2] = [320 - shoulder_dx[i] / 2, 180]
        kp[i, R_SHOULDER, :2] = [320 + shoulder_dx[i] / 2, 180]
        kp[i, L_HIP, :2] = [320 - hip_dx[i] / 2, 250]
        kp[i, R_HIP, :2] = [320 + hip_dx[i] / 2, 250]
        kp[i, L_WRIST, :2] = [hand_x[i], 220]
        kp[i, R_WRIST, :2] = [hand_x[i] + 8, 224]
        kp[i, NOSE, :2] = [nose_x[i], 130]
    return PoseSequence(kp, 120.0, 640, 480)


def test_turning_through_the_camera_stays_physically_possible():
    # Hips and shoulders both turn so far that left and right swap sides in the
    # image, with the hips getting there first. This produced 461 degrees.
    n = 200
    t = np.linspace(0, 1, n)
    shoulder_dx = 70 * np.cos(np.pi * np.clip((t - 0.45) / 0.3, 0, 1))
    hip_dx = 55 * np.cos(np.pi * np.clip((t - 0.35) / 0.3, 0, 1))
    hand_x = 300 + 150 * np.clip((t - 0.40) / 0.3, 0, 1)
    m = compute(_pose(n, shoulder_dx, hip_dx, hand_x))
    assert 0 < abs(m.max_separation) <= 180, m.max_separation
    assert np.nanmax(m.hip_angle) - np.nanmin(m.hip_angle) <= 180
    assert np.nanmax(m.shoulder_angle) - np.nanmin(m.shoulder_angle) <= 180
    assert abs(m.shoulder_tilt_deg) < 5, m.shoulder_tilt_deg


def test_contact_uses_first_burst_not_follow_through():
    speed = np.zeros(200)
    speed[60:70] = [2, 5, 9, 14, 17, 16, 12, 7, 3, 1]    # through contact
    speed[120:130] = [2, 6, 10, 15, 18, 15, 11, 6, 2, 1]  # follow-through, slightly faster
    assert _first_major_peak(speed) == 64


def test_small_early_movement_is_not_contact():
    speed = np.zeros(200)
    speed[20:23] = [3, 5, 3]         # adjusting in the stance
    speed[90:95] = [6, 12, 18, 11, 5]
    assert _first_major_peak(speed) == 92


def test_walking_off_after_the_swing_is_not_head_movement():
    n = 200
    t = np.arange(n)
    hand_x = 300 + 150 * np.clip((t - 70) / 15, 0, 1)
    nose_x = 320 + 3.0 * np.clip(t - 110, 0, None)   # walks away after the swing
    m = compute(_pose(n, hand_x=hand_x, nose_x=nose_x))
    assert m.contact_frame < 110, m.contact_frame
    assert m.head_drift_ratio < 0.1, m.head_drift_ratio


if __name__ == "__main__":
    test_turning_through_the_camera_stays_physically_possible()
    test_contact_uses_first_burst_not_follow_through()
    test_small_early_movement_is_not_contact()
    test_walking_off_after_the_swing_is_not_head_movement()
    print("All real-clip regression checks passed.")
