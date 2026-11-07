"""Validate Phase-1 bat/ball integration without trained models or video.

We hand-build a BatTrack (knob/tip rising through contact) and a BallTrack
(ball passing through the bat tip at frame 80), push them through
compute() + diagnose(), and check that the pipeline switches to bat-derived
metrics — and that empty tracks fall back cleanly to Phase-0 behavior.

    PYTHONPATH=. .venv/bin/python tests/test_phase1.py
"""

from __future__ import annotations

import numpy as np

from swing_lab.diagnose import diagnose
from swing_lab.metrics import compute
from swing_lab.pose import PoseSequence
from swing_lab.tracking.bat_ball import BallTrack, BatTrack, empty_tracks


def _flat_pose(n=120) -> PoseSequence:
    """A static but plausible body — enough for compute() to run."""
    kp = np.zeros((n, 17, 3), np.float32)
    kp[:, :, 2] = 0.9
    base = {
        0: (320, 120), 5: (280, 180), 6: (360, 180), 7: (260, 230), 8: (380, 230),
        9: (250, 270), 10: (390, 270), 11: (290, 260), 12: (350, 260),
        13: (285, 340), 14: (355, 340), 15: (285, 420), 16: (355, 420),
    }
    for i, (x, y) in base.items():
        kp[:, i, 0], kp[:, i, 1] = x, y
    return PoseSequence(kp, 120.0, 640, 480)


def _bat_ball(n=120, contact=80):
    tip = np.zeros((n, 2), np.float32)
    knob = np.zeros((n, 2), np.float32)
    for i in range(n):
        prog = float(np.clip((i - 60) / 25, 0, 1))
        tx, ty = 250 + 170 * prog, 260 - 60 * prog   # bat tip rises through contact
        tip[i] = [tx, ty]
        knob[i] = [tx - 90, ty + 30]                 # constant bat vector
    bat = BatTrack(knob, tip, np.full(n, 0.9, np.float32))

    center = np.zeros((n, 2), np.float32)
    for i in range(n):
        center[i] = tip[contact] + (i - contact) * np.array([6.0, 3.0])  # ball passes tip@80
    ball = BallTrack(center, np.full(n, 0.9, np.float32))
    return bat, ball


def test_phase1():
    seq = _flat_pose()
    bat, ball = _bat_ball()
    m = compute(seq, bat_track=bat, ball_track=ball)

    assert m.bat_tracked and m.ball_tracked
    assert "proximity" in m.contact_method, m.contact_method
    assert abs(m.contact_frame - 80) <= 2, m.contact_frame
    assert np.isfinite(m.bat_attack_angle_deg)
    assert m.attack_angle_deg == m.bat_attack_angle_deg  # bat path beats hand proxy
    assert 0.0 <= m.on_plane_ratio <= 1.0

    findings = diagnose(m)
    timing = next(f for f in findings if "Timing" in f.flaw)
    assert "low" not in timing.confidence, timing.confidence  # upgraded by ball track

    # Fallback: empty tracks => identical to Phase 0 (nothing tracked).
    eb, ebl = empty_tracks(seq.n_frames)
    m0 = compute(seq, bat_track=eb, ball_track=ebl)
    assert not m0.bat_tracked and not m0.ball_tracked

    print("Phase1 OK — contact", m.contact_frame, "|", m.contact_method)
    print("  bat attack", round(m.bat_attack_angle_deg, 1),
          "| bat angle@contact", round(m.bat_angle_at_contact_deg, 1),
          "| on_plane", round(m.on_plane_ratio, 2),
          "| tip speed", round(m.bat_speed_peak, 1))
    for f in findings:
        print(f"  [{f.severity}] {f.flaw} [{f.confidence}]")


if __name__ == "__main__":
    test_phase1()
    print("\nAll Phase 1 checks passed.")
