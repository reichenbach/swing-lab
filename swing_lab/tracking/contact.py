"""Contact-frame detection.

Phase 0 estimates contact as peak hand speed — fine, but it can miss by a few
frames. With a ball track we can do much better: contact is the frame where the
ball is closest to the bat (point-to-segment distance, knob->tip). That also
gives us contact-point depth later.
"""

from __future__ import annotations

import numpy as np

from .bat_ball import BallTrack, BatTrack


def _point_seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    """Distance from point p to segment a-b (all xy)."""
    ab = b - a
    denom = float(ab @ ab) + 1e-9
    t = np.clip((p - a) @ ab / denom, 0.0, 1.0)
    proj = a + t * ab
    return float(np.linalg.norm(p - proj))


def detect_contact(ball: BallTrack, bat: BatTrack, n_frames: int,
                   fallback_frame: int) -> tuple[int, str]:
    """Return (contact_frame, method).

    Uses ball-bat proximity when both tracks exist; otherwise returns the
    Phase-0 fallback (hand-speed peak) unchanged.
    """
    if not (ball.available and bat.available):
        return fallback_frame, "hand-speed peak (no ball/bat track)"

    best_f, best_d = fallback_frame, np.inf
    for f in range(n_frames):
        bc = ball.center[f]
        knob, tip = bat.knob[f], bat.tip[f]
        if not (np.isfinite(bc).all() and np.isfinite(tip).all()):
            continue
        a = knob if np.isfinite(knob).all() else tip
        d = _point_seg_dist(bc, a, tip)
        if d < best_d:
            best_d, best_f = d, f
    return best_f, f"ball-bat proximity (min gap {best_d:.0f}px)"
