"""Bat-derived metrics — the real versions of what Phase 0 only proxied.

With an actual bat vector (knob->tip) per frame we can measure the bat path
directly instead of inferring it from the hands:

  - **attack angle**  : slope of the bat *tip* path through contact (+ = upward)
  - **bat angle**     : orientation of the bat at contact (vertical bat angle)
  - **bat speed**     : tip speed (px/frame; calibrate to mph later with a
                        known reference length in frame)
  - **on-plane ratio**: how steadily the bat holds its plane into contact —
                        the signal that separates a true on-plane swing from a
                        chop/roll or an upper-cut.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..metrics import _interp_nan, _line_angle, _smooth, _unwrap_deg
from .bat_ball import BatTrack


@dataclass
class BatMetrics:
    available: bool
    attack_angle_deg: float = float("nan")
    bat_angle_at_contact_deg: float = float("nan")
    bat_speed_peak: float = float("nan")     # px/frame of the tip
    on_plane_ratio: float = float("nan")     # 0..1 over the approach to contact


def compute_bat_metrics(bat: BatTrack, contact_frame: int, fps: float,
                        plane_band_deg: float = 12.0) -> BatMetrics:
    if not bat.available:
        return BatMetrics(available=False)

    n = len(bat.tip)
    tip = np.stack([_interp_nan(bat.tip[:, 0]), _interp_nan(bat.tip[:, 1])], axis=1)
    knob = np.stack([_interp_nan(bat.knob[:, 0]), _interp_nan(bat.knob[:, 1])], axis=1)

    # Attack angle from the tip path through contact (image y is down -> negate).
    a = max(0, contact_frame - 3)
    b = min(n - 1, contact_frame + 1)
    dx, dy = tip[b, 0] - tip[a, 0], tip[b, 1] - tip[a, 1]
    attack = float(np.degrees(np.arctan2(-dy, abs(dx) + 1e-6)))

    # Bat orientation (knob->tip) over time; value at contact = vertical bat angle.
    bat_angle = _smooth(_unwrap_deg(_line_angle(knob, tip)))
    bat_angle_contact = float(bat_angle[contact_frame])

    # Tip speed; peak in the down-swing window.
    tip_speed = _smooth(
        np.concatenate([[0.0], np.linalg.norm(np.diff(tip, axis=0), axis=1)])
    )
    lo = max(0, contact_frame - int(0.4 * fps))
    bat_speed_peak = float(np.max(tip_speed[lo:contact_frame + 1])) if contact_frame > lo else 0.0

    # On-plane: over the last ~0.12s into contact, how steadily does the bat
    # hold the plane it arrives on? Fraction of frames within +-band of the
    # contact-frame bat angle.
    w = max(2, int(0.12 * fps))
    seg = bat_angle[max(0, contact_frame - w):contact_frame + 1]
    on_plane = float(np.mean(np.abs(seg - bat_angle_contact) < plane_band_deg)) if len(seg) else float("nan")

    return BatMetrics(
        available=True,
        attack_angle_deg=attack,
        bat_angle_at_contact_deg=bat_angle_contact,
        bat_speed_peak=bat_speed_peak,
        on_plane_ratio=on_plane,
    )
