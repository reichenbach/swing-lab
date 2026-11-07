"""Feature extraction: turn the pose time-series into biomechanics numbers.

Everything here works from a single side-view camera, so the rotational
metrics (hip/shoulder angle, separation) are 2D *projections* of true 3D
rotation. They're directionally meaningful and great for tracking a player
against themselves over time, but they are not lab-grade. A second face-on
camera is what unlocks real 3D — that's a later phase. Comments flag every
spot where a metric is a proxy.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .pose import (
    L_ANKLE, L_HIP, L_SHOULDER, L_WRIST, NOSE,
    R_ANKLE, R_HIP, R_SHOULDER, R_WRIST, PoseSequence,
)


@dataclass
class SwingMetrics:
    fps: float
    n_frames: int
    contact_frame: int

    # Time series (length n_frames), angles in degrees, image coords.
    hip_angle: np.ndarray
    shoulder_angle: np.ndarray
    separation: np.ndarray          # shoulder_angle - hip_angle
    hand_speed: np.ndarray          # px/frame, lead-hands midpoint (raw, scale-dependent)
    hand_speed_norm: np.ndarray     # shoulder-widths/sec — scale- AND fps-invariant

    # Scalar summary.
    body_scale_px: float            # median shoulder width (px) — the scale reference
    max_separation: float
    max_separation_frame: int
    pelvis_peak_frame: int
    torso_peak_frame: int
    hands_peak_frame: int
    sequence_ok: bool               # pelvis -> torso -> hands order correct?
    attack_angle_deg: float         # hand-path angle near contact (+ = upward)
    head_drift_ratio: float         # head horizontal travel / shoulder width
    shoulder_tilt_deg: float        # back-shoulder drop at contact
    # Peak hand speed in shoulder-widths/sec. This is the trend-safe bat-speed
    # proxy: unlike raw px/frame it means the same thing whether the camera was
    # close or far, at 30fps or 240fps — so baseline-vs-latest comparisons are valid.
    hand_speed_peak_norm: float

    notes: list[str] = field(default_factory=list)

    # Phase 1 — populated only when bat/ball models are trained and supplied;
    # otherwise these stay at defaults and behavior is identical to Phase 0.
    contact_method: str = "hand-speed peak"
    bat_tracked: bool = False
    ball_tracked: bool = False
    bat_attack_angle_deg: float = float("nan")
    bat_angle_at_contact_deg: float = float("nan")
    bat_speed_peak: float = float("nan")          # px/frame (raw, scale-dependent)
    bat_speed_peak_norm: float = float("nan")     # shoulder-widths/sec (trend-safe)
    on_plane_ratio: float = float("nan")


def _interp_nan(arr: np.ndarray) -> np.ndarray:
    """Linearly fill NaN gaps (missed detections) along a 1-D series."""
    out = arr.copy()
    n = len(out)
    idx = np.arange(n)
    good = ~np.isnan(out)
    if good.sum() < 2:
        return np.nan_to_num(out)
    out[~good] = np.interp(idx[~good], idx[good], out[good])
    return out


def _smooth(arr: np.ndarray, win: int = 5) -> np.ndarray:
    """Centered moving average; odd window."""
    if win < 3:
        return arr
    win = win if win % 2 else win + 1
    pad = win // 2
    padded = np.pad(arr, pad, mode="edge")
    kernel = np.ones(win) / win
    return np.convolve(padded, kernel, mode="valid")


def _line_angle(p_a: np.ndarray, p_b: np.ndarray) -> np.ndarray:
    """Angle (deg) of the line A->B per frame. p_* are [n,2]. Used for the bat."""
    d = p_b - p_a
    return np.degrees(np.arctan2(d[:, 1], d[:, 0]))


def _unwrap_deg(a: np.ndarray) -> np.ndarray:
    return np.degrees(np.unwrap(np.radians(a)))


def _axial_rotation(p_a: np.ndarray, p_b: np.ndarray) -> np.ndarray:
    """2D proxy for rotation about the vertical axis, in degrees (0 to 180).

    A body segment turning toward or away from the camera gets shorter in the
    image rather than tilting. So rotation comes from how much the horizontal
    span A->B shrinks against its widest span in the clip. The result is bounded
    and stays continuous when the segment points straight at the camera.
    """
    dx = p_b[:, 0] - p_a[:, 0]
    widest = float(np.percentile(np.abs(dx), 95)) + 1e-6
    return np.degrees(np.arccos(np.clip(dx / widest, -1.0, 1.0)))


def _shoulder_tilt(l_sh: np.ndarray, r_sh: np.ndarray) -> np.ndarray:
    """Shoulder-line tilt in degrees, unaffected by how far the torso has turned."""
    d = r_sh - l_sh
    widest = float(np.percentile(np.linalg.norm(d, axis=1), 95)) + 1e-6
    return np.degrees(np.arcsin(np.clip(d[:, 1] / widest, -1.0, 1.0)))


def _first_major_peak(speed: np.ndarray, frac: float = 0.8) -> int:
    """Index of the top of the first burst reaching `frac` of the clip's peak.

    Clips often hold a second burst of similar speed as the bat comes around in
    the follow-through. Contact belongs to the first one.
    """
    above = speed >= frac * float(np.max(speed))
    start = int(np.argmax(above))
    end = start
    while end + 1 < len(speed) and above[end + 1]:
        end += 1
    return start + int(np.argmax(speed[start : end + 1]))


def compute(seq: PoseSequence, contact_frame: int | None = None,
            bat_track=None, ball_track=None) -> SwingMetrics:
    """Extract swing features.

    bat_track / ball_track are optional Phase-1 tracks (swing_lab.tracking).
    When present they sharpen the contact frame and replace the hand-path
    attack angle with the true bat-path attack angle. When absent the function
    behaves exactly as in Phase 0.
    """
    kp = seq.keypoints  # [n,17,3]
    n = seq.n_frames
    xy = kp[:, :, :2].astype(float)
    # Drop low-confidence points to NaN so interpolation handles them.
    conf = kp[:, :, 2]
    xy[conf < 0.3] = np.nan

    def pt(i):
        return np.stack([_interp_nan(xy[:, i, 0]), _interp_nan(xy[:, i, 1])], axis=1)

    l_sh, r_sh = pt(L_SHOULDER), pt(R_SHOULDER)
    l_hip, r_hip = pt(L_HIP), pt(R_HIP)
    l_wr, r_wr = pt(L_WRIST), pt(R_WRIST)
    l_ank, r_ank = pt(L_ANKLE), pt(R_ANKLE)
    nose = pt(NOSE)

    hands = (l_wr + r_wr) / 2.0

    # --- body scale: median shoulder width (px) -----------------------------
    # The reference length that makes distances/speeds comparable across videos
    # filmed at different camera distances or zoom. Everything scale-sensitive
    # is divided by this so a closer camera doesn't inflate the numbers.
    body_scale_px = float(np.nanmedian(np.linalg.norm(r_sh - l_sh, axis=1))) + 1e-6

    # --- rotation proxies (2D projection of 3D rotation) --------------------
    shoulder_angle = _smooth(_axial_rotation(l_sh, r_sh))
    hip_angle = _smooth(_axial_rotation(l_hip, r_hip))
    separation = shoulder_angle - hip_angle
    shoulder_tilt = _smooth(_shoulder_tilt(l_sh, r_sh))

    # --- hand speed = bat-speed proxy until we track the bat itself ---------
    hand_speed = _smooth(
        np.concatenate([[0.0], np.linalg.norm(np.diff(hands, axis=0), axis=1)])
    )
    # Scale- and fps-invariant version: (px/frame ÷ body_scale_px) × fps gives
    # shoulder-widths per second. This is what gets trended over time.
    hand_speed_norm = hand_speed / body_scale_px * seq.fps

    # --- contact frame ------------------------------------------------------
    # Priority: explicit override > ball-bat proximity > peak hand speed.
    contact_method = "hand-speed peak"
    if contact_frame is not None:
        contact_method = "manual override"
    else:
        contact_frame = _first_major_peak(hand_speed)
        if ball_track is not None and bat_track is not None:
            from .tracking.contact import detect_contact

            contact_frame, contact_method = detect_contact(
                ball_track, bat_track, n, contact_frame
            )
    contact_frame = int(np.clip(contact_frame, 0, n - 1))

    # --- kinematic sequence: when does each segment peak its angular speed? --
    pelvis_vel = np.abs(np.gradient(hip_angle))
    torso_vel = np.abs(np.gradient(shoulder_angle))
    # Look only in the down-swing window so backswing peaks don't fool us.
    lo, hi = max(0, contact_frame - int(0.4 * seq.fps)), contact_frame + 1
    win = slice(lo, hi)

    def peak_in_win(v):
        seg = v[win]
        return lo + int(np.argmax(seg)) if len(seg) else 0

    pelvis_peak = peak_in_win(pelvis_vel)
    torso_peak = peak_in_win(torso_vel)
    hands_peak = peak_in_win(hand_speed)
    sequence_ok = pelvis_peak <= torso_peak <= hands_peak

    # --- max hip/shoulder separation (look up to foot plant ~ down-swing) ----
    sep_win = separation[lo : contact_frame + 1]
    if len(sep_win):
        max_sep_frame = lo + int(np.argmax(np.abs(sep_win)))
        max_sep = float(separation[max_sep_frame])
    else:
        max_sep_frame, max_sep = 0, 0.0

    # --- attack angle: slope of hand path through contact (+ = upward) -------
    a = max(0, contact_frame - 3)
    b = min(n - 1, contact_frame + 1)
    dx = hands[b, 0] - hands[a, 0]
    dy = hands[b, 1] - hands[a, 1]
    # Image y grows downward, so negate dy: positive => hands rising => uppercut.
    attack_angle = float(np.degrees(np.arctan2(-dy, abs(dx) + 1e-6)))

    # --- head stability: horizontal nose travel during the swing only --------
    # Measured up to contact so walking off after the swing doesn't count.
    swing_nose_x = nose[lo : contact_frame + 1, 0]
    head_drift = float((np.nanmax(swing_nose_x) - np.nanmin(swing_nose_x)) / body_scale_px)

    # --- trend-safe peak hand speed (shoulder-widths/sec) -------------------
    hand_speed_peak_norm = float(np.nanmax(hand_speed_norm)) if n else 0.0

    # --- shoulder tilt at contact (back-shoulder drop) ----------------------
    tilt = float(shoulder_tilt[contact_frame] - np.median(shoulder_tilt[:max(1, lo)]))

    # --- Phase 1: prefer true bat-path metrics when a bat track is present ---
    bat_tracked = ball_tracked = False
    bat_attack = bat_angle_c = bat_speed = bat_speed_norm = on_plane = float("nan")
    if bat_track is not None and getattr(bat_track, "available", False):
        from .tracking.metrics_bat import compute_bat_metrics

        bm = compute_bat_metrics(bat_track, contact_frame, seq.fps)
        bat_tracked = True
        bat_attack, bat_angle_c = bm.attack_angle_deg, bm.bat_angle_at_contact_deg
        bat_speed, on_plane = bm.bat_speed_peak, bm.on_plane_ratio
        # Same normalization as hand speed: shoulder-widths/sec, trend-safe.
        bat_speed_norm = bat_speed / body_scale_px * seq.fps
        attack_angle = bm.attack_angle_deg  # true bat path beats the hand proxy
    if ball_track is not None and getattr(ball_track, "available", False):
        ball_tracked = True

    notes = []
    if seq.fps < 100:
        notes.append(
            f"Video is {seq.fps:.0f}fps. Use 120-240fps slow-mo for reliable "
            "contact + attack-angle numbers."
        )
    notes.append(
        "Rotation/separation are 2D proxies, not true 3D — best used to track a "
        "player against their own baseline."
    )
    notes.append(
        "Camera angle is not detected automatically. Attack angle, posture and "
        "head movement are graded only for side-on clips; rotation and separation "
        "only for face-on clips."
    )
    if bat_tracked:
        notes.append("Attack angle is from the tracked bat path (not the hand proxy).")

    return SwingMetrics(
        fps=seq.fps, n_frames=n, contact_frame=contact_frame,
        hip_angle=hip_angle, shoulder_angle=shoulder_angle, separation=separation,
        hand_speed=hand_speed, hand_speed_norm=hand_speed_norm,
        body_scale_px=body_scale_px, max_separation=max_sep,
        max_separation_frame=max_sep_frame, pelvis_peak_frame=pelvis_peak,
        torso_peak_frame=torso_peak, hands_peak_frame=hands_peak,
        sequence_ok=sequence_ok, attack_angle_deg=attack_angle,
        head_drift_ratio=head_drift, shoulder_tilt_deg=tilt,
        hand_speed_peak_norm=hand_speed_peak_norm, notes=notes,
        contact_method=contact_method, bat_tracked=bat_tracked,
        ball_tracked=ball_tracked, bat_attack_angle_deg=bat_attack,
        bat_angle_at_contact_deg=bat_angle_c, bat_speed_peak=bat_speed,
        bat_speed_peak_norm=bat_speed_norm, on_plane_ratio=on_plane,
    )
