"""Rendering: annotated swing video + a metrics plot the kid can look at."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

from .metrics import SwingMetrics
from .pose import (
    L_ANKLE, L_EAR, L_WRIST, NOSE, R_ANKLE, R_EAR, R_WRIST, SKELETON, PoseSequence,
)


def _to_browser_h264(path: str) -> str:
    """Re-encode an mp4v/MPEG-4 file to H.264 so it plays in an HTML5 <video>.

    OpenCV's default `mp4v` (MPEG-4 Part 2) does not play in Chrome/Safari, so
    the annotated clip is invisible in the web app. If ffmpeg is on PATH we
    transcode in place to yuv420p H.264 with `+faststart` (moov atom up front so
    the browser can start playing before the whole file downloads). If ffmpeg is
    missing we leave the file as-is and let the caller decide — the CLI is happy
    with mp4v; only the browser cares.
    """
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return path
    src = Path(path)
    tmp = src.with_name(src.stem + "_h264.mp4")
    try:
        subprocess.run(
            [ffmpeg, "-y", "-loglevel", "error", "-i", str(src),
             "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-movflags", "+faststart", str(tmp)],
            check=True,
        )
    except (subprocess.CalledProcessError, OSError):
        return path  # transcode failed; keep the original rather than lose it
    tmp.replace(src)
    return str(src)


# Colors are BGR.
SKELETON_COLOR = (255, 225, 120)
OUTLINE_COLOR = (25, 25, 25)
DOT_COLOR = (240, 240, 240)
HAND_PATH_COLOR = (255, 60, 230)      # magenta tracer
BAT_COLOR = (235, 235, 235)
BALL_COLOR = (0, 255, 255)
OK_COLOR = (110, 215, 90)             # green: head inside the box
WARN_COLOR = (0, 185, 255)            # amber: just outside
BAD_COLOR = (70, 70, 235)             # red: well outside
TEXT_COLOR = (245, 245, 245)
MUTED_COLOR = (175, 175, 175)
NEUTRAL_COLOR = (210, 210, 210)       # head box when the view can't grade it
CONTACT_COLOR = (40, 40, 230)

MIN_CONF = 0.3
HEAD_POINTS = (NOSE, L_EAR, R_EAR)


def _valid_xy(kp: np.ndarray, idx) -> np.ndarray:
    """[n, len(idx), 2] positions with low-confidence points set to NaN."""
    pts = kp[:, idx, :2].astype(float)
    conf = kp[:, idx, 2]
    pts[~(conf >= MIN_CONF)] = np.nan
    return pts


def _centroid(kp: np.ndarray, idx) -> np.ndarray:
    """Per-frame mean of the confident points in idx; NaN when none are."""
    pts = _valid_xy(kp, list(idx))
    with np.errstate(invalid="ignore"):
        cnt = np.isfinite(pts[:, :, 0]).sum(axis=1)
        tot = np.nansum(pts, axis=1)
    out = tot / np.maximum(cnt, 1)[:, None]
    out[cnt == 0] = np.nan
    return out


def _settled(track: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Median position over frames [lo, hi], falling back to the next valid one."""
    seg = track[lo : hi + 1]
    seg = seg[np.isfinite(seg).all(axis=1)]
    if len(seg):
        return np.median(seg, axis=0)
    later = track[hi:]
    later = later[np.isfinite(later).all(axis=1)]
    return later[0] if len(later) else np.array([np.nan, np.nan])


def _text(img, txt, org, scale, color=TEXT_COLOR, thick=1):
    import cv2

    # Outline by offset copies at the same thickness: a thicker outline pass
    # advances each glyph further and drifts out of line with the text.
    x, y = org
    for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        cv2.putText(img, txt, (x + dx, y + dy), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    OUTLINE_COLOR, thick, cv2.LINE_AA)
    cv2.putText(img, txt, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick,
                cv2.LINE_AA)


def _text_width(txt, scale, thick=1) -> int:
    import cv2

    return cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0]


def _dashed_vline(img, x, color, thick, dash, gap):
    import cv2

    for y0 in range(0, img.shape[0], dash + gap):
        cv2.line(img, (x, y0), (x, min(img.shape[0] - 1, y0 + dash)), color, thick,
                 cv2.LINE_AA)


def _panel(img, x, y, w, h, alpha=0.55):
    """Darken a rectangle so text on top stays readable on any background."""
    x2, y2 = min(img.shape[1], x + w), min(img.shape[0], y + h)
    roi = img[y:y2, x:x2]
    roi[:] = (roi * (1 - alpha)).astype(img.dtype)


def _drift_color(level: float):
    if level <= 1.0:
        return OK_COLOR
    return WARN_COLOR if level <= 1.5 else BAD_COLOR


def render_video(seq: PoseSequence, m: SwingMetrics, in_path: str, out_path: str,
                 bat_track=None, ball_track=None, web_safe: bool = True, *,
                 view: str | None = None, smoothed: bool = False,
                 hand_trail_s: float = 0.3, swing_window_s: float = 0.4) -> str:
    """Draw the swing overlay on each frame of the source video.

    Elements, all in image space with no absolute units:

    - skeleton from `seq` (pass the smoothed sequence; `smoothed` only changes
      the footer label);
    - head stability box: fixed where the head (nose/ear centroid) settled at
      swing start, sized to the head-drift grading threshold, with a head trace
      that turns amber and red as it leaves the box (side-on only; other views
      draw it neutral, since they can't see the head move toward the pitcher);
    - stance plumb line through the ankle midpoint at swing start;
    - fading magenta tracer of the hands over the last `hand_trail_s` seconds;
    - phase label (pre-contact / contact / follow-through) and a CONTACT flash;
    - readout panel showing only what `view` can support ("face", "side", or
      None for no readouts);
    - bat line and ball trail when those tracks are available.

    "Swing start" is `swing_window_s` before the contact frame, the same window
    the metrics use for the downswing and head drift.

    web_safe=True transcodes the result to browser-playable H.264 via ffmpeg
    (see `_to_browser_h264`). Set False to skip the transcode for a faster
    CLI-only render.
    """
    import cv2

    from .diagnose import THRESHOLDS

    def _xy(p):
        return (int(round(p[0])), int(round(p[1])))

    def _ok(*pts):
        return all(np.isfinite(p).all() for p in pts)

    n, fps, contact = seq.n_frames, seq.fps, m.contact_frame
    kp = seq.keypoints
    s = max(0.6, seq.height / 720.0)                 # UI scale vs a 720p frame
    lw = max(2, int(round(3 * s)))                   # skeleton line width

    # --- precomputed tracks ------------------------------------------------
    head = _centroid(kp, HEAD_POINTS) if n else np.zeros((0, 2))
    hands = _centroid(kp, (L_WRIST, R_WRIST)) if n else np.zeros((0, 2))
    ankles = _centroid(kp, (L_ANKLE, R_ANKLE)) if n else np.zeros((0, 2))

    start = max(0, contact - int(round(swing_window_s * fps)))
    settle = max(0, start - int(round(0.1 * fps)))
    trace_end = min(n - 1, contact + int(round(0.25 * fps)))
    anchor = _settled(head, settle, start) if n else np.array([np.nan, np.nan])
    plumb = _settled(ankles, settle, start) if n else np.array([np.nan, np.nan])
    half = 0.5 * THRESHOLDS["head_drift_max"] * m.body_scale_px

    # How far outside the box the head is, per frame: <= 1 inside.
    with np.errstate(invalid="ignore"):
        drift = np.max(np.abs(head - anchor) / max(half, 1e-6), axis=1)

    trail_n = max(4, int(round(hand_trail_s * fps)))

    # Only a side-on camera sees the head move toward the pitcher, so only then
    # does the box carry a verdict color. Otherwise it is drawn neutral.
    def head_color(level):
        return _drift_color(level) if view == "side" else NEUTRAL_COLOR

    cap = cv2.VideoCapture(in_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (seq.width, seq.height))

    f = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if f < n:
            k = kp[f]

            # Stance plumb line, behind everything else.
            if np.isfinite(plumb[0]):
                layer = frame.copy()
                _dashed_vline(layer, int(round(plumb[0])), (220, 220, 220),
                              max(1, int(round(1.5 * s))), int(14 * s), int(10 * s))
                cv2.addWeighted(layer, 0.45, frame, 0.55, 0, frame)

            # Head box (fixed reference) and the head trace through the swing.
            if np.isfinite(anchor).all() and half > 0:
                tl = _xy(anchor - half)
                br = _xy(anchor + half)
                in_swing = start <= f <= trace_end
                box_col = head_color(drift[f]) if in_swing and np.isfinite(drift[f]) \
                    else NEUTRAL_COLOR
                cv2.rectangle(frame, tl, br, OUTLINE_COLOR, lw + 2, cv2.LINE_AA)
                cv2.rectangle(frame, tl, br, box_col, max(1, lw - 1), cv2.LINE_AA)
                last = min(f, trace_end)
                for i in range(start + 1, last + 1):
                    a, b = head[i - 1], head[i]
                    if _ok(a, b):
                        cv2.line(frame, _xy(a), _xy(b), head_color(drift[i]),
                                 max(2, lw - 1), cv2.LINE_AA)
                if in_swing and _ok(head[f]):
                    cv2.circle(frame, _xy(head[f]), lw + 1, head_color(drift[f]), -1,
                               cv2.LINE_AA)

            # Skeleton: dark outline under a light line reads on any background.
            segs = [(k[a], k[b]) for a, b in SKELETON
                    if k[a, 2] >= MIN_CONF and k[b, 2] >= MIN_CONF and _ok(k[a, :2], k[b, :2])]
            for pa, pb in segs:
                cv2.line(frame, _xy(pa), _xy(pb), OUTLINE_COLOR, lw + 2, cv2.LINE_AA)
            for pa, pb in segs:
                cv2.line(frame, _xy(pa), _xy(pb), SKELETON_COLOR, lw, cv2.LINE_AA)
            for x, y, c in k:
                if c >= MIN_CONF and np.isfinite(x) and np.isfinite(y):
                    cv2.circle(frame, (int(round(x)), int(round(y))), max(2, lw - 1),
                               DOT_COLOR, -1, cv2.LINE_AA)

            # Hand-path tracer: older segments thinner and more transparent.
            lo = max(0, f - trail_n)
            pts = [(i, hands[i]) for i in range(lo, f + 1) if _ok(hands[i])]
            if len(pts) >= 2:
                bands = ((0.0, 0.34, 0.25), (0.34, 0.67, 0.55), (0.67, 1.01, 1.0))
                for b_lo, b_hi, alpha in bands:
                    layer = frame.copy() if alpha < 1 else frame
                    drew = False
                    for (i0, p0), (i1, p1) in zip(pts[:-1], pts[1:]):
                        age = (i1 - lo) / max(1, f - lo)        # 0 = oldest, 1 = now
                        if b_lo <= age < b_hi and i1 - i0 <= 3:
                            th = max(1, int(round((1 + 3 * age) * s)))
                            cv2.line(layer, _xy(p0), _xy(p1), HAND_PATH_COLOR, th,
                                     cv2.LINE_AA)
                            drew = True
                    if drew and alpha < 1:
                        cv2.addWeighted(layer, alpha, frame, 1 - alpha, 0, frame)
                cv2.circle(frame, _xy(pts[-1][1]), lw + 1, HAND_PATH_COLOR, -1,
                           cv2.LINE_AA)

            # Bat vector (knob -> tip).
            if bat_track is not None and getattr(bat_track, "available", False):
                knob, tip = bat_track.knob[f], bat_track.tip[f]
                if _ok(knob, tip):
                    cv2.line(frame, _xy(knob), _xy(tip), BAT_COLOR, lw + 1, cv2.LINE_AA)
                    cv2.circle(frame, _xy(tip), lw + 2, BAT_COLOR, -1, cv2.LINE_AA)

            # Ball + short trail.
            if ball_track is not None and getattr(ball_track, "available", False):
                for j in range(max(0, f - 6), f + 1):
                    c = ball_track.center[j]
                    if _ok(c):
                        r = 6 if j == f else 2
                        cv2.circle(frame, _xy(c), r, BALL_COLOR, -1, cv2.LINE_AA)

            _draw_hud(frame, f, m, view, s, smoothed,
                      in_swing=start <= f <= trace_end,
                      head_level=drift[f] if f < len(drift) else np.nan,
                      worst_level=float(np.nanmax(drift[start : trace_end + 1]))
                      if np.isfinite(drift[start : trace_end + 1]).any() else np.nan,
                      after_swing=f > trace_end)
        writer.write(frame)
        f += 1

    cap.release()
    writer.release()
    return _to_browser_h264(out_path) if web_safe else out_path


def _draw_hud(frame, f, m: SwingMetrics, view, s, smoothed, *, in_swing,
              head_level, worst_level, after_swing):
    """Readout panel (top-left), phase label (top-right), footer, contact flash."""
    h, w = frame.shape[:2]
    pad = int(12 * s)
    fs = 0.55 * s                                  # body font scale
    th = max(1, int(round(s)))
    line_h = int(24 * s)

    # --- readout panel: only what the camera angle can support ------------
    rows: list[tuple[str, str, tuple]] = []
    if view == "face":
        title = "FACE-ON"
        rows.append(("hip-shoulder sep", f"{m.separation[f]:+.0f} deg", TEXT_COLOR))
        if f >= m.max_separation_frame:
            rows.append(("peak pre-contact", f"{m.max_separation:+.0f} deg", MUTED_COLOR))
    elif view == "side":
        title = "SIDE-ON"
        if after_swing and np.isfinite(worst_level):
            state = "stable" if worst_level <= 1.0 else "drifted"
            rows.append(("head thru contact", state, _drift_color(worst_level)))
        elif in_swing and np.isfinite(head_level):
            state = "stable" if head_level <= 1.0 else "drifting"
            rows.append(("head", state, _drift_color(head_level)))
        else:
            rows.append(("head", "--", MUTED_COLOR))
        rows.append(("hand speed", f"{m.hand_speed_norm[f]:.0f} sw/s", TEXT_COLOR))
        rows.append(("sw = shoulder widths", "", MUTED_COLOR))
    else:
        title = "VIEW NOT SET"
        rows.append(("no graded readouts", "", MUTED_COLOR))

    label_w = max(_text_width(r[0], fs, th) for r in rows)
    value_w = max([_text_width(r[1], fs, th) for r in rows] + [0])
    panel_w = max(label_w + value_w + int(28 * s), _text_width(title, fs, th) + 2 * pad)
    panel_h = pad + line_h * (len(rows) + 1)
    _panel(frame, pad, pad, panel_w + pad, panel_h)
    x0, y = 2 * pad, pad + line_h
    _text(frame, title, (x0, y), fs * 0.9, MUTED_COLOR, th)
    for label, value, color in rows:
        y += line_h
        _text(frame, label, (x0, y), fs, MUTED_COLOR if value else color, th)
        if value:
            _text(frame, value, (x0 + label_w + int(16 * s), y), fs, color, th)

    # --- phase label, small caps in the top-right corner ------------------
    if abs(f - m.contact_frame) <= 1:
        phase, pcol = "CONTACT", CONTACT_COLOR
    elif f < m.contact_frame:
        phase, pcol = "PRE-CONTACT", TEXT_COLOR
    else:
        phase, pcol = "FOLLOW-THROUGH", TEXT_COLOR
    _text(frame, phase, (w - pad - _text_width(phase, fs, th), pad + line_h), fs, pcol, th)

    # --- footer: say what the drawing is ----------------------------------
    footer = "2D pose estimate, " + ("smoothed" if smoothed else "raw keypoints")
    _text(frame, footer, (pad, h - pad), fs * 0.8, MUTED_COLOR, th)

    # --- big CONTACT flash, top centre ------------------------------------
    if abs(f - m.contact_frame) <= 1:
        big = 1.2 * s
        bt = max(2, int(round(3 * s)))
        _text(frame, "CONTACT", ((w - _text_width("CONTACT", big, bt)) // 2,
                                 pad + int(44 * s)), big, CONTACT_COLOR, bt)


def render_plot(m: SwingMetrics, out_path: str) -> str:
    """Hip vs shoulder rotation, separation, and hand speed over time."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.arange(m.n_frames) / m.fps
    fig, ax = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    ax[0].plot(t, m.hip_angle - m.hip_angle[0], label="pelvis (hips)", color="#d1495b")
    ax[0].plot(t, m.shoulder_angle - m.shoulder_angle[0], label="torso (shoulders)",
               color="#3066be")
    ax[0].set_ylabel("rotation (deg, vs start)")
    ax[0].legend(loc="upper left")
    ax[0].set_title("Rotation — want hips to lead the shoulders")

    ax[1].plot(t, m.separation, color="#2a9d8f")
    ax[1].axhline(0, color="gray", lw=0.6)
    ax[1].set_ylabel("hip-shoulder\nseparation (deg)")
    ax[1].set_title(f"Separation (max {m.max_separation:+.0f}deg)")

    ax[2].plot(t, m.hand_speed_norm, color="#e76f51")
    ax[2].set_ylabel("hand speed\n(shoulder-widths/sec)")
    ax[2].set_xlabel("time (s)")
    ax[2].set_title(f"Hand speed (bat-speed proxy, peak {m.hand_speed_peak_norm:.0f})")

    for a in ax:
        a.axvline(m.contact_frame / m.fps, color="red", ls="--", lw=1, label="contact")
    ax[0].axvline(m.contact_frame / m.fps, color="red", ls="--", lw=1)

    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path
