"""Rendering: annotated skeleton video + a metrics plot the kid can look at."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

from .metrics import SwingMetrics
from .pose import SKELETON, PoseSequence


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


def render_video(seq: PoseSequence, m: SwingMetrics, in_path: str, out_path: str,
                 bat_track=None, ball_track=None, web_safe: bool = True) -> str:
    """Draw the tracked skeleton (+ bat/ball when available) on each frame.

    bat_track/ball_track are optional Phase-1 tracks; when present the bat line
    (knob→tip, magenta) and the ball (yellow) are drawn too. A short trail of
    recent ball positions is drawn to make the pitch/contact readable.

    web_safe=True transcodes the result to browser-playable H.264 via ffmpeg
    (see `_to_browser_h264`). Set False to skip the transcode for a faster
    CLI-only render.
    """
    import cv2

    def _xy(p):
        return (int(p[0]), int(p[1]))

    def _ok(*pts):
        return all(np.isfinite(p).all() for p in pts)

    cap = cv2.VideoCapture(in_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, seq.fps, (seq.width, seq.height))

    f = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if f < seq.n_frames:
            kp = seq.keypoints[f]
            for a, b in SKELETON:
                pa, pb = kp[a], kp[b]
                if pa[2] > 0.3 and pb[2] > 0.3:
                    cv2.line(frame, _xy(pa), _xy(pb), (0, 255, 0), 2)
            for x, y, c in kp:
                if c > 0.3:
                    cv2.circle(frame, (int(x), int(y)), 3, (0, 200, 255), -1)

            # Bat vector (knob -> tip).
            if bat_track is not None and getattr(bat_track, "available", False):
                knob, tip = bat_track.knob[f], bat_track.tip[f]
                if _ok(knob, tip):
                    cv2.line(frame, _xy(knob), _xy(tip), (255, 0, 255), 3)
                    cv2.circle(frame, _xy(tip), 5, (255, 0, 255), -1)

            # Ball + short trail.
            if ball_track is not None and getattr(ball_track, "available", False):
                for k in range(max(0, f - 6), f + 1):
                    c = ball_track.center[k]
                    if _ok(c):
                        r = 6 if k == f else 2
                        cv2.circle(frame, _xy(c), r, (0, 255, 255), -1)

            if abs(f - m.contact_frame) <= 1:
                cv2.putText(frame, "CONTACT", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        writer.write(frame)
        f += 1

    cap.release()
    writer.release()
    return _to_browser_h264(out_path) if web_safe else out_path


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
