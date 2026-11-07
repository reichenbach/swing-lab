"""Validate the analysis math without needing a real video.

We synthesize a hitter's keypoints for a right-handed swing where the hips lead
the shoulders (good sequence) and the hands swing slightly up through contact,
then push the same arrays through compute() -> diagnose() -> render_plot().

Run:  .venv/bin/python -m pytest tests/ -q
   or .venv/bin/python tests/test_pipeline.py
"""

from __future__ import annotations

import numpy as np

from swing_lab.diagnose import diagnose
from swing_lab.metrics import compute
from swing_lab.overlay import render_plot
from swing_lab.pose import PoseSequence


def _synthetic_swing(n=120, fps=120.0) -> PoseSequence:
    """Build a believable side-view swing as COCO-17 keypoints [n,17,3]."""
    kp = np.zeros((n, 17, 3), dtype=np.float32)
    kp[:, :, 2] = 0.9  # all confident

    t = np.linspace(0, 1, n)
    contact = int(0.7 * n)

    # Pelvis rotates first, torso lags then catches up (good sequence).
    hip_rot = 40 * _ramp(t, start=0.45, end=0.70)        # degrees
    sh_rot = 40 * _ramp(t, start=0.55, end=0.78)         # starts later

    cx, cy = 320.0, 240.0
    sh_w, hip_w = 70.0, 55.0

    for i in range(n):
        # Face-on view: rotation tilts the shoulder/hip line in the image plane.
        lsx, lsy = _rot(-sh_w / 2, 0, sh_rot[i])
        lhx, lhy = _rot(-hip_w / 2, 0, hip_rot[i])
        kp[i, 5] = [cx + lsx, cy - 60 + lsy, 0.9]    # L_shoulder
        kp[i, 6] = [cx - lsx, cy - 60 - lsy, 0.9]    # R_shoulder
        kp[i, 11] = [cx + lhx, cy + 10 + lhy, 0.9]   # L_hip
        kp[i, 12] = [cx - lhx, cy + 10 - lhy, 0.9]   # R_hip

        # Hands: swing forward (x) and slightly up (y down decreases) at contact.
        prog = _ramp(t, 0.5, 0.75)[i]
        hx = cx - 30 + 120 * prog
        hy = cy - 20 - 25 * prog            # rising => uppercut-ish
        kp[i, 9] = [hx, hy, 0.9]            # L_wrist
        kp[i, 10] = [hx + 8, hy + 4, 0.9]  # R_wrist

        kp[i, 0] = [cx - 5, cy - 110, 0.9]                  # nose (quiet head)
        kp[i, 15] = [cx - 40, cy + 150, 0.9]                # L_ankle
        kp[i, 16] = [cx + 40, cy + 150, 0.9]                # R_ankle

    return PoseSequence(kp, fps, 640, 480)


def _ramp(t, start, end):
    return np.clip((t - start) / (end - start), 0, 1)


def _cos(deg):
    return float(np.cos(np.radians(deg)))


def _rot(x, y, deg):
    r = np.radians(deg)
    return x * np.cos(r) - y * np.sin(r), x * np.sin(r) + y * np.cos(r)


def test_pipeline_runs_and_is_sane(tmp_path=None):
    import tempfile
    from pathlib import Path

    out = Path(tempfile.mkdtemp()) if tmp_path is None else tmp_path
    seq = _synthetic_swing()

    m = compute(seq)
    assert 0 <= m.contact_frame < seq.n_frames
    assert np.isfinite(m.attack_angle_deg)
    assert np.isfinite(m.max_separation)
    # Hands rise through contact in our synthetic swing -> positive attack angle.
    assert m.attack_angle_deg > 0, m.attack_angle_deg

    findings = diagnose(m)
    assert len(findings) >= 3  # sequence + attack angle + timing always present
    flaws = {f.flaw for f in findings}
    assert any("sequence" in f.lower() or "hip" in f.lower() for f in flaws)

    plot = render_plot(m, str(out / "metrics.png"))
    assert Path(plot).exists() and Path(plot).stat().st_size > 0

    print("OK — contact frame", m.contact_frame,
          "| attack", round(m.attack_angle_deg, 1),
          "| max sep", round(m.max_separation, 1),
          "| sequence_ok", m.sequence_ok)
    print("Findings:")
    for f in findings:
        print(f"  [{f.severity}] {f.flaw}")


if __name__ == "__main__":
    test_pipeline_runs_and_is_sane()
    print("\nAll checks passed.")
