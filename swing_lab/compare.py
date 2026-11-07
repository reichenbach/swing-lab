"""Compare two swings — a swing vs the player's baseline, or any two reports.

"Is the change sticking?" is the question a coach actually has. This diffs the
scalar metrics and labels each delta as better / worse / flat using a small
table of which direction is good and how big a move is meaningful.
"""

from __future__ import annotations

import math

# For each metric: (good_direction, meaningful_delta, label).
# good_direction: +1 means higher is better, -1 lower is better, 0 neutral.
# Speeds are the *_norm (shoulder-widths/sec) versions, not raw px/frame — raw
# pixel speeds aren't comparable across sessions (camera distance/zoom/fps drift)
# and would make progress tracking lie. See metrics.py.
TREND = {
    "max_separation": (+1, 4.0, "hip-shoulder separation (deg, abs)"),
    "on_plane_ratio": (+1, 0.10, "on-plane ratio"),
    "hand_speed_peak_norm": (+1, 1.5, "peak hand speed (shoulder-widths/sec)"),
    "bat_speed_peak_norm": (+1, 1.5, "peak bat-tip speed (shoulder-widths/sec)"),
    "head_drift_ratio": (-1, 0.10, "head drift (x shoulder width)"),
    "shoulder_tilt_deg": (-1, 4.0, "back-shoulder tilt (deg, abs)"),
    "attack_angle_deg": (0, 3.0, "attack angle (deg)"),
}

# Metrics where we compare absolute value (magnitude), not signed value.
_ABS = {"max_separation", "shoulder_tilt_deg"}


def diff(baseline: dict, latest: dict) -> list[dict]:
    out = []
    for key, (good, thresh, label) in TREND.items():
        a, b = baseline.get(key), latest.get(key)
        if a is None or b is None:
            continue
        try:
            a, b = float(a), float(b)
        except (TypeError, ValueError):
            continue
        if math.isnan(a) or math.isnan(b):
            continue  # NaN = metric not available (e.g. bat not tracked yet)
        if key in _ABS:
            a, b = abs(a), abs(b)
        delta = b - a
        if abs(delta) < thresh:
            verdict = "flat"
        elif good == 0:
            verdict = "changed"
        else:
            verdict = "better" if (delta * good) > 0 else "worse"
        out.append({"key": key, "label": label, "from": a, "to": b,
                    "delta": delta, "verdict": verdict})
    return out


def format_progress(baseline: dict, latest: dict, player: str | None = None) -> str:
    mark = {"better": "↑", "worse": "↓", "flat": "·", "changed": "~"}
    lines = [f"Progress{' — ' + player if player else ''} (baseline → latest):"]
    for d in diff(baseline, latest):
        lines.append(
            f"  {mark[d['verdict']]} {d['label']}: "
            f"{d['from']:+.2f} → {d['to']:+.2f}  ({d['delta']:+.2f}, {d['verdict']})"
        )
    if len(lines) == 1:
        lines.append("  (no comparable metrics yet)")
    return "\n".join(lines)
