"""The analysis pipeline as one reusable call.

Both the CLI (analyze.py) and the web app (web/app.py) go through
`analyze_video()` so there's a single code path: pose -> optional bat/ball
tracking -> metrics -> diagnosis -> render -> coaching -> serializable report.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .coach import coaching_report
from .diagnose import Finding, diagnose
from .metrics import SwingMetrics, compute
from .overlay import render_plot, render_video
from .pose import get_backend


# Starting point, not calibrated on real footage: below this share of frames with
# a detected athlete, the numbers are not trustworthy enough to present plainly.
LOW_DETECTION_THRESHOLD = 0.70


def detection_warning(detected_frac: float,
                      threshold: float = LOW_DETECTION_THRESHOLD) -> str | None:
    """Return a user-facing warning when too few frames had a detected athlete."""
    if detected_frac >= threshold:
        return None
    return (f"Low detection: the athlete was found in only {detected_frac:.0%} of "
            f"frames (below {threshold:.0%}). Treat every number and finding from "
            "this clip as unreliable, and re-film with one hitter fully in frame.")


def gate_findings(findings: list[Finding], warning: str | None) -> list[Finding]:
    """Drop every finding to low confidence when detection was poor."""
    if not warning:
        return findings
    return [dataclasses.replace(f, confidence="low (poor detection)") for f in findings]


@dataclass
class AnalysisResult:
    metrics: SwingMetrics
    findings: list[Finding]
    out_dir: str
    detected_frac: float
    warning: str | None = None
    bat_available: bool = False
    ball_available: bool = False
    coaching_text: str | None = None
    coaching_source: str | None = None
    outputs: dict = field(default_factory=dict)
    report: dict = field(default_factory=dict)


def _scalar_metrics(m: SwingMetrics) -> dict:
    md = dataclasses.asdict(m)
    for k in ("hip_angle", "shoulder_angle", "separation", "hand_speed",
              "hand_speed_norm"):
        md.pop(k, None)
    return md


def analyze_video(video: str, out_dir: str, *, pose_model: str = "yolo11n-pose.pt",
                  bat_model: str | None = None, ball_model: str | None = None,
                  contact_frame: int | None = None, render_annotated: bool = True,
                  coach: bool = True, player: str | None = None,
                  capture_fps: float | None = None) -> AnalysisResult:
    video = str(video)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    seq = get_backend("yolo", model=pose_model).estimate(video, capture_fps=capture_fps)
    detected = float((~np.isnan(seq.keypoints[:, 0, 0])).mean()) if seq.n_frames else 0.0

    bat_track = ball_track = None
    if bat_model or ball_model:
        from .tracking import BatBallTracker

        tracker = BatBallTracker(bat_model=bat_model, ball_model=ball_model)
        bat_track, ball_track = tracker.track(video, seq.n_frames)

    m = compute(seq, contact_frame=contact_frame,
                bat_track=bat_track, ball_track=ball_track)
    warning = detection_warning(detected)
    findings = gate_findings(diagnose(m), warning)

    outputs: dict = {"plot": None, "annotated_video": None, "coaching": None}
    outputs["plot"] = render_plot(m, str(out / "metrics.png"))
    if render_annotated:
        outputs["annotated_video"] = render_video(
            seq, m, video, str(out / "annotated.mp4"),
            bat_track=bat_track, ball_track=ball_track,
        )

    coaching_text = coaching_source = None
    if coach:
        coaching_text, coaching_source = coaching_report(m, findings, player=player)
        if warning:
            coaching_text = f"**{warning}**\n\n{coaching_text}"
        coach_path = out / "coaching.md"
        coach_path.write_text(coaching_text)
        outputs["coaching"] = str(coach_path)

    report = {
        "video": video,
        "player": player,
        "detected_frac": detected,
        "warning": warning,
        "metrics": _scalar_metrics(m),
        "findings": [dataclasses.asdict(f) for f in findings],
        "coaching": coaching_text,
        "outputs": outputs,
    }
    (out / "report.json").write_text(json.dumps(report, indent=2, default=float))

    return AnalysisResult(
        metrics=m, findings=findings, out_dir=str(out), detected_frac=detected,
        warning=warning,
        bat_available=bool(bat_track and bat_track.available),
        ball_available=bool(ball_track and ball_track.available),
        coaching_text=coaching_text, coaching_source=coaching_source,
        outputs=outputs, report=report,
    )
