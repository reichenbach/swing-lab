"""Findings may only carry a verdict the camera angle can support."""

from __future__ import annotations

import os

import numpy as np

from swing_lab.coach import coaching_report
from swing_lab.diagnose import diagnose
from swing_lab.metrics import compute
from swing_lab.pipeline import detection_warning, gate_findings
from swing_lab.pose import PoseSequence


def _metrics():
    kp = np.zeros((90, 17, 3), np.float32)
    kp[:, :, 2] = 0.9
    pts = {0: (160, 60), 5: (140, 90), 6: (180, 90), 9: (135, 120), 10: (185, 120),
           11: (145, 150), 12: (175, 150), 15: (145, 220), 16: (175, 220)}
    for i, (x, y) in pts.items():
        kp[:, i, 0], kp[:, i, 1] = x, y
    m = compute(PoseSequence(kp, 120.0, 320, 240))
    m.attack_angle_deg = -30.0    # would be flagged as chopping down
    m.max_separation = 40.0       # would pass as a good hip lead
    m.sequence_ok = True
    m.head_drift_ratio = 2.0      # would be flagged as excess head movement
    return m


def _by_flaw(findings, text):
    return next(f for f in findings if text in f.flaw)


def test_unknown_view_grades_nothing():
    findings = diagnose(_metrics())
    assert {f.severity for f in findings} == {"info"}
    assert not any(f.drill for f in findings)
    assert "-30deg" in _by_flaw(findings, "Attack angle").evidence


def test_face_on_grades_hips_only():
    findings = diagnose(_metrics(), view="face")
    assert _by_flaw(findings, "Hip lead").severity == "ok"
    attack = _by_flaw(findings, "Attack angle")
    assert attack.severity == "info" and "side-on" in attack.evidence
    assert "productive" not in attack.evidence
    assert _by_flaw(findings, "Head movement").severity == "info"


def test_side_on_grades_attack_angle_and_head_only():
    findings = diagnose(_metrics(), view="side")
    assert _by_flaw(findings, "Chopping").severity == "flag"
    assert _by_flaw(findings, "Excess head").severity == "watch"
    assert _by_flaw(findings, "Hip lead").severity == "info"


def _report(findings, metrics):
    for k in ("SWING_LAB_LLM_BASE_URL", "SWING_LAB_LLM_MODEL"):
        os.environ.pop(k, None)
    return coaching_report(metrics, findings)[0]


def test_ungraded_clip_has_no_fix_this_first():
    m = _metrics()
    text = _report(diagnose(m), m)
    assert "Fix this first" not in text
    assert "could be graded" in text and "Measured, not judged" in text


def test_timing_without_ball_tracking_is_never_the_priority():
    m = _metrics()
    m.max_separation = 40.0
    text = _report(diagnose(m, view="face"), m)
    assert "Fix this first" not in text
    assert "What's working" in text


def test_real_issue_outranks_low_confidence_flag():
    m = _metrics()
    # Chopping down is a low-confidence flag; excess head movement is a firm watch.
    text = _report(diagnose(m, view="side"), m)
    assert text.index("Fix this first") < text.index("Excess head movement")
    assert "Chopping down" in text.split("Worth a look")[1]


def test_poor_detection_removes_the_priority_but_keeps_not_judged():
    m = _metrics()
    warning = detection_warning(0.2)
    findings = gate_findings(diagnose(m, view="side"), warning)
    assert "Fix this first" not in _report(findings, m)
    timing = _by_flaw(findings, "Timing")
    assert timing.confidence == "not judged"


if __name__ == "__main__":
    test_unknown_view_grades_nothing()
    test_face_on_grades_hips_only()
    test_side_on_grades_attack_angle_and_head_only()
    test_ungraded_clip_has_no_fix_this_first()
    test_timing_without_ball_tracking_is_never_the_priority()
    test_real_issue_outranks_low_confidence_flag()
    test_poor_detection_removes_the_priority_but_keeps_not_judged()
    print("All view checks passed.")
