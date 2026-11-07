"""Validate the templated coaching report (offline / no LLM path).

With no SWING_LAB_LLM_BASE_URL configured, coaching_report() must fall back to
a clean templated markdown built from the findings.

    PYTHONPATH=. .venv/bin/python tests/test_coach.py
"""

from __future__ import annotations

import os

import numpy as np

from swing_lab.coach import build_messages, coaching_report
from swing_lab.diagnose import diagnose
from swing_lab.metrics import compute
from swing_lab.pose import PoseSequence


def _pose(n=90) -> PoseSequence:
    kp = np.zeros((n, 17, 3), np.float32)
    kp[:, :, 2] = 0.9
    pts = {0: (160, 60), 5: (140, 90), 6: (180, 90), 9: (135, 120), 10: (185, 120),
           11: (145, 150), 12: (175, 150), 15: (145, 220), 16: (175, 220)}
    for i, (x, y) in pts.items():
        kp[:, i, 0], kp[:, i, 1] = x, y
    return PoseSequence(kp, 120.0, 320, 240)


def test_template_report():
    # Ensure no endpoint configured so we exercise the fallback deterministically.
    for k in ("SWING_LAB_LLM_BASE_URL", "SWING_LAB_LLM_MODEL"):
        os.environ.pop(k, None)

    m = compute(_pose())
    findings = diagnose(m)

    # Prompt builder produces a sane two-message chat payload.
    msgs = build_messages(m, findings, player="Alex")
    assert msgs[0]["role"] == "system" and msgs[1]["role"] == "user"
    assert "METRICS" in msgs[1]["content"] and "FINDINGS" in msgs[1]["content"]

    text, source = coaching_report(m, findings, player="Alex")
    assert source == "template", source
    assert "Swing Report — Alex" in text
    assert "Fix this first" in text
    print("Coach OK — source:", source)
    print("----- report -----")
    print(text)


if __name__ == "__main__":
    test_template_report()
    print("\nAll coach checks passed.")
