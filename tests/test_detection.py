"""Low-detection clips must warn and must not present confident findings."""

from __future__ import annotations

from swing_lab.diagnose import Finding
from swing_lab.pipeline import detection_warning, gate_findings


def _finding() -> Finding:
    return Finding(flaw="Attack angle", severity="ok", confidence="high",
                   evidence="test", drill="test")


def test_good_detection_passes_through():
    assert detection_warning(0.95) is None
    assert detection_warning(0.70) is None
    assert gate_findings([_finding()], None)[0].confidence == "high"


def test_low_detection_warns_and_downgrades():
    warning = detection_warning(0.40)
    assert warning and "40%" in warning
    gated = gate_findings([_finding()], warning)
    assert gated[0].confidence == "low (poor detection)"
    assert gated[0].flaw == "Attack angle"


if __name__ == "__main__":
    test_good_detection_passes_through()
    test_low_detection_warns_and_downgrades()
    print("All detection checks passed.")
