"""Validate the player library (SQLite) and swing comparison.

    PYTHONPATH=. .venv/bin/python tests/test_library.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from swing_lab import compare, library


def test_library_and_compare():
    db = str(Path(tempfile.mkdtemp()) / "swings.db")

    base = {"max_separation": -10.0, "on_plane_ratio": 0.60, "head_drift_ratio": 0.70,
            "shoulder_tilt_deg": 20.0, "attack_angle_deg": 5.0, "sequence_ok": 0}
    better = {"max_separation": -20.0, "on_plane_ratio": 0.85, "head_drift_ratio": 0.40,
              "shoulder_tilt_deg": 10.0, "attack_angle_deg": 8.0, "sequence_ok": 1}

    assert library.count("Alex", db) == 0
    library.record("Alex", "april.mp4", base, db=db, created_at=1000)
    library.record("Alex", "june.mp4", better, db=db, created_at=2000)
    assert library.count("Alex", db) == 2

    bl = library.baseline("Alex", db)   # returns the metrics dict
    lt = library.latest("Alex", db)
    assert bl["max_separation"] == -10.0 and lt["max_separation"] == -20.0

    deltas = {d["key"]: d for d in compare.diff(bl, lt)}
    # Separation grew in magnitude (10 -> 20) => better.
    assert deltas["max_separation"]["verdict"] == "better", deltas["max_separation"]
    # On-plane up, head drift down, tilt down => all better.
    assert deltas["on_plane_ratio"]["verdict"] == "better"
    assert deltas["head_drift_ratio"]["verdict"] == "better"
    assert deltas["shoulder_tilt_deg"]["verdict"] == "better"

    print(compare.format_progress(bl, lt, "Alex"))
    print("\nLibrary OK — 2 swings, comparison verdicts correct.")


if __name__ == "__main__":
    test_library_and_compare()
    print("\nAll library checks passed.")
