"""Validate the web routes/templates without running the pose model.

We seed the library with two swings (so progress renders), drop dummy media
files in the swing's out_dir, and exercise every route via Flask's test client.

    PYTHONPATH=. .venv/bin/python tests/test_web.py
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from swing_lab import library
from swing_lab.web import create_app


def _report(player):
    return {
        "video": "x.mp4", "player": player,
        "metrics": {"contact_frame": 80, "contact_method": "hand-speed peak",
                    "attack_angle_deg": 6.0, "max_separation": -15.0,
                    "sequence_ok": 1, "bat_tracked": 0, "on_plane_ratio": None,
                    "bat_angle_at_contact_deg": None},
        "findings": [{"flaw": "Steep uppercut", "severity": "flag",
                      "confidence": "medium", "evidence": "attack 30deg",
                      "drill": "high tee"}],
        "coaching": "# Swing Report — Alex\n\nNice extension. Work the hips.",
    }


def test_web_routes():
    tmp = Path(tempfile.mkdtemp())
    db = str(tmp / "swings.db")
    out_dir = tmp / "swings" / "abc"
    out_dir.mkdir(parents=True)
    (out_dir / "metrics.png").write_bytes(b"\x89PNG\r\n\x1a\n")     # dummy
    (out_dir / "annotated.mp4").write_bytes(b"\x00\x00\x00\x18ftyp")  # dummy

    base = {"attack_angle_deg": 6.0, "max_separation": -10.0, "on_plane_ratio": 0.6,
            "head_drift_ratio": 0.7, "shoulder_tilt_deg": 20.0, "sequence_ok": 0}
    later = {"attack_angle_deg": 8.0, "max_separation": -18.0, "on_plane_ratio": 0.8,
             "head_drift_ratio": 0.4, "shoulder_tilt_deg": 10.0, "sequence_ok": 1}
    library.record("Alex", "a.mp4", base, db=db, created_at=1000)
    sid = library.record("Alex", "b.mp4", later, db=db, created_at=2000,
                         report=_report("Alex"), out_dir=str(out_dir))

    app = create_app(db=db, output_root=str(tmp))
    app.testing = True
    c = app.test_client()

    r = c.get("/")
    assert r.status_code == 200 and b"Alex" in r.data

    r = c.get(f"/swing/{sid}")
    assert r.status_code == 200
    assert b"Steep uppercut" in r.data and b"Coaching report" in r.data

    r = c.get("/player/Alex")
    assert r.status_code == 200 and b"Progress" in r.data

    r = c.get(f"/media/{sid}/metrics.png")
    assert r.status_code == 200 and r.data.startswith(b"\x89PNG")

    assert c.get("/swing/99999").status_code == 404
    assert c.get("/player/Nobody").status_code == 404

    # Upload validation (no real processing): missing fields -> redirect back.
    assert c.post("/upload", data={}).status_code == 302

    print("Web OK — index, swing, player, media, 404s, upload validation.")


def test_media_and_notes_with_relative_out_dir():
    # Real uploads store a relative out_dir, which used to 404 the video and plot.
    tmp = Path(tempfile.mkdtemp())
    (tmp / "output" / "swings" / "abc").mkdir(parents=True)
    (tmp / "output" / "swings" / "abc" / "metrics.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    report = _report("Alex")
    report["metrics"]["notes"] = ["Camera angle is not detected."]

    previous = os.getcwd()
    os.chdir(tmp)
    try:
        db = "swings.db"
        sid = library.record("Alex", "a.mp4", {"attack_angle_deg": 6.0}, db=db,
                             created_at=1000, report=report,
                             out_dir="output/swings/abc")
        c = create_app(db=db, output_root="output").test_client()
        assert c.get(f"/media/{sid}/metrics.png").status_code == 200
        assert b"Camera angle is not detected." in c.get(f"/swing/{sid}").data
    finally:
        os.chdir(previous)


if __name__ == "__main__":
    test_web_routes()
    test_media_and_notes_with_relative_out_dir()
    print("\nAll web checks passed.")
