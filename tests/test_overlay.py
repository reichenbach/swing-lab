"""Overlay v2 renders on a blank synthetic video for every view setting.

    PYTHONPATH=. .venv/bin/python tests/test_overlay.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_pipeline import _synthetic_swing  # noqa: E402

from swing_lab.metrics import compute  # noqa: E402
from swing_lab.overlay import render_video  # noqa: E402
from swing_lab.smoothing import smooth_sequence  # noqa: E402


def _blank_video(path: Path, n: int, w: int, h: int, fps: float) -> None:
    import cv2

    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    frame = np.full((h, w, 3), 60, np.uint8)
    for _ in range(n):
        writer.write(frame)
    writer.release()


def _read_frames(path: Path) -> list[np.ndarray]:
    import cv2

    cap = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        frames.append(fr)
    cap.release()
    return frames


def test_overlay_renders_for_each_view():
    seq = smooth_sequence(_synthetic_swing())
    seq.keypoints[5:8] = np.nan        # a missed detection must not crash drawing
    m = compute(seq)
    out = Path(tempfile.mkdtemp())
    src = out / "blank.mp4"
    _blank_video(src, seq.n_frames, seq.width, seq.height, 30.0)

    for view in (None, "face", "side"):
        dst = out / f"annotated_{view}.mp4"
        path = render_video(seq, m, str(src), str(dst), view=view, smoothed=True,
                            web_safe=False)
        frames = _read_frames(Path(path))
        assert len(frames) == seq.n_frames, (view, len(frames))
        contact = frames[m.contact_frame]
        assert contact.std() > 5, view                 # something was drawn
        # The CONTACT flash is red; make sure it shows on the contact frame.
        red = (contact[:, :, 2] > 150) & (contact[:, :, 1] < 90)
        assert red.sum() > 50, (view, red.sum())


def test_old_signature_still_works():
    seq = _synthetic_swing(n=40)
    m = compute(seq)
    out = Path(tempfile.mkdtemp())
    src = out / "blank.mp4"
    _blank_video(src, seq.n_frames, seq.width, seq.height, 30.0)
    path = render_video(seq, m, str(src), str(out / "a.mp4"), None, None, False)
    assert Path(path).stat().st_size > 0


def test_pipeline_smooths_by_default_and_reports_it():
    import json

    from swing_lab import pipeline

    seq = _synthetic_swing(n=60)
    out = Path(tempfile.mkdtemp())
    src = out / "blank.mp4"
    _blank_video(src, seq.n_frames, seq.width, seq.height, 30.0)

    class _Backend:
        def estimate(self, video, capture_fps=None):
            return seq

    import inspect

    assert inspect.signature(pipeline.analyze_video).parameters["smooth"].default is True
    real = pipeline.get_backend
    pipeline.get_backend = lambda *a, **k: _Backend()
    try:
        for smooth in (True, False):
            res = pipeline.analyze_video(str(src), str(out / str(smooth)), coach=False,
                                         view="side", smooth=smooth)
            report = json.loads((out / str(smooth) / "report.json").read_text())
            assert report["smoothing"] is smooth
            assert Path(res.outputs["annotated_video"]).exists()
    finally:
        pipeline.get_backend = real


if __name__ == "__main__":
    test_overlay_renders_for_each_view()
    test_old_signature_still_works()
    test_pipeline_smooths_by_default_and_reports_it()
    print("All overlay checks passed.")
