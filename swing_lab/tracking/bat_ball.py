"""Bat and ball trackers (Ultralytics YOLO).

Two small purpose-trained models:

  - **bat**  : a YOLO *pose* model, 1 class "bat", 2 keypoints [knob, tip].
               Two keypoints give us the bat as a vector -> orientation, path,
               and attack angle directly (a plain bbox can't).
  - **ball** : a YOLO *detect* model, 1 class "ball". Box center = position.

Neither model exists until you train it (see ../../training/). Until then call
`empty_tracks(n)` and the pipeline runs Phase-0 style.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class BatTrack:
    knob: np.ndarray   # [n, 2] xy px, NaN where missing
    tip: np.ndarray    # [n, 2]
    conf: np.ndarray   # [n]

    @property
    def available(self) -> bool:
        return bool(np.isfinite(self.tip).any())


@dataclass
class BallTrack:
    center: np.ndarray  # [n, 2] xy px, NaN where missing
    conf: np.ndarray    # [n]

    @property
    def available(self) -> bool:
        return bool(np.isfinite(self.center).any())


def empty_tracks(n: int) -> tuple[BatTrack, BallTrack]:
    """Unavailable tracks — the no-trained-model / no-detection case."""
    nan2 = np.full((n, 2), np.nan, dtype=np.float32)
    nan1 = np.zeros(n, dtype=np.float32)
    return BatTrack(nan2.copy(), nan2.copy(), nan1.copy()), BallTrack(nan2.copy(), nan1.copy())


class BatBallTracker:
    """Runs the trained bat (pose) and ball (detect) models over a video.

    Pass either/both weights paths. A missing/None weight simply yields an
    unavailable track for that object, so you can land the ball model first and
    add the bat model later without touching callers.
    """

    def __init__(self, bat_model: str | None = None, ball_model: str | None = None,
                 device: str | None = None, min_conf: float = 0.25):
        self.min_conf = min_conf
        self.device = device or self._auto_device()
        self.bat = self._load(bat_model)
        self.ball = self._load(ball_model)

    @staticmethod
    def _auto_device() -> str:
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"

    @staticmethod
    def _load(path: str | None):
        if not path or not Path(path).exists():
            return None
        from ultralytics import YOLO
        return YOLO(path)

    def track(self, video_path: str, n_frames: int) -> tuple[BatTrack, BallTrack]:
        bat, ball = empty_tracks(n_frames)
        if self.bat is not None:
            bat = self._track_bat(video_path, n_frames)
        if self.ball is not None:
            ball = self._track_ball(video_path, n_frames)
        return bat, ball

    # --- bat (pose, 2 keypoints) -------------------------------------------
    def _track_bat(self, video_path: str, n: int) -> BatTrack:
        knob = np.full((n, 2), np.nan, np.float32)
        tip = np.full((n, 2), np.nan, np.float32)
        conf = np.zeros(n, np.float32)
        results = self.bat.predict(source=video_path, stream=True,
                                   device=self.device, verbose=False, conf=self.min_conf)
        for f, r in enumerate(results):
            if f >= n or r.keypoints is None or len(r.keypoints) == 0:
                continue
            # Highest-confidence bat detection this frame.
            boxconf = r.boxes.conf.cpu().numpy() if r.boxes is not None else None
            idx = int(np.argmax(boxconf)) if boxconf is not None and len(boxconf) else 0
            kp = r.keypoints.data.cpu().numpy()[idx]  # [2, 3] -> knob, tip
            if kp.shape[0] >= 2:
                knob[f] = kp[0, :2]
                tip[f] = kp[1, :2]
                conf[f] = float(boxconf[idx]) if boxconf is not None else 1.0
        return BatTrack(knob, tip, conf)

    # --- ball (detect) -----------------------------------------------------
    def _track_ball(self, video_path: str, n: int) -> BallTrack:
        center = np.full((n, 2), np.nan, np.float32)
        conf = np.zeros(n, np.float32)
        results = self.ball.predict(source=video_path, stream=True,
                                    device=self.device, verbose=False, conf=self.min_conf)
        for f, r in enumerate(results):
            if f >= n or r.boxes is None or len(r.boxes) == 0:
                continue
            boxconf = r.boxes.conf.cpu().numpy()
            idx = int(np.argmax(boxconf))
            x, y = r.boxes.xywh.cpu().numpy()[idx, :2]
            center[f] = [x, y]
            conf[f] = float(boxconf[idx])
        return BallTrack(center, conf)
