"""Pose estimation backends.

A backend reads a video and returns a PoseSequence: for every frame, the 17
COCO keypoints (x, y, confidence) of the *primary athlete* — the hitter — with
other people in frame filtered out by continuity tracking.

COCO-17 keypoint indices (Ultralytics pose order):
    0 nose        5 L_shoulder   9 L_wrist    13 L_knee
    1 L_eye       6 R_shoulder  10 R_wrist    14 R_knee
    2 R_eye       7 L_elbow     11 L_hip      15 L_ankle
    3 L_ear       8 R_elbow     12 R_hip      16 R_ankle
    4 R_ear
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# COCO-17 index constants — use these instead of magic numbers everywhere.
NOSE = 0
L_EAR, R_EAR = 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

KEYPOINT_NAMES = [
    "nose", "L_eye", "R_eye", "L_ear", "R_ear",
    "L_shoulder", "R_shoulder", "L_elbow", "R_elbow",
    "L_wrist", "R_wrist", "L_hip", "R_hip",
    "L_knee", "R_knee", "L_ankle", "R_ankle",
]

# Skeleton edges for drawing.
SKELETON = [
    (L_SHOULDER, R_SHOULDER), (L_HIP, R_HIP),
    (L_SHOULDER, L_HIP), (R_SHOULDER, R_HIP),
    (L_SHOULDER, L_ELBOW), (L_ELBOW, L_WRIST),
    (R_SHOULDER, R_ELBOW), (R_ELBOW, R_WRIST),
    (L_HIP, L_KNEE), (L_KNEE, L_ANKLE),
    (R_HIP, R_KNEE), (R_KNEE, R_ANKLE),
]


@dataclass
class PoseSequence:
    """Per-frame keypoints for the primary athlete.

    keypoints: float array [n_frames, 17, 3] -> (x_px, y_px, confidence).
               NaN x/y means the athlete was not detected that frame.
    """

    keypoints: np.ndarray
    fps: float
    width: int
    height: int

    @property
    def n_frames(self) -> int:
        return self.keypoints.shape[0]


class YoloPoseBackend:
    """Ultralytics YOLO11-pose. Weights auto-download on first run."""

    def __init__(self, model: str = "yolo11n-pose.pt", device: str | None = None,
                 min_conf: float = 0.3):
        from ultralytics import YOLO

        self.model = YOLO(model)
        self.min_conf = min_conf
        self.device = device or self._auto_device()

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

    def estimate(self, video_path: str, verbose: bool = False,
                 capture_fps: float | None = None) -> PoseSequence:
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")
        # iPhone slow-mo exports often report the *container* rate (e.g. 30)
        # while the real capture was 120/240 — which silently corrupts every
        # time-based metric (contact window, per-second speeds). When the user
        # tells us the true capture fps, trust it over the container metadata.
        fps = float(capture_fps) if capture_fps else (cap.get(cv2.CAP_PROP_FPS) or 30.0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        results = self.model.predict(
            source=video_path, stream=True, device=self.device,
            verbose=verbose, conf=self.min_conf,
        )

        frames: list[np.ndarray] = []
        prev_centroid: np.ndarray | None = None
        for r in results:
            kp = self._select_primary(r, prev_centroid)
            if kp is not None:
                prev_centroid = self._hip_centroid(kp)
            frames.append(
                kp if kp is not None else np.full((17, 3), np.nan, dtype=np.float32)
            )

        return PoseSequence(np.stack(frames), float(fps), width, height)

    # --- primary-athlete selection ------------------------------------------
    def _select_primary(self, result, prev_centroid) -> np.ndarray | None:
        """Pick one person per frame.

        First frame (no history): largest bounding box — the hitter is closest
        to camera in a typical BP setup. After that: track by nearest hip
        centroid so we don't jump to a passing teammate.
        """
        if result.keypoints is None or len(result.keypoints) == 0:
            return None
        data = result.keypoints.data.cpu().numpy()  # [n_people, 17, 3]
        if data.shape[0] == 0:
            return None

        if prev_centroid is None:
            boxes = result.boxes.xywh.cpu().numpy() if result.boxes is not None else None
            if boxes is not None and len(boxes):
                areas = boxes[:, 2] * boxes[:, 3]
                return data[int(np.argmax(areas))]
            return data[0]

        centroids = np.array([self._hip_centroid(p) for p in data])
        dists = np.linalg.norm(centroids - prev_centroid, axis=1)
        return data[int(np.argmin(dists))]

    @staticmethod
    def _hip_centroid(kp: np.ndarray) -> np.ndarray:
        return np.nanmean(kp[[L_HIP, R_HIP], :2], axis=0)


def get_backend(name: str = "yolo", **kwargs):
    if name == "yolo":
        return YoloPoseBackend(**kwargs)
    raise ValueError(f"Unknown pose backend: {name!r} (only 'yolo' so far)")
