"""swing_lab — markerless baseball/softball swing analysis from video.

Phase-0 pipeline:
    video -> pose estimation -> biomechanics features -> flaw diagnosis
          -> annotated video + metric plots + JSON report

Pose backend is pluggable (see pose.py). Default is Ultralytics YOLO11-pose,
which installs cleanly on Python 3.13 / Apple Silicon and is the same framework
we'll use for the Phase-1 bat detector. A MediaPipe backend (lighter, great on
Jetson) is the intended drop-in swap later.
"""

__version__ = "0.0.1"
