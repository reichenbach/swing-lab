"""Phase 1 — bat & ball tracking.

This is the differentiator over Phase 0's hands-as-proxy. Once you've labeled a
few hundred frames and trained the two small YOLO models (see ../../training/),
this package turns video into:

  - a **bat track** (knob + tip per frame) -> real bat path, true attack angle,
    on-plane time, bat speed, and confident rollover detection
  - a **ball track** (center per frame) -> contact frame from ball-bat proximity,
    contact-point depth, and timing

Everything degrades gracefully: with no trained weights the tracks come back
`available == False` and the rest of swing_lab falls straight back to Phase-0
behavior. Nothing here breaks the working pipeline before you have data.
"""

from .bat_ball import BallTrack, BatBallTracker, BatTrack, empty_tracks
from .contact import detect_contact
from .metrics_bat import BatMetrics, compute_bat_metrics

__all__ = [
    "BatBallTracker", "BatTrack", "BallTrack", "empty_tracks",
    "detect_contact", "BatMetrics", "compute_bat_metrics",
]
