"""Temporal keypoint smoothing for a PoseSequence.

Raw per-frame pose estimates jitter by a few pixels even when the athlete is
still. Drawn as-is, a head box twitches and a hand path zig-zags, which reads as
measurement precision the data doesn't have. This module cleans the sequence
once, after pose estimation, so metrics and the overlay both see the same
steadier signal.

Two steps, per keypoint and per coordinate:

1. **Fill short gaps.** A keypoint missing (NaN) or below `min_conf` for at most
   `max_gap` frames is linearly interpolated between its neighbors and given the
   lower of the two neighbors' confidences. Longer gaps are left alone: guessing
   through a long dropout would invent motion.

2. **Zero-phase local quadratic fit.** Each frame's value is replaced by a
   confidence-weighted quadratic fitted over a centered window (a weighted
   Savitzky-Golay filter, written in numpy because scipy is not a dependency).
   The window is symmetric, so there is no lag: a peak stays on its frame.
   A quadratic follows curvature that a moving average would flatten, and
   low-confidence points get proportionally less say in the fit.

Window sizes are set in **seconds** and converted with `seq.fps`, so a 240 fps
clip gets a window eight times as many frames wide as a 30 fps clip.

Joints get different windows on purpose:

- Nose, eyes, and ears (`HEAD_WINDOW_S`) barely move during a good swing, so
  they get the widest window. This is what steadies the head box.
- Torso, hips, and legs (`CORE_WINDOW_S`) rotate quickly but not as fast as the
  hands, so they get a medium window.
- Elbows and wrists (`LIMB_WINDOW_S`) get a window under half as wide. The
  hands cover most of their path in the last ~0.15 s before contact, and a wide
  window would cut the peak hand speed and round off the path through contact.
  At 30 fps their window rounds down to nothing, so they are left raw: there
  the hands move far more per frame than they jitter.

`strength` scales all three windows. 1.0 is the default; 0 turns smoothing off but
still fills short gaps.
"""

from __future__ import annotations

import numpy as np

from .pose import L_EAR, L_ELBOW, L_WRIST, NOSE, R_EAR, R_ELBOW, R_WRIST, PoseSequence

# Half-window widths in seconds at strength 1.0.
HEAD_WINDOW_S = 0.15
CORE_WINDOW_S = 0.10
LIMB_WINDOW_S = 0.04
HEAD_KEYPOINTS = (NOSE, 1, 2, L_EAR, R_EAR)  # nose, eyes, ears
LIMB_KEYPOINTS = (L_ELBOW, R_ELBOW, L_WRIST, R_WRIST)

# Same confidence cutoff the metrics and overlay use.
MIN_CONF = 0.3
MAX_GAP_FRAMES = 5


def smooth_sequence(seq: PoseSequence, strength: float = 1.0,
                    max_gap: int = MAX_GAP_FRAMES,
                    min_conf: float = MIN_CONF) -> PoseSequence:
    """Return a new, smoothed PoseSequence. The input is not modified."""
    kp = np.array(seq.keypoints, dtype=np.float64, copy=True)
    for j in range(kp.shape[1]):
        _fill_short_gaps(kp[:, j], max_gap, min_conf)
        window_s = _window_s(j)
        half = int(round(window_s * strength * seq.fps))
        if half >= 2:  # a quadratic through fewer than 5 points smooths nothing
            for c in (0, 1):
                kp[:, j, c] = _smooth_track(kp[:, j, c], kp[:, j, 2], half, min_conf)
    return PoseSequence(kp.astype(seq.keypoints.dtype), seq.fps, seq.width, seq.height)


def _window_s(j: int) -> float:
    if j in HEAD_KEYPOINTS:
        return HEAD_WINDOW_S
    if j in LIMB_KEYPOINTS:
        return LIMB_WINDOW_S
    return CORE_WINDOW_S


def _valid(xyc: np.ndarray, min_conf: float) -> np.ndarray:
    conf = xyc[:, 2]
    return np.isfinite(xyc[:, :2]).all(axis=1) & np.isfinite(conf) & (conf >= min_conf)


def _fill_short_gaps(xyc: np.ndarray, max_gap: int, min_conf: float) -> None:
    """In place on one keypoint's [n, 3] track: interpolate gaps <= max_gap."""
    good = _valid(xyc, min_conf)
    idx = np.flatnonzero(good)
    for a, b in zip(idx[:-1], idx[1:]):
        gap = b - a - 1
        if 0 < gap <= max_gap:
            w = (np.arange(1, gap + 1) / (gap + 1))[:, None]
            xyc[a + 1 : b, :2] = (1 - w) * xyc[a, :2] + w * xyc[b, :2]
            xyc[a + 1 : b, 2] = min(xyc[a, 2], xyc[b, 2])


def _smooth_track(y: np.ndarray, conf: np.ndarray, half: int,
                  min_conf: float) -> np.ndarray:
    """Weighted local quadratic fit over each contiguous run of valid frames.

    Runs are fitted separately so a fit never reaches across a long gap. Frames
    outside any run (long gaps, low confidence) are returned unchanged.
    """
    out = y.copy()
    good = np.isfinite(y) & np.isfinite(conf) & (conf >= min_conf)
    edges = np.flatnonzero(np.diff(np.concatenate([[0], good.astype(int), [0]])))
    for start, stop in zip(edges[::2], edges[1::2]):
        out[start:stop] = _local_quadratic(y[start:stop], conf[start:stop], half)
    return out


def _local_quadratic(y: np.ndarray, w: np.ndarray, half: int) -> np.ndarray:
    """Evaluate a weighted quadratic fit centered on every sample.

    For each frame t, minimise sum_k w[t+k] (y[t+k] - (c0 + c1 k + c2 k^2))^2
    over k in [-half, half] and return c0. The weighted moments that form the
    normal equations are correlations of the data with k^p, so the whole track
    is solved at once. Near the ends the window simply holds fewer samples.
    """
    n = len(y)
    if n < 3:
        return y.copy()
    k = np.arange(-half, half + 1, dtype=np.float64)
    wp = np.pad(w, half)
    wyp = np.pad(w * y, half)

    def corr(sig, p):
        return np.correlate(sig, k ** p, mode="valid")

    s = [corr(wp, p) for p in range(5)]   # sum w k^p, p = 0..4
    t = [corr(wyp, p) for p in range(3)]  # sum w y k^p, p = 0..2

    a = np.stack([np.stack([s[0], s[1], s[2]], -1),
                  np.stack([s[1], s[2], s[3]], -1),
                  np.stack([s[2], s[3], s[4]], -1)], -2)       # [n, 3, 3]
    rhs = np.stack(t, -1)                                      # [n, 3]

    out = y.copy()
    # The fit is well posed when the 3x3 system isn't near-singular, which needs
    # at least three weighted samples. Scale-free check via the condition number.
    ok = np.linalg.cond(a) < 1e10
    if ok.any():
        out[ok] = np.linalg.solve(a[ok], rhs[ok][..., None])[:, 0, 0]
    return out
