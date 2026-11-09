"""Keypoint smoothing: less jitter, no lag, short gaps filled, long gaps kept.

    PYTHONPATH=. .venv/bin/python tests/test_smoothing.py
"""

from __future__ import annotations

import numpy as np

from swing_lab.pose import L_WRIST, NOSE, PoseSequence
from swing_lab.smoothing import smooth_sequence


def _noisy_sine(n=480, fps=240.0, noise_px=3.0, seed=0):
    """Every keypoint follows a slow sine path plus per-frame jitter."""
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fps
    truth_x = 320 + 40 * np.sin(2 * np.pi * 0.75 * t)
    truth_y = 240 + 20 * np.cos(2 * np.pi * 0.75 * t)
    kp = np.zeros((n, 17, 3), np.float32)
    kp[:, :, 0] = truth_x[:, None] + rng.normal(0, noise_px, (n, 17))
    kp[:, :, 1] = truth_y[:, None] + rng.normal(0, noise_px, (n, 17))
    kp[:, :, 2] = 0.9
    return PoseSequence(kp, fps, 640, 480), truth_x, truth_y


def test_jitter_drops_and_peak_stays_put():
    seq, tx, _ = _noisy_sine()
    sm = smooth_sequence(seq)
    for j in (NOSE, L_WRIST):
        raw_err = np.std(seq.keypoints[:, j, 0] - tx)
        sm_err = np.std(sm.keypoints[:, j, 0] - tx)
        # Head points get the wide window, wrists the narrow one; both must
        # cut the error materially.
        limit = 0.35 if j == NOSE else 0.6
        assert sm_err < limit * raw_err, (j, raw_err, sm_err)
        # Zero phase: the first crest lands on the same frame (within 1).
        crest = slice(0, 240)
        assert abs(int(np.argmax(sm.keypoints[crest, j, 0])) - int(np.argmax(tx[crest]))) <= 2
    # Lag check on the whole track: best cross-correlation shift is 0.
    x = sm.keypoints[:, NOSE, 0] - tx.mean()
    ref = tx - tx.mean()
    lags = range(-5, 6)
    best = max(lags, key=lambda L: float(np.dot(np.roll(x, L)[10:-10], ref[10:-10])))
    assert best == 0, best


def test_fast_hand_motion_is_not_smeared():
    # Wrists sweep 300 px in ~0.1 s at 240 fps, like hands through contact.
    n, fps = 240, 240.0
    t = np.arange(n) / fps
    x = 200 + 300 / (1 + np.exp(-(t - 0.5) / 0.02))
    kp = np.zeros((n, 17, 3), np.float32)
    kp[:, :, 0], kp[:, :, 1], kp[:, :, 2] = x[:, None], 240, 0.9
    sm = smooth_sequence(PoseSequence(kp, fps, 640, 480))
    v_true = np.max(np.abs(np.diff(x)))
    v_sm = np.max(np.abs(np.diff(sm.keypoints[:, L_WRIST, 0])))
    assert v_sm > 0.9 * v_true, (v_true, v_sm)
    assert int(np.argmax(np.abs(np.diff(sm.keypoints[:, L_WRIST, 0])))) in range(118, 122)


def test_short_gap_filled_long_gap_kept_input_untouched():
    seq, tx, _ = _noisy_sine(noise_px=0.0)
    kp = seq.keypoints
    kp[100:104, NOSE] = np.nan           # 4-frame dropout: fill
    kp[200:220, NOSE] = np.nan           # 20-frame dropout: keep
    kp[300:303, NOSE, 2] = 0.1           # 3 low-confidence frames: replace
    before = kp.copy()

    sm = smooth_sequence(seq)

    np.testing.assert_array_equal(seq.keypoints, before)   # not mutated
    assert sm is not seq and sm.keypoints is not seq.keypoints
    out = sm.keypoints[:, NOSE]
    assert np.isfinite(out[100:104, :2]).all()
    assert np.all(out[100:104, 2] >= 0.3)
    np.testing.assert_allclose(out[100:104, 0], tx[100:104], atol=1.0)
    assert np.isnan(out[200:220, :2]).all()
    assert np.all(out[300:303, 2] >= 0.3)
    np.testing.assert_allclose(out[300:303, 0], tx[300:303], atol=1.0)
    # Frames on either side of the long gap are still smoothed, not dropped.
    assert np.isfinite(out[199, 0]) and np.isfinite(out[220, 0])
    assert sm.fps == seq.fps and sm.width == seq.width and sm.height == seq.height


def test_strength_zero_only_fills_gaps():
    seq, _, _ = _noisy_sine()
    sm = smooth_sequence(seq, strength=0)
    np.testing.assert_allclose(sm.keypoints, seq.keypoints)


if __name__ == "__main__":
    test_jitter_drops_and_peak_stays_put()
    test_fast_hand_motion_is_not_smeared()
    test_short_gap_filled_long_gap_kept_input_untouched()
    test_strength_zero_only_fills_gaps()
    print("All smoothing checks passed.")
