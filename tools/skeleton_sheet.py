#!/usr/bin/env python
"""Draw detected poses from a clip on a plain background.

Shares what the tool sees without publishing any pixels from the source video.

    PYTHONPATH=. python tools/skeleton_sheet.py swing.mp4 docs/swing.png \
        --start 92 --end 125 --step 3 --mark 113 --mark-label "visible contact"
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from swing_lab.pose import get_backend

COCO_EDGES = [
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("video")
    ap.add_argument("out")
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--step", type=int, default=3)
    ap.add_argument("--mark", type=int, default=None, help="frame to highlight")
    ap.add_argument("--mark-label", default="")
    ap.add_argument("--model", default="yolo11n-pose.pt")
    ap.add_argument("--columns", type=int, default=6)
    args = ap.parse_args()

    kp = get_backend("yolo", model=args.model).estimate(args.video).keypoints
    frames = list(range(args.start, args.end + 1, args.step))
    selected = kp[frames][:, :, :2]
    centers = np.nanmedian(kp[frames][:, [11, 12], 0], axis=1, keepdims=True)
    half_w = max(float(np.nanmax(np.abs(selected[:, :, 0] - centers))), 1.0) * 1.05
    y_low, y_high = np.nanmin(selected[:, :, 1]), np.nanmax(selected[:, :, 1])
    pad = (y_high - y_low) * 0.08

    rows = int(np.ceil(len(frames) / args.columns))
    fig, axes = plt.subplots(rows, args.columns,
                             figsize=(1.7 * args.columns, 3.0 * rows),
                             facecolor="#11151c", squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, f in zip(axes.flat, frames):
        marked = f == args.mark
        color = "#ff6b5b" if marked else "#5fd38d"
        center = np.nanmedian(kp[f, [11, 12], 0])
        for a, b in COCO_EDGES:
            if kp[f, a, 2] >= 0.3 and kp[f, b, 2] >= 0.3:
                ax.plot(kp[f, [a, b], 0], kp[f, [a, b], 1], color=color, lw=2)
        ax.set_xlim(center - half_w, center + half_w)
        ax.set_ylim(y_high + pad, y_low - pad)
        ax.set_aspect("equal")
        title = f"frame {f}" + (f"\n{args.mark_label}" if marked and args.mark_label else "")
        ax.set_title(title, color=color if marked else "#c8ced8", fontsize=9)
    fig.tight_layout()
    fig.savefig(args.out, dpi=130, facecolor=fig.get_facecolor())
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
