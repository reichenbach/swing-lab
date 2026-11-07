#!/usr/bin/env python
"""Sample frames from swing videos for labeling.

Step 1 of building the bat/ball training set: pull still frames out of your
clips so you can annotate them. Sampling every Nth frame (rather than every
frame) gives you variety without labeling near-duplicates.

    .venv/bin/python tools/extract_frames.py data/ --every 3 --out data/frames

Then label data/frames/ in your annotation tool (see training/README.md).
"""

from __future__ import annotations

import argparse
from pathlib import Path

VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi"}


def iter_videos(src: Path):
    if src.is_file():
        yield src
    else:
        for p in sorted(src.rglob("*")):
            if p.suffix.lower() in VIDEO_EXTS:
                yield p


def main() -> None:
    ap = argparse.ArgumentParser(description="Extract frames from videos for labeling.")
    ap.add_argument("src", help="video file or a folder of videos")
    ap.add_argument("--out", default="data/frames", help="output image folder")
    ap.add_argument("--every", type=int, default=3, help="keep every Nth frame")
    ap.add_argument("--max-per-video", type=int, default=400)
    args = ap.parse_args()

    import cv2

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    total = 0
    for video in iter_videos(Path(args.src)):
        cap = cv2.VideoCapture(str(video))
        f = kept = 0
        while kept < args.max_per_video:
            ok, frame = cap.read()
            if not ok:
                break
            if f % args.every == 0:
                name = f"{video.stem}_{f:05d}.jpg"
                cv2.imwrite(str(out / name), frame)
                kept += 1
            f += 1
        cap.release()
        total += kept
        print(f"  {video.name}: {kept} frames")
    print(f"Extracted {total} frames -> {out}/")


if __name__ == "__main__":
    main()
