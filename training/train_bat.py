#!/usr/bin/env python
"""Train the bat keypoint model (knob + tip).

Run from the swing_lab/ root once you've labeled and exported your dataset and
filled in `path` in training/data_bat.yaml:

    .venv/bin/python training/train_bat.py --epochs 100 --imgsz 1280

Notes:
- We fine-tune from COCO `yolo11n-pose.pt` — it already knows "pose," we just
  retrain it to find one 2-keypoint object (the bat).
- imgsz defaults high (1280) because a bat is thin and the tip is small; detail
  matters more here than for body pose.
- Best weights land in runs/pose/<name>/weights/best.pt. Point analyze.py at it:
    analyze.py swing.mp4 --bat-model runs/pose/bat/weights/best.pt
"""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the bat keypoint detector.")
    ap.add_argument("--data", default="training/data_bat.yaml")
    ap.add_argument("--base", default="yolo11n-pose.pt", help="starting weights")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="mps / cuda / cpu (auto if unset)")
    ap.add_argument("--name", default="bat")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.base)
    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, device=args.device, name=args.name,
    )
    print("Done. Best weights: runs/pose/%s/weights/best.pt" % args.name)


if __name__ == "__main__":
    main()
