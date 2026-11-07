#!/usr/bin/env python
"""Train the ball detector.

    .venv/bin/python training/train_ball.py --epochs 100 --imgsz 1280

The ball is small and fast, so high imgsz + high-fps source footage matter.
Best weights -> runs/detect/<name>/weights/best.pt:
    analyze.py swing.mp4 --ball-model runs/detect/ball/weights/best.pt
"""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the ball detector.")
    ap.add_argument("--data", default="training/data_ball.yaml")
    ap.add_argument("--base", default="yolo11n.pt", help="starting weights")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default=None, help="mps / cuda / cpu (auto if unset)")
    ap.add_argument("--name", default="ball")
    args = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(args.base)
    model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, device=args.device, name=args.name,
    )
    print("Done. Best weights: runs/detect/%s/weights/best.pt" % args.name)


if __name__ == "__main__":
    main()
