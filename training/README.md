# Phase 1 — Bat & Ball Tracking: data + training

This is the moat. Phase 0 infers bat path from the hands; Phase 1 tracks the
**actual bat** (knob + tip) and the **ball**, which unlocks true attack angle,
on-plane time, confident rollover detection, real contact frame, and timing.

You can't train without footage. This folder is the recipe to follow the moment
you have clips. Land the **ball** model first if you want quick wins (contact +
timing); the **bat** model is more labeling effort but higher payoff.

## The pipeline

```
clips → extract frames → label → export YOLO → train → point analyze.py at weights
```

### 1. Collect clips
Drop swing videos in `swing_lab/data/`. Side-on and face-on both fine for
detection. Higher fps (120–240) = the ball moves less between frames = far
easier to track. Aim for variety: different kids, lighting, backgrounds, bat
colors.

### 2. Extract frames to label
```bash
cd swing_lab
.venv/bin/python tools/extract_frames.py data/ --every 3 --out data/frames
```
Target a few hundred labeled frames per model to start; grow from there. The
frames around the swing (load → contact → extension) are the ones that matter —
label those densely.

### 3. Label
Use any tool that exports **YOLO** format — [Roboflow](https://roboflow.com)
(easiest, has keypoint support), CVAT, or Label Studio.

- **Ball** → object detection. One class `ball`, a tight box around the ball.
- **Bat** → keypoints. One class `bat` with **2 keypoints in this order**:
  `0 = knob` (bottom of handle), `1 = tip` (end of barrel). Mark a keypoint
  not-visible when occluded rather than guessing.

Schemas are defined in `data_bat.yaml` / `data_ball.yaml`.

### 4. Export
Export YOLO format into `data/bat_dataset/` and `data/ball_dataset/` with:
```
images/train  images/val
labels/train  labels/val
```
Then set the `path:` field in each YAML to that dataset root.

### 5. Train
```bash
.venv/bin/python training/train_ball.py --epochs 100 --imgsz 1280
.venv/bin/python training/train_bat.py  --epochs 100 --imgsz 1280
```
Weights land in `runs/detect/ball/weights/best.pt` and
`runs/pose/bat/weights/best.pt`. (`runs/` is gitignored.)

### 6. Use it
```bash
.venv/bin/python analyze.py data/swing.mp4 \
    --ball-model runs/detect/ball/weights/best.pt \
    --bat-model  runs/pose/bat/weights/best.pt
```
With weights present, the report switches from hand-path proxies to the tracked
bat path: rollover confidence goes from *low* to *high*, contact comes from
ball-bat proximity, and you get bat angle / on-plane / bat speed. With **no**
weights, `analyze.py` runs exactly as in Phase 0 — nothing breaks while you
build the dataset.

## How the code consumes the models
- `swing_lab/tracking/bat_ball.py` — `BatBallTracker` loads the weights and
  returns per-frame `BatTrack` (knob/tip) and `BallTrack` (center).
- `swing_lab/tracking/contact.py` — contact frame from ball-bat proximity.
- `swing_lab/tracking/metrics_bat.py` — attack angle, bat angle, bat speed,
  on-plane ratio from the bat vector.

These feed straight into `metrics.compute(..., bat_track=, ball_track=)` and
`diagnose()`, which are already wired to use them when available.

## Tips
- **Bat speed in px/frame** becomes **mph** once you add a known reference
  length in frame (plate width, bat length) for pixel→inch scaling — a small
  later addition.
- Start small, label iteratively: train on 200 frames, run on new clips, and
  label the frames the model got wrong (active learning) — fastest path to a
  solid model.
