#!/usr/bin/env python
"""swing_lab CLI — thin wrapper over swing_lab.pipeline.analyze_video.

    python analyze.py path/to/swing.mp4
    python analyze.py swing.mp4 --player Alex --save
    python analyze.py swing.mp4 --bat-model runs/pose/bat/weights/best.pt

Side-view, slow-motion (120-240fps) video of one hitter works best.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from swing_lab.diagnose import VIEWS
from swing_lab.pipeline import analyze_video


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze a baseball/softball swing video.")
    ap.add_argument("video", help="path to the swing video")
    ap.add_argument("--out", default=None, help="output dir (default: output/<videoname>)")
    ap.add_argument("--contact-frame", type=int, default=None,
                    help="override the auto-detected contact frame")
    ap.add_argument("--model", default="yolo11n-pose.pt",
                    help="YOLO pose weights (n/s/m; bigger = more accurate, slower)")
    ap.add_argument("--fps", type=float, default=None,
                    help="true capture fps (e.g. 120/240 for iPhone slo-mo). "
                         "Overrides the container's reported fps, which is often "
                         "wrong for exported slow-motion.")
    ap.add_argument("--view", choices=sorted(VIEWS), default=None,
                    help="camera angle. Only what this angle can show gets graded; "
                         "without it, numbers are shown with no verdict.")
    ap.add_argument("--bat-model", default=None,
                    help="trained bat keypoint weights (see training/)")
    ap.add_argument("--ball-model", default=None,
                    help="trained ball detector weights (see training/)")
    ap.add_argument("--no-smooth", action="store_true",
                    help="use raw per-frame keypoints (skip temporal smoothing)")
    ap.add_argument("--no-video", action="store_true", help="skip annotated-video render")
    ap.add_argument("--player", default=None, help="player name for the coaching report")
    ap.add_argument("--no-coach", action="store_true",
                    help="skip the plain-English coaching report")
    ap.add_argument("--save", action="store_true",
                    help="record this swing to the player library and show progress "
                         "vs baseline (requires --player)")
    args = ap.parse_args()

    video = Path(args.video)
    if not video.exists():
        raise SystemExit(f"No such video: {video}")
    out_dir = Path(args.out) if args.out else Path("output") / video.stem

    print(f"Analyzing {video.name} …")
    result = analyze_video(
        str(video), str(out_dir), pose_model=args.model,
        bat_model=args.bat_model, ball_model=args.ball_model,
        contact_frame=args.contact_frame, render_annotated=not args.no_video,
        coach=not args.no_coach, player=args.player, capture_fps=args.fps,
        view=args.view, smooth=not args.no_smooth,
    )
    m, findings = result.metrics, result.findings
    print(f"  {m.n_frames} frames @ {m.fps:.0f}fps, athlete detected "
          f"{result.detected_frac:.0%}; bat={result.bat_available} ball={result.ball_available}")
    if result.warning:
        print(f"  WARNING: {result.warning}")
    if result.coaching_source:
        print(f"  coaching report ({result.coaching_source}) -> coaching.md")

    # Save to player library + progress vs baseline.
    if args.save:
        if not args.player:
            print("  --save needs --player; skipping library record.")
        else:
            from swing_lab import compare, library

            prior = library.baseline(args.player)
            scalars = library.scalars_from_metrics(m)
            library.record(args.player, str(video), scalars,
                           report=result.report, out_dir=result.out_dir)
            if prior is not None:
                print("\n" + compare.format_progress(prior, scalars, args.player))
            else:
                print(f"  saved first baseline for {args.player}.")

    # Console summary.
    print("\n" + "=" * 64)
    print(f"SWING REPORT  —  {video.name}")
    print("=" * 64)
    print(f"  contact frame ~{m.contact_frame}  ({m.contact_frame / m.fps:.2f}s)  "
          f"[{m.contact_method}]")
    print(f"  attack angle   {m.attack_angle_deg:+.0f}deg   "
          f"max separation {m.max_separation:+.0f}deg   "
          f"sequence_ok={m.sequence_ok}")
    if m.bat_tracked:
        print(f"  bat path       angle@contact {m.bat_angle_at_contact_deg:+.0f}deg   "
              f"peak tip speed {m.bat_speed_peak_norm:.0f} sw/s   "
              f"on-plane {m.on_plane_ratio:.2f}")
    print("-" * 64)
    for f in findings:
        mark = {"ok": "✅", "watch": "🟡", "flag": "🔴"}.get(f.severity, "•")
        print(f"{mark} {f.flaw}  [{f.confidence}]")
        print(f"    why : {f.evidence}")
        if f.drill:
            print(f"    drill: {f.drill}")
    print("-" * 64)
    for note in m.notes:
        print(f"  note: {note}")
    print(f"\n  → {result.out_dir}/")


if __name__ == "__main__":
    main()
