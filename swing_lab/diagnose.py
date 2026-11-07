"""Diagnosis: map biomechanics numbers to common hitting flaws.

This is a transparent rules engine, NOT a black box. Every threshold lives in
THRESHOLDS below and is a *starting point* you calibrate against swings you
already know are good/bad. Encoding coaching knowledge as readable rules is the
right first move; an ML model trained on your labeled library comes later.

Honesty about confidence is built in. Some flaws (rolling over) genuinely need
bat tracking to call reliably — those findings say so rather than bluffing.
"""

from __future__ import annotations

from dataclasses import dataclass

from .metrics import SwingMetrics

# --- calibration knobs --- tweak these against your own known-good swings ----
THRESHOLDS = {
    "min_separation_deg": 12.0,     # below this = hips & torso firing together
    "uppercut_deg": 28.0,           # attack angle above => steep uppercut/pop-up risk
    "chop_deg": -8.0,               # attack angle below => chopping down / rollover risk
    "head_drift_max": 0.55,         # head travel as fraction of shoulder width
    "shoulder_tilt_max": 18.0,      # back-shoulder drop (deg) at contact
}


@dataclass
class Finding:
    flaw: str
    severity: str          # "ok" | "watch" | "flag"
    confidence: str        # "high" | "medium" | "low (needs bat tracking)"
    evidence: str
    drill: str


def diagnose(m: SwingMetrics, t: dict | None = None) -> list[Finding]:
    t = {**THRESHOLDS, **(t or {})}
    out: list[Finding] = []

    # 1) Hip usage / kinematic sequence -------------------------------------
    if not m.sequence_ok or abs(m.max_separation) < t["min_separation_deg"]:
        out.append(Finding(
            flaw="Limited hip lead / hip-shoulder separation",
            severity="flag",
            confidence="medium",
            evidence=(
                f"Peak order pelvis->torso->hands = "
                f"{m.pelvis_peak_frame}->{m.torso_peak_frame}->{m.hands_peak_frame} "
                f"(want ascending); max separation {abs(m.max_separation):.0f}deg "
                f"(want >{t['min_separation_deg']:.0f})."
            ),
            drill=(
                "Separation/coil drills: hook-em / 'show the back hip' tee work, "
                "step-back load. Cue: lower half starts while hands stay back."
            ),
        ))
    else:
        out.append(Finding(
            flaw="Hip lead / sequence",
            severity="ok",
            confidence="medium",
            evidence=(
                f"Sequence ascending and separation {abs(m.max_separation):.0f}deg "
                f">= {t['min_separation_deg']:.0f}deg."
            ),
            drill="Maintain — keep loading the back hip before the hands go.",
        ))

    # 2) Attack angle -> pop-up vs rollover ---------------------------------
    aa = m.attack_angle_deg
    if aa > t["uppercut_deg"]:
        out.append(Finding(
            flaw="Steep uppercut (pop-up / under-the-ball risk)",
            severity="flag", confidence="medium",
            evidence=f"Hand-path attack angle {aa:+.0f}deg > {t['uppercut_deg']:.0f}deg.",
            drill=(
                "High-tee / top-hand drives; cue 'stay above the ball', flatten "
                "the bat path. Check back-shoulder isn't dumping under."
            ),
        ))
    elif aa < t["chop_deg"]:
        if m.bat_tracked:
            conf = "high (bat tracked)"
            extra = (f" Bat path tracked; on-plane ratio into contact "
                     f"{m.on_plane_ratio:.2f}.")
        else:
            conf = "low (needs bat tracking)"
            extra = (" True rollover (early wrist roll) needs the Phase-1 bat "
                     "tracker to confirm — this is the hands-path proxy only.")
        out.append(Finding(
            flaw="Chopping down (rollover / weak grounder risk)",
            severity="flag",
            confidence=conf,
            evidence=(
                f"{'Bat' if m.bat_tracked else 'Hand'}-path attack angle "
                f"{aa:+.0f}deg < {t['chop_deg']:.0f}deg.{extra}"
            ),
            drill=(
                "Low-tee back-side oppo work; cue 'stay through the ball', palm-up/"
                "palm-down at contact. Hold finish to fight early wrist roll."
            ),
        ))
    else:
        out.append(Finding(
            flaw="Attack angle", severity="ok", confidence="medium",
            evidence=f"Hand-path attack angle {aa:+.0f}deg in productive window.",
            drill="On-plane — keep it.",
        ))

    # 3) Back-shoulder drop --------------------------------------------------
    if abs(m.shoulder_tilt_deg) > t["shoulder_tilt_max"]:
        out.append(Finding(
            flaw="Back-shoulder drop at contact",
            severity="watch", confidence="medium",
            evidence=(
                f"Shoulder line tilts {m.shoulder_tilt_deg:+.0f}deg vs setup "
                f"(> {t['shoulder_tilt_max']:.0f}deg)."
            ),
            drill="Level-shoulder turn; cue 'chest stays tall', no collapse on back side.",
        ))

    # 4) Head stability ------------------------------------------------------
    if m.head_drift_ratio > t["head_drift_max"]:
        out.append(Finding(
            flaw="Excess head movement",
            severity="watch", confidence="high",
            evidence=(
                f"Head travels {m.head_drift_ratio:.2f}x shoulder width "
                f"(> {t['head_drift_max']:.2f})."
            ),
            drill="Quiet-head load; cue 'see it on the bat', minimize drift toward pitcher.",
        ))

    # 5) Timing / contact ----------------------------------------------------
    if m.ball_tracked:
        out.append(Finding(
            flaw="Timing / contact",
            severity="watch", confidence="medium",
            evidence=(
                f"Contact at frame {m.contact_frame} ({m.contact_frame / m.fps:.2f}s) "
                f"via {m.contact_method}. Early/late vs pitch and contact-point "
                "depth come from the ball track over a full session of reps."
            ),
            drill="Front-toss timing ladders; chart contact depth across the session.",
        ))
    else:
        out.append(Finding(
            flaw="Timing vs pitch",
            severity="watch", confidence="low (needs ball tracking)",
            evidence=(
                f"Estimated contact at frame {m.contact_frame} "
                f"({m.contact_frame / m.fps:.2f}s) via {m.contact_method}. Judging "
                "early/late vs the pitch needs the Phase-1 ball track."
            ),
            drill="Front-toss timing ladders once ball tracking lands.",
        ))

    return out
