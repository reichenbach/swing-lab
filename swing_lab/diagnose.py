"""Diagnosis: map biomechanics numbers to common hitting flaws.

This is a transparent rules engine, NOT a black box. Every threshold lives in
THRESHOLDS below and is a *starting point* you calibrate against swings you
already know are good/bad. Encoding coaching knowledge as readable rules is the
right first move; an ML model trained on your labeled library comes later.

Honesty about confidence is built in. Some flaws (rolling over) genuinely need
bat tracking to call reliably — those findings say so rather than bluffing.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

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
    severity: str          # "ok" | "watch" | "flag" | "info" (shown, not judged)
    confidence: str        # "high" | "medium" | "low (...)" | "not judged"
    evidence: str
    drill: str


# A single camera can only judge some things. Each rule maps findings to the view
# that can support them, a neutral label, and the plain measurement to show when
# the view can't, so a number never carries a verdict it hasn't earned.
VIEWS = {"side": "side-on", "face": "face-on"}
_VIEW_RULES = (
    (("Limited hip lead", "Hip lead"), "face", "Hip lead / sequence",
     lambda m: (f"Separation {abs(m.max_separation):.0f}deg; pelvis, torso and hands "
                f"peak at frames {m.pelvis_peak_frame}, {m.torso_peak_frame} and "
                f"{m.hands_peak_frame}.")),
    (("Steep uppercut", "Chopping down", "Attack angle"), "side", "Attack angle",
     lambda m: f"{'Bat' if m.bat_tracked else 'Hand'}-path angle {m.attack_angle_deg:+.0f}deg."),
    (("Back-shoulder drop",), "side", "Back-shoulder drop",
     lambda m: f"Shoulder line moved {m.shoulder_tilt_deg:+.0f}deg vs setup."),
    (("Excess head movement",), "side", "Head movement",
     lambda m: f"Head travels {m.head_drift_ratio:.2f}x shoulder width."),
)


def _apply_view(f: Finding, m: SwingMetrics, view: str | None) -> Finding:
    """Turn a finding into a plain measurement if this camera can't support it."""
    for prefixes, needs, label, measure in _VIEW_RULES:
        if f.flaw.startswith(prefixes) and view != needs:
            why = ("this clip's camera angle wasn't set" if view is None
                   else f"this clip was marked {VIEWS[view]}")
            return replace(f, flaw=label, severity="info", confidence="not judged",
                           evidence=(f"{measure(m)} Only meaningful from a "
                                     f"{VIEWS[needs]} camera; {why}."),
                           drill="")
    return f


def diagnose(m: SwingMetrics, t: dict | None = None,
             view: str | None = None) -> list[Finding]:
    """Judge a swing. `view` is "side", "face", or None if the angle is unknown."""
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
            extra = (" True rollover (early wrist roll) needs bat tracking to "
                     "confirm; this uses the hand path only.")
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
            severity="info", confidence="not judged",
            evidence=(
                f"Estimated contact at frame {m.contact_frame} "
                f"({m.contact_frame / m.fps:.2f}s) via {m.contact_method}. Judging "
                "early or late vs the pitch needs ball tracking."
            ),
            drill="",
        ))

    return [_apply_view(f, m, view) for f in out]
