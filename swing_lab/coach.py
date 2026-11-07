"""Phase 2 — turn metrics + findings into a plain-English coaching report.

A coach doesn't want angles; they want "here's what's happening and here are
two drills." This module builds that report.

LLM is optional and provider-agnostic: it calls any OpenAI-compatible
/chat/completions endpoint, configured by env vars so it works with a
self-hosted llama.cpp/Jetson server *or* OpenAI/Gemini's OpenAI-compat API:

    SWING_LAB_LLM_BASE_URL   e.g. http://localhost:8080/v1  (llama.cpp server)
    SWING_LAB_LLM_MODEL      model name the endpoint expects
    SWING_LAB_LLM_API_KEY    optional bearer token

If no endpoint is configured or the call fails, it falls back to a clean
templated report built from the findings — so this works offline today.
"""

from __future__ import annotations

import os

from .diagnose import Finding
from .metrics import SwingMetrics

SYSTEM_PROMPT = (
    "You are an encouraging youth baseball hitting coach. You translate "
    "swing-analysis data into clear, kind, actionable feedback a 13-18 year old "
    "and their parent can use. No jargon dumps. Be specific and positive. "
    "Always end with concrete drills and simple cues."
)


def _metrics_block(m: SwingMetrics) -> str:
    lines = [
        f"- fps: {m.fps:.0f}",
        f"- contact frame: {m.contact_frame} ({m.contact_method})",
        f"- attack angle: {m.attack_angle_deg:+.0f} deg "
        f"({'tracked bat path' if m.bat_tracked else 'hand-path proxy'})",
        f"- max hip-shoulder separation: {m.max_separation:+.0f} deg",
        f"- kinematic sequence correct (hips->torso->hands): {m.sequence_ok}",
        f"- head drift (x shoulder width): {m.head_drift_ratio:.2f}",
        f"- back-shoulder tilt at contact: {m.shoulder_tilt_deg:+.0f} deg",
    ]
    if m.bat_tracked:
        lines += [
            f"- bat angle at contact: {m.bat_angle_at_contact_deg:+.0f} deg",
            f"- on-plane ratio into contact: {m.on_plane_ratio:.2f}",
            f"- peak bat-tip speed: {m.bat_speed_peak_norm:.0f} shoulder-widths/sec",
        ]
    return "\n".join(lines)


def _findings_block(findings: list[Finding]) -> str:
    out = []
    for f in findings:
        out.append(f"- [{f.severity}/{f.confidence}] {f.flaw}: {f.evidence}")
    return "\n".join(out)


def build_messages(m: SwingMetrics, findings: list[Finding],
                   player: str | None = None) -> list[dict]:
    who = f" for {player}" if player else ""
    user = (
        f"Write a short hitting report{who} from this swing analysis.\n\n"
        f"METRICS:\n{_metrics_block(m)}\n\n"
        f"AUTOMATED FINDINGS:\n{_findings_block(findings)}\n\n"
        "Structure:\n"
        "1. One upbeat sentence on what looks good.\n"
        "2. The single most important thing to fix first, in plain language.\n"
        "3. 2-3 drills with a one-line cue each.\n"
        "Keep it under ~200 words. Note any low-confidence items honestly."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def _llm_report(messages: list[dict]) -> str | None:
    base = os.environ.get("SWING_LAB_LLM_BASE_URL")
    model = os.environ.get("SWING_LAB_LLM_MODEL")
    if not base or not model:
        return None
    try:
        import requests

        headers = {"Content-Type": "application/json"}
        key = os.environ.get("SWING_LAB_LLM_API_KEY")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        resp = requests.post(
            f"{base.rstrip('/')}/chat/completions",
            json={"model": model, "messages": messages, "temperature": 0.4},
            headers=headers, timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:  # offline / misconfigured / endpoint down -> fallback
        print(f"      (LLM report unavailable: {e}; using template)")
        return None


def _template_report(m: SwingMetrics, findings: list[Finding],
                     player: str | None) -> str:
    name = player or "Hitter"
    goods = [f for f in findings if f.severity == "ok"]
    flags = [f for f in findings if f.severity == "flag"]
    watches = [f for f in findings if f.severity == "watch"]

    lines = [f"# Swing Report — {name}", ""]
    if goods:
        lines += ["**What's working:** " + "; ".join(g.flaw for g in goods), ""]

    priority = flags[0] if flags else (watches[0] if watches else None)
    if priority:
        lines += [
            "## Fix this first",
            f"**{priority.flaw}** ({priority.confidence})",
            f"- Why: {priority.evidence}",
            f"- Drill: {priority.drill}",
            "",
        ]

    rest = [f for f in (flags + watches) if f is not priority]
    if rest:
        lines += ["## Also work on"]
        for f in rest:
            lines += [f"- **{f.flaw}** — {f.drill}"]
        lines += [""]

    lines += ["---", "_Auto-generated from swing metrics. "
              "Low-confidence items improve once bat/ball tracking is trained._"]
    return "\n".join(lines)


def coaching_report(m: SwingMetrics, findings: list[Finding],
                    player: str | None = None, use_llm: bool = True) -> tuple[str, str]:
    """Return (report_markdown, source). source is 'llm:<model>' or 'template'."""
    if use_llm:
        text = _llm_report(build_messages(m, findings, player))
        if text:
            return text, f"llm:{os.environ.get('SWING_LAB_LLM_MODEL')}"
    return _template_report(m, findings, player), "template"
