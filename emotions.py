# humanoid_neurograph/app/emotions.py
from dataclasses import dataclass
from typing import Dict

@dataclass
class CogFlags:
    flat_affect: bool = False
    stress_high: bool = False
    blink_low: bool = False
    blink_high: bool = False
    low_pupil_reactivity: bool = False
    asymmetry: bool = False
    bradykinesia_hint: bool = False

def analyze_cognitive(feats, pupil_state, blink_count_window: int) -> Dict:
    """Heuristic screen for cognitive / neuro-psychiatric cues (demo-grade)."""
    flags = CogFlags()
    notes = []

    # Affect (very rough)
    if feats.emotion in ("neutral", "tired") and feats.speaking == 0 and feats.eyes_open > 0.4:
        flags.flat_affect = True; notes.append("Flat/neutral affect")

    # Stress
    if feats.stress > 0.65:
        flags.stress_high = True; notes.append("Elevated facial stress proxy")

    # Blink rate (window ~ few seconds)
    if blink_count_window <= 1:
        flags.blink_low = True; notes.append("Low blink rate")
    elif blink_count_window > 25:
        flags.blink_high = True; notes.append("High blink rate")

    # Pupil reactivity
    if abs(pupil_state.roc) < 0.02:
        flags.low_pupil_reactivity = True; notes.append("Low pupil reactivity")

    # Left/right asymmetry (eye openness difference as a quick proxy)
    # expect feats provides just one value; skip or extend when both available
    # (we could add: cheek redness asymm, shoulder height asymm from pose etc.)
    # For now: use head roll as hint of habitual tilt.
    if abs(feats.roll) > 12:
        flags.asymmetry = True; notes.append("Head tilt/asymmetry")

    # Bradykinesia hint: low hand activity + steady face
    if feats.left_hand == 0 and feats.right_hand == 0 and feats.speaking == 0 and feats.blink == 0:
        flags.bradykinesia_hint = True; notes.append("Low spontaneous movement")

    summary = " • ".join(notes) if notes else "No prominent risk cues detected."
    risk = min(1.0, 0.15*sum([
        flags.flat_affect, flags.stress_high, flags.blink_low, flags.blink_high,
        flags.low_pupil_reactivity, flags.asymmetry, flags.bradykinesia_hint
    ]))
    return dict(flags=flags, summary=summary, risk=risk)
