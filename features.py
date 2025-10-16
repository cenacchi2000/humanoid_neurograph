# humanoid_neurograph/app/features.py
from dataclasses import dataclass

@dataclass
class LiveFeatures:
    eyes_open: float = 0.0
    blink: float = 0.0
    pupil_diam: float = 0.0
    mouth_open: float = 0.0
    speaking: float = 0.0
    yaw: float = 0.0
    pitch: float = 0.0
    roll: float = 0.0
    left_hand: float = 0.0
    right_hand: float = 0.0
    tremor_hz: float = 0.0
    stress: float = 0.0
    emotion: str = "neutral"
