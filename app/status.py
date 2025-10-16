# humanoid_neurograph/app/status.py
from __future__ import annotations
from collections import deque
from typing import Deque

class StatusPanel:
    """Updates a single caption element; avoids duplicate lines."""
    def __init__(self, caption_placeholder):
        self._ph = caption_placeholder

    def update(
        self,
        fps_hist: Deque[float],
        blink_times: Deque[float],
        yaw_deg: float,
        emotion: str | None,
        pupil_text: str = ""
    ):
        fps = sum(fps_hist) / max(1, len(fps_hist))
        blinks = len(blink_times)  # already windowed upstream
        txt = (
            f"FPS: {fps:.1f} • Blink count (win): {blinks} • "
            f"Yaw: {yaw_deg:+.0f}° • Emotion: {emotion or '—'}"
        )
        if pupil_text:
            txt += f" • {pupil_text}"
        self._ph.caption(txt)
