# app/pupil.py
from __future__ import annotations
from dataclasses import dataclass
from collections import deque
from typing import Optional, Deque, Tuple
import time
import numpy as np
import cv2

@dataclass
class PupilState:
    event: str = "steady"          # 'dilate' | 'constrict' | 'steady'
    zscore: float = 0.0
    roc: float = 0.0               # rate of change / s
    cognitive_index: float = 0.0

class PupilAnalyzer:
    """
    Keeps a small rolling window of normalized pupil diameter and emits a state.
    Also draws perfect rings centered on iris centers (from detector).
    """
    def __init__(self, fps_hint: float = 30.0, win_sec: float = 6.0):
        self.fs = float(max(1.0, fps_hint))
        self.maxlen = int(max(20, win_sec * self.fs))
        self.times: Deque[float] = deque(maxlen=self.maxlen)
        self.values: Deque[float] = deque(maxlen=self.maxlen)
        self.last_t: Optional[float] = None
        self.last_v: Optional[float] = None

    # --- analysis ---
    def update(self, pupil_diam_norm: Optional[float], t_now: float) -> PupilState:
        t_now = float(t_now)
        if pupil_diam_norm is None:
            return PupilState()

        v = float(np.clip(pupil_diam_norm, 0.0, 1.0))
        self.times.append(t_now)
        self.values.append(v)

        if self.last_t is None:
            self.last_t, self.last_v = t_now, v

        # z-score in rolling window
        arr = np.asarray(self.values, dtype=np.float32)
        mu = float(arr.mean())
        sd = float(arr.std() + 1e-6)
        z = (v - mu) / sd

        # rate of change
        dt = max(1e-3, t_now - self.last_t)
        roc = (v - self.last_v) / dt
        self.last_t, self.last_v = t_now, v

        # simple cognitive index (stable narrow pupils -> lower; sustained dilation -> higher)
        cog = float(np.clip(0.5 * max(0.0, z) + 2.0 * max(0.0, roc), 0.0, 1.0))
        event = "dilate" if roc > +0.08 else ("constrict" if roc < -0.08 else "steady")
        return PupilState(event=event, zscore=z, roc=roc, cognitive_index=cog)

    # --- drawing ---
    @staticmethod
    def draw(frame_bgr,
             centerL: Optional[Tuple[int, int]] = None,
             centerR: Optional[Tuple[int, int]] = None,
             rL: Optional[float] = None,
             rR: Optional[float] = None,
             state: Optional[PupilState] = None,
             **kwargs) -> None:
        """
        Draw precise rings using iris centers/radii. If radii are None, draw small ring.
        Text is drawn bottom-right once per frame, no repetitions.
        Extra kwargs are ignored for backward-compatibility.
        """
        H, W = frame_bgr.shape[:2]

        def ring(c, r):
            if c is None: return
            x, y = int(c[0]), int(c[1])
            rr = int(max(2.0, (r if r is not None else 6.0) * 1.05))
            cv2.circle(frame_bgr, (x, y), rr, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(frame_bgr, (x, y), 2, (255, 255, 255), -1, cv2.LINE_AA)

        ring(centerL, rL)
        ring(centerR, rR)

        if state is not None:
            text = f"Pupil {state.event}   z={state.zscore:+.1f}   roc={state.roc:+.02f}/s   cog={state.cognitive_index:.2f}"
            # bottom-right HUD (single, non-overlapping)
            pad = 8
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            x = max(0, W - tw - 2 * pad)
            y = max(th + pad, H - pad)
            box_tl = (x - pad, y - th - int(0.6*pad))
            box_br = (x + tw + pad, y + int(0.6*pad))
            cv2.rectangle(frame_bgr, box_tl, box_br, (0, 0, 0), -1)
            cv2.putText(frame_bgr, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
