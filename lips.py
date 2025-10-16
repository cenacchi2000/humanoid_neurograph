# app/lips.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np
import cv2

Pt = Tuple[int, int]

@dataclass
class LipsState:
    ok: bool = False
    label: str = "unknown"
    gap_px: float = 0.0
    width_px: float = 1.0
    cheek_asym: float = 0.0  # |L-R|/avg
    is_speaking: bool = False

class LipsAnalyzer:
    """
    Super light-weight lips & cheeks analyzer.
    Needs these points inside meta["points"]:
      - "lipUp"(13), "lipDn"(14), "lipL"(61), "lipR"(291)
      - "cheekL"(234), "cheekR"(454)
    """

    def __init__(self, target_fs: float = 30.0):
        self.target_fs = target_fs
        # thresholds (tweak if you like)
        self.shut_thr = 0.06
        self.speech_low = 0.06
        self.speech_high = 0.22
        self.wide_thr = 0.28
        self.yawn_thr = 0.38

    @staticmethod
    def _get(meta: Dict, name: str) -> Optional[Pt]:
        p = meta.get("points", {}).get(name)
        return (int(p[0]), int(p[1])) if p is not None else None

    def update(self, frame_bgr, meta: Dict, t: float) -> LipsState:
        lu = self._get(meta, "lipUp")
        ld = self._get(meta, "lipDn")
        ll = self._get(meta, "lipL")
        lr = self._get(meta, "lipR")
        cl = self._get(meta, "cheekL")
        cr = self._get(meta, "cheekR")

        st = LipsState(ok=False)
        if None in (lu, ld, ll, lr) or frame_bgr is None:
            return st

        # geometry
        gap = float(abs(ld[1] - lu[1]))
        width = float(max(1, abs(lr[0] - ll[0])))
        gap_norm = gap / width

        # cheeks asymmetry (proxy)
        if cl and cr:
            st.cheek_asym = abs(float(cl[1] - cr[1])) / (0.5 * (abs(cl[1]) + abs(cr[1]) + 1e-3))

        # classify
        if gap_norm < self.shut_thr:
            label = "shut"
        elif self.speech_low <= gap_norm <= self.speech_high:
            label = "speaking"
        elif gap_norm > self.yawn_thr:
            label = "yawn"
        elif gap_norm > self.wide_thr:
            label = "wide open"
        else:
            label = "open"

        st.ok = True
        st.label = label
        st.is_speaking = (label == "speaking")
        st.gap_px = gap
        st.width_px = width
        return st

    @staticmethod
    def draw(img, st: LipsState, meta: Dict) -> "np.ndarray":
        if img is None or not st or not st.ok:
            return img

        lu = meta["points"].get("lipUp")
        ld = meta["points"].get("lipDn")
        ll = meta["points"].get("lipL")
        lr = meta["points"].get("lipR")
        cl = meta["points"].get("cheekL")
        cr = meta["points"].get("cheekR")

        if None in (lu, ld, ll, lr):
            return img

        lu = tuple(map(int, lu)); ld = tuple(map(int, ld))
        ll = tuple(map(int, ll)); lr = tuple(map(int, lr))

        # mouth ellipse (center and radii)
        cx = int(0.5 * (ll[0] + lr[0]))
        cy = int(0.5 * (lu[1] + ld[1]))
        rx = max(4, int(0.5 * abs(lr[0] - ll[0])))
        ry = max(3, int(0.55 * abs(ld[1] - lu[1]) + 4))

        color = (60, 220, 240) if st.is_speaking else (240, 220, 60)
        cv2.ellipse(img, (cx, cy), (rx, ry), 0, 0, 360, color, 2, cv2.LINE_AA)

        # cheeks
        if cl is not None:
            cv2.circle(img, (int(cl[0]), int(cl[1])), 8, (90, 255, 120), 2, cv2.LINE_AA)
        if cr is not None:
            cv2.circle(img, (int(cr[0]), int(cr[1])), 8, (90, 255, 120), 2, cv2.LINE_AA)

        # label
        tag = f"Lips: {st.label}  gap/w={st.gap_px/max(1,st.width_px):.2f}  asym={st.cheek_asym:.2f}"
        cv2.putText(img, tag, (max(3, cx - rx), cy + ry + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.56, color, 2, cv2.LINE_AA)
        return img
