# app/attention.py
from __future__ import annotations
import cv2
import numpy as np

# Fallback-friendly downscale factor
try:
    from .config import ATTN_DOWNSCALE
except Exception:   # keep running even if missing in config
    ATTN_DOWNSCALE = 3

class AttentionComputer:
    """
    Stateless attention tiles for the bottom row.
    Output keys are stable: 'edges', 'flow', 'sal', 'motion'
    """

    def __init__(self, max_side: int | None = None, downscale: int | None = None):
        self.downscale = max(1, int(downscale or ATTN_DOWNSCALE))

    def _u8(self, arr: np.ndarray) -> np.ndarray:
        arr = np.abs(arr).astype(np.float32)
        if arr.size == 0:
            return np.zeros_like(arr, dtype=np.uint8)
        mn, rng = float(arr.min()), float(np.ptp(arr))  # NumPy 2.0 safe
        if rng < 1e-12:
            return np.zeros_like(arr, dtype=np.uint8)
        return ((arr - mn) * (255.0 / rng)).clip(0, 255).astype(np.uint8)

    def compute(self, frame_bgr: np.ndarray) -> dict:
        if frame_bgr is None or frame_bgr.size == 0:
            h, w = 120, 160
            blank = np.zeros((h, w, 3), np.uint8)
            return {"edges": blank, "flow": blank, "sal": blank, "motion": blank}

        H, W = frame_bgr.shape[:2]
        if self.downscale > 1:
            small = cv2.resize(frame_bgr, (W // self.downscale, H // self.downscale), interpolation=cv2.INTER_AREA)
        else:
            small = frame_bgr

        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

        # Edges
        edges = cv2.Canny(gray, 60, 160)
        edges_c = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        # Pseudo-flow (Sobel magnitude)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        mag_u8 = self._u8(mag)
        flow_c = cv2.applyColorMap(mag_u8, cv2.COLORMAP_INFERNO)

        # Spectral residual saliency
        f = np.fft.fft2(gray.astype(np.float32))
        log_amp = np.log(np.abs(f) + 1e-6)
        blur = cv2.GaussianBlur(log_amp, (7, 7), 0)
        sr = np.exp(log_amp - blur)
        sal = np.fft.ifft2(sr * np.exp(1j * np.angle(f)))
        sal_u8 = self._u8(sal)
        sal_c = cv2.applyColorMap(sal_u8, cv2.COLORMAP_MAGMA)

        # Motion (visual blend)
        motion = cv2.addWeighted(edges_c, 0.45, flow_c, 0.55, 0)

        # Upscale tiles back to original width for consistent layout
        if self.downscale > 1:
            target_w = W
            scale = target_w / float(edges_c.shape[1])
            target_h = int(edges_c.shape[0] * scale)
            edges_c = cv2.resize(edges_c, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            flow_c  = cv2.resize(flow_c,  (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            sal_c   = cv2.resize(sal_c,   (target_w, target_h), interpolation=cv2.INTER_NEAREST)
            motion  = cv2.resize(motion,  (target_w, target_h), interpolation=cv2.INTER_NEAREST)

        return {"edges": edges_c, "flow": flow_c, "sal": sal_c, "motion": motion}
