# app/tremor.py
from dataclasses import dataclass
from collections import deque
from typing import Deque, Dict, Tuple, Optional, List
import numpy as np

@dataclass
class TremorState:
    have_data: bool = False
    is_tremor: bool = False
    peak_hz: float = 0.0
    band_ratio: float = 0.0      # energy in 3–12 Hz / total 0.5–15 Hz
    rms_amp_px: float = 0.0      # RMS amplitude (px)
    window_sec: float = 0.0

class TremorTracker:
    """
    Tracks a single hand centroid and detects tremor using FFT.
    """
    def __init__(self, keep_sec: float = 8.0, target_fs: float = 30.0):
        self.keep_sec = keep_sec
        self.target_fs = target_fs
        self.buf: Deque[Tuple[float, float, float]] = deque(maxlen=int(keep_sec * (target_fs + 10)))
        self.last_state = TremorState()

    def update(self, t: float, x: float, y: float) -> TremorState:
        self.buf.append((t, x, y))
        return self._analyze()

    def _analyze(self) -> TremorState:
        fs = self.target_fs
        if len(self.buf) < int(1.5 * fs):
            self.last_state = TremorState(have_data=False)
            return self.last_state

        ts = np.array([b[0] for b in self.buf], dtype=np.float64)
        xs = np.array([b[1] for b in self.buf], dtype=np.float64)
        ys = np.array([b[2] for b in self.buf], dtype=np.float64)
        T = ts[-1] - ts[0]
        if T < 1.0:
            self.last_state = TremorState(have_data=False)
            return self.last_state

        # Resample uniformly
        t_u = np.arange(ts[0], ts[-1], 1.0/fs, dtype=np.float64)
        if t_u.size < int(1.0*fs):
            self.last_state = TremorState(have_data=False)
            return self.last_state
        x_u = np.interp(t_u, ts, xs)
        y_u = np.interp(t_u, ts, ys)

        # Detrend & center
        def detrend(z):
            n = z.size
            t = np.arange(n, dtype=np.float64)
            A = np.vstack([t, np.ones(n)]).T
            m, c = np.linalg.lstsq(A, z, rcond=None)[0]
            return z - (m*t + c)
        sig = detrend(x_u) + detrend(y_u)

        rms = float(np.sqrt(np.mean(sig**2)))

        # FFT power
        window = np.hanning(sig.size)
        Z = np.fft.rfft(sig * window)
        freqs = np.fft.rfftfreq(sig.size, d=1.0/fs)
        P = (Z.real**2 + Z.imag**2)

        valid = (freqs >= 0.5) & (freqs <= 15.0)
        band  = (freqs >= 3.0) & (freqs <= 12.0)
        if not np.any(valid):
            self.last_state = TremorState(have_data=False)
            return self.last_state

        P_tot = float(np.sum(P[valid])) + 1e-12
        P_band = float(np.sum(P[band]))
        ratio = P_band / P_tot
        peak_hz = float(freqs[band][np.argmax(P[band])]) if np.any(band) else 0.0

        is_tremor = (ratio > 0.45 and rms > 1.8 and T >= 2.0)

        self.last_state = TremorState(
            have_data=True,
            is_tremor=bool(is_tremor),
            peak_hz=peak_hz,
            band_ratio=ratio,
            rms_amp_px=rms,
            window_sec=T,
        )
        return self.last_state


class TremorAnalyzer:
    """ Manages trackers for both hands ('Left', 'Right'). """
    def __init__(self, keep_sec: float = 8.0, target_fs: float = 30.0):
        self.trackers: Dict[str, TremorTracker] = {
            "Left":  TremorTracker(keep_sec, target_fs),
            "Right": TremorTracker(keep_sec, target_fs),
        }

    def update(self, t: float, hands: List[Dict]) -> Dict[str, TremorState]:
        states: Dict[str, TremorState] = {}
        for h in hands or []:
            name = h.get("name")
            if name in self.trackers:
                states[name] = self.trackers[name].update(t, float(h.get("cx", 0.0)), float(h.get("cy", 0.0)))
        return states
