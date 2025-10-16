# humanoid_neurograph/app/detectors/__init__.py
from typing import Optional
from ..utils import HAVE_MP
from .cv_detector import CVDetector

def make_detector():
    """
    Factory for detectors.
    Returns MediaPipe detector if available, otherwise OpenCV Haar fallback.
    """
    if HAVE_MP:
        try:
            from .mp_detector import MPDetector
            return MPDetector()
        except Exception:
            # If MP import fails at runtime, fall back gracefully
            return CVDetector()
    return CVDetector()
