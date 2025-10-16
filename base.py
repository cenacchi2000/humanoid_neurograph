# humanoid_neurograph/app/detectors/base.py
from abc import ABC, abstractmethod
import numpy as np
from typing import Tuple, Dict, Any
from ..features import LiveFeatures

class BaseDetector(ABC):
    """Interface for all detectors: must return (features, meta)."""
    @abstractmethod
    def detect(self, frame_bgr: np.ndarray) -> Tuple[LiveFeatures, Dict[str, Any]]:
        ...
