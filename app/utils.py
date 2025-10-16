# humanoid_neurograph/app/utils.py
import cv2
import numpy as np

def clamp(v, a, b): 
    return max(a, min(b, v))

def put_label(img, xy, text, color=(255, 255, 255), bg=(20, 20, 20)):
    """Left-attached label box near ROI."""
    x, y = int(xy[0]), int(xy[1])
    f = 0.56
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, f, 1)
    pad = 4
    cv2.rectangle(img, (x - tw - 2*pad - 8, y - th - 2*pad),
                  (x, y + pad), bg, -1)
    cv2.putText(img, text, (x - tw - pad, y - pad),
                cv2.FONT_HERSHEY_SIMPLEX, f, color, 1, cv2.LINE_AA)

# Simple Haar cache
_HAAR = {}
def get_haar(name, path):
    if name in _HAAR:
        return _HAAR[name]
    c = cv2.CascadeClassifier(cv2.data.haarcascades + path)
    _HAAR[name] = c
    return c

# Optional MediaPipe availability flag
HAVE_MP = True
try:
    import mediapipe as mp  # noqa: F401
except Exception:
    HAVE_MP = False
