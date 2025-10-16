# app/emotions_basic.py
from typing import Dict, Any, Tuple
import numpy as np

EMO_LABELS = ["anger","disgust","fear","happiness","sadness","surprise","neutral","contempt","confusion"]

def softmax(z):
    z = np.asarray(z, dtype=float)
    z = z - z.max()
    e = np.exp(z)
    return (e / (e.sum() + 1e-9)).tolist()

def infer_probs(feats, meta: Dict[str,Any]) -> Dict[str,float]:
    p = dict.fromkeys(EMO_LABELS, 0.0)

    mouth_open = getattr(feats, "mouth_open", 0.0)
    eyes_open  = getattr(feats, "eyes_open", 0.0)
    yaw, pitch, roll = getattr(feats, "yaw",0.0), getattr(feats,"pitch",0.0), getattr(feats,"roll",0.0)

    # simple cues
    smile = max(0.0, 0.8*eyes_open + 0.6*mouth_open - 0.5)
    surprise = max(0.0, 1.2*mouth_open + 0.7*eyes_open - 0.8)
    sadness = max(0.0, 0.4*(1.0-eyes_open) + 0.2*abs(roll)/30.0)
    anger = max(0.0, 0.3*(1.0-eyes_open) + 0.2*abs(pitch)/20.0)
    disgust = max(0.0, 0.25*(1.0-eyes_open) + 0.25*abs(roll)/25.0)
    fear = max(0.0, 0.5*eyes_open + 0.4*(1.0-mouth_open) - 0.3)
    contempt = max(0.0, 0.35*abs(roll)/25.0)
    confusion = max(0.0, 0.45*abs(yaw)/25.0 + 0.25*abs(roll)/25.0)
    neutral = max(0.0, 0.7 - (smile+surprise+sadness+anger+disgust+fear+contempt+confusion))

    scores = [anger, disgust, fear, smile, sadness, surprise, neutral, contempt, confusion]
    probs = softmax(scores)
    return dict(zip(EMO_LABELS, probs))
