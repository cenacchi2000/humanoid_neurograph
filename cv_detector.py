# humanoid_neurograph/app/detectors/cv_detector.py
from typing import Tuple, Dict, Any
import numpy as np, cv2
from ..features import LiveFeatures
from ..pose import solve_headpose
from ..utils import get_haar
from .base import BaseDetector

class CVDetector(BaseDetector):
    def __init__(self):
        self.face = get_haar("face","haarcascade_frontalface_default.xml")
        self.eye  = get_haar("eye","haarcascade_eye.xml")
        self.mouth= get_haar("mouth","haarcascade_smile.xml")

    def detect(self, frame_bgr: np.ndarray) -> Tuple[LiveFeatures, Dict[str, Any]]:
        feats = LiveFeatures()
        meta = {"boxes": [], "points": {}}
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        H,W = gray.shape

        faces = self.face.detectMultiScale(gray,1.15,5)
        if len(faces):
            x,y,w,h = faces[0]
            meta["boxes"].append(("head",(x,y,w,h)))
            roi = gray[y:y+h, x:x+w]
            eyes = self.eye.detectMultiScale(roi,1.2,10)
            mouth= self.mouth.detectMultiScale(roi,1.4,15)

            if len(eyes)>=2:
                e1 = np.array([x+eyes[0][0]+eyes[0][2]*0.5, y+eyes[0][1]+eyes[0][3]*0.5])
                e2 = np.array([x+eyes[1][0]+eyes[1][2]*0.5, y+eyes[1][1]+eyes[1][3]*0.5])
                nose= np.array([x+w*0.5, y+h*0.38])
                lipL= np.array([x+w*0.40, y+h*0.68])
                lipR= np.array([x+w*0.60, y+h*0.68])
                pts = np.vstack([nose, e1, e2, lipL, lipR]).astype(np.float64)
                feats.yaw,feats.pitch,feats.roll = solve_headpose(pts, W, H)
                meta["points"].update({
                    "pupilL":tuple(e1.astype(int)), "pupilR":tuple(e2.astype(int)),
                    "nose":tuple(nose.astype(int))
                })
                feats.eyes_open = 0.5
            feats.mouth_open = 1.0 if len(mouth)>0 else 0.0
            feats.speaking   = feats.mouth_open

        return feats, meta
