# humanoid_neurograph/app/msk.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import numpy as np
import cv2

try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
    HAVE_POSE = True
except Exception:
    mp_pose = None
    HAVE_POSE = False


@dataclass
class ROM:
    shoulder_abduction: Optional[float] = None  # deg
    elbow_flexion: Optional[float] = None       # deg
    wrist_flexion: Optional[float] = None       # deg
    head_tilt: Optional[float] = None           # deg (roll)

def _angle(a, b, c):
    """Return angle ABC (deg), from 3 points (x,y)."""
    a = np.array(a, np.float32); b = np.array(b, np.float32); c = np.array(c, np.float32)
    v1 = a - b; v2 = c - b
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 < 1e-3 or n2 < 1e-3: return None
    cos = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))

def _norm_xy(lm, W, H): return (int(lm.x * W), int(lm.y * H))

class MSKAnalyzer:
    """
    Lightweight pose extraction + ROM estimates. Uses MediaPipe Pose if available,
    otherwise only draws nothing but preserves interface (safe no-op).
    """
    def __init__(self):
        self.pose = mp_pose.Pose(min_detection_confidence=0.5,
                                 min_tracking_confidence=0.5) if HAVE_POSE else None
        self.rom = ROM()

    def process(self, frame_bgr):
        """Return (angles-overlay drawn on black), overlay image."""
        H, W = frame_bgr.shape[:2]
        overlay = np.zeros_like(frame_bgr)
        if not self.pose:
            # no-op, but keep interface stable
            self.rom = ROM()
            return self.rom, overlay

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.pose.process(rgb)
        if not res.pose_landmarks:
            self.rom = ROM()
            return self.rom, overlay

        lm = res.pose_landmarks.landmark

        # helpers (with visibility check)
        def get(idx):
            if idx >= len(lm): return None
            if lm[idx].visibility < 0.4: return None
            return _norm_xy(lm[idx], W, H)

        # keypoints
        l_sh = get(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
        r_sh = get(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
        l_el = get(mp_pose.PoseLandmark.LEFT_ELBOW.value)
        r_el = get(mp_pose.PoseLandmark.RIGHT_ELBOW.value)
        l_wr = get(mp_pose.PoseLandmark.LEFT_WRIST.value)
        r_wr = get(mp_pose.PoseLandmark.RIGHT_WRIST.value)
        l_hip = get(mp_pose.PoseLandmark.LEFT_HIP.value)
        r_hip = get(mp_pose.PoseLandmark.RIGHT_HIP.value)
        nose  = get(mp_pose.PoseLandmark.NOSE.value)
        l_eye = get(mp_pose.PoseLandmark.LEFT_EYE.value)
        r_eye = get(mp_pose.PoseLandmark.RIGHT_EYE.value)

        # draw light skeleton
        def line(a,b,c): 
            if a and b: cv2.line(overlay,a,b,c,2,cv2.LINE_AA)
        def dot(a,c): 
            if a: cv2.circle(overlay,a,3,c,-1,cv2.LINE_AA)

        for pair in [(l_sh,l_el), (l_el,l_wr), (r_sh,r_el), (r_el,r_wr),
                     (l_sh,r_sh), (l_sh,l_hip), (r_sh,r_hip)]:
            line(pair[0], pair[1], (140,210,255))
        for p in [l_sh,r_sh,l_el,r_el,l_wr,r_wr,l_hip,r_hip,nose,l_eye,r_eye]:
            dot(p,(240,220,120))

        # ROM estimates
        shoulder_abd = None
        elbow_flex = None
        wrist_flex = None
        head_tilt = None

        # Shoulder abduction: angle between torso axis and upper arm (use left if available)
        if l_sh and l_el and l_hip:
            torso_vec = np.array(l_hip) - np.array(l_sh)
            arm_vec = np.array(l_el) - np.array(l_sh)
            def ang(v1,v2):
                n1=np.linalg.norm(v1); n2=np.linalg.norm(v2)
                if n1<1e-3 or n2<1e-3: return None
                c=np.clip(np.dot(v1,v2)/(n1*n2),-1.0,1.0)
                return float(np.degrees(np.arccos(c)))
            shoulder_abd = ang(torso_vec, arm_vec)

        # Elbow flexion
        if l_sh and l_el and l_wr:
            elbow_flex = _angle(l_sh, l_el, l_wr)  # ~180 straight, lower flexed

        # Wrist flexion: angle between forearm and hand direction
        if l_el and l_wr and r_wr:
            # use forearm vector and a short hand segment (wrist to wrist of other hand as proxy baseline)
            wrist_dir = np.array(r_wr) - np.array(l_wr)
            forearm = np.array(l_wr) - np.array(l_el)
            def ang(v1,v2):
                n1=np.linalg.norm(v1); n2=np.linalg.norm(v2)
                if n1<1e-3 or n2<1e-3: return None
                c=np.clip(np.dot(v1,v2)/(n1*n2),-1.0,1.0)
                return float(np.degrees(np.arccos(c)))
            wrist_flex = ang(forearm, wrist_dir)

        # Head tilt (roll) from eyes
        if l_eye and r_eye:
            dy = float(l_eye[1]-r_eye[1]); dx = float(l_eye[0]-r_eye[0])
            head_tilt = float(np.degrees(np.arctan2(dy, dx)))

        self.rom = ROM(shoulder_abduction=shoulder_abd,
                       elbow_flexion=elbow_flex,
                       wrist_flexion=wrist_flex,
                       head_tilt=head_tilt)
        return self.rom, overlay

    @staticmethod
    def draw(img, overlay, rom: ROM, last_rom: ROM):
        out = cv2.addWeighted(img, 1.0, overlay, 0.45, 0.0)
        # On-frame ROM text (compact)
        y = 24
        def put(label, val, unit="°"):
            nonlocal y
            if val is None: return
            cv2.putText(out, f"{label}: {val:.0f}{unit}", (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 240, 255), 2, cv2.LINE_AA)
            y += 20

        put("Shoulder abd", rom.shoulder_abduction or 0.0)
        put("Elbow flex", rom.elbow_flexion or 0.0)
        put("Wrist flex", rom.wrist_flexion or 0.0)
        put("Head tilt", rom.head_tilt or 0.0)
        return out
