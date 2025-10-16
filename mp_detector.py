# app/detectors/mp_detector.py
import math
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, Any

import numpy as np
import cv2

try:
    import mediapipe as mp
    mp_face = mp.solutions.face_mesh
    mp_face_det = mp.solutions.face_detection
    mp_hands = mp.solutions.hands
    HAVE_MP = True
except Exception:
    HAVE_MP = False
    mp_face = mp_face_det = mp_hands = None

from ..utils import clamp
from ..pose import solve_headpose
from ..features import LiveFeatures

HAND_CONNECTIONS = [
    # Mediapipe hands connections (subset for skeleton)
    (0,1),(1,2),(2,3),(3,4),      # thumb
    (0,5),(5,6),(6,7),(7,8),      # index
    (0,9),(9,10),(10,11),(11,12), # middle
    (0,13),(13,14),(14,15),(15,16),# ring
    (0,17),(17,18),(18,19),(19,20) # pinky
]

def _xy(lm, W, H):
    return np.array([lm.x * W, lm.y * H], dtype=np.float64)

class MPDetector:
    """
    MediaPipe FaceMesh + Hands wrapper.
    Emits iris centers/radii, mouth & cheeks points, hand landmarks & skeleton lines.
    """
    def __init__(self):
        if not HAVE_MP:
            raise RuntimeError("MediaPipe not available")

        self.face_mesh = mp_face.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        try:
            self.hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        except Exception:
            self.hands = None

    def detect(self, frame_bgr) -> Tuple[LiveFeatures, Dict[str, Any]]:
        H, W = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        feats = LiveFeatures()
        meta: Dict[str, Any] = {"boxes": [], "points": {}, "hands": [], "hand_lines": []}

        # ---------- Face + iris + mouth/cheeks ----------
        fm = self.face_mesh.process(rgb)
        if fm.multi_face_landmarks:
            lm = fm.multi_face_landmarks[0].landmark

            xs = [p.x for p in lm]; ys = [p.y for p in lm]
            x0, y0 = int(min(xs) * W), int(min(ys) * H)
            x1, y1 = int(max(xs) * W), int(max(ys) * H)
            meta["boxes"].append(("head", (x0, y0, x1 - x0, y1 - y0)))

            nose  = _xy(lm[1], W, H)
            cheekL= _xy(lm[234], W, H); cheekR= _xy(lm[454], W, H)
            lipUp = _xy(lm[13], W, H);  lipDn = _xy(lm[14], W, H)
            lipL  = _xy(lm[61], W, H);  lipR  = _xy(lm[291], W, H)

            # Eye rings for openness/blink
            eyeL_ring = np.vstack([ _xy(lm[i], W, H) for i in [33,246,161,160,159] ])
            eyeR_ring = np.vstack([ _xy(lm[i], W, H) for i in [263,466,388,387,386] ])

            def eye_open(ey):
                v = np.linalg.norm(ey[3] - ey[1]) + np.linalg.norm(ey[4] - ey[2])
                h = np.linalg.norm(ey[0] - ey[1]) + 1e-3
                return clamp((v / h) * 1.7, 0.0, 1.0)

            eL = eye_open(eyeL_ring); eR = eye_open(eyeR_ring)
            feats.eyes_open = 0.5 * (eL + eR)
            feats.blink = float(feats.eyes_open < 0.25)

            # Iris centers/radii (468..477)
            irisA = [468, 469, 470, 471, 472]
            irisB = [473, 474, 475, 476, 477]
            haveA = all(i < len(lm) for i in irisA)
            haveB = all(i < len(lm) for i in irisB)
            if haveA and haveB:
                A = np.vstack([_xy(lm[i], W, H) for i in irisA])
                B = np.vstack([_xy(lm[i], W, H) for i in irisB])
                Ac, Bc = A.mean(0), B.mean(0)
                mL, mR = eyeL_ring.mean(0), eyeR_ring.mean(0)
                if np.linalg.norm(Ac - mL) < np.linalg.norm(Ac - mR):
                    irisL_c, irisR_c, Lset, Rset = Ac, Bc, A, B
                else:
                    irisL_c, irisR_c, Lset, Rset = Bc, Ac, B, A
                irisL_r = float(np.mean(np.linalg.norm(Lset - irisL_c, axis=1)))
                irisR_r = float(np.mean(np.linalg.norm(Rset - irisR_c, axis=1)))

                def eye_width(left: bool) -> float:
                    try:
                        if left:
                            a = _xy(lm[33], W, H); b = _xy(lm[133], W, H)
                        else:
                            a = _xy(lm[263], W, H); b = _xy(lm[362], W, H)
                        return float(np.linalg.norm(a - b))
                    except Exception:
                        return 40.0
                wL = max(eye_width(True), 1e-3)
                wR = max(eye_width(False), 1e-3)
                diam_norm_L = clamp(2.0 * irisL_r / wL, 0.0, 1.0)
                diam_norm_R = clamp(2.0 * irisR_r / wR, 0.0, 1.0)
                feats.pupil_diam = float(0.5 * (diam_norm_L + diam_norm_R))

                meta["points"]["irisL"] = tuple(irisL_c.astype(int))
                meta["points"]["irisR"] = tuple(irisR_c.astype(int))
                meta["points"]["irisL_r"] = irisL_r
                meta["points"]["irisR_r"] = irisR_r
            else:
                feats.pupil_diam = float(
                    min(1.0,
                        np.linalg.norm(eyeL_ring[0] - eyeL_ring[1]) / 40.0 +
                        np.linalg.norm(eyeR_ring[0] - eyeR_ring[1]) / 40.0)
                )

            mouth_gap = float(np.linalg.norm(lipDn - lipUp))
            mouth_width = float(np.linalg.norm(lipR - lipL) + 1e-3)
            feats.mouth_open = clamp(mouth_gap / mouth_width, 0.0, 1.0)
            feats.speaking   = float(feats.mouth_open > 0.25)

            pts2d = np.array([nose, eyeL_ring.mean(0), eyeR_ring.mean(0), lipL, lipR,
                              _xy(lm[152], W, H), _xy(lm[10], W, H)], dtype=np.float64)
            feats.yaw, feats.pitch, feats.roll = solve_headpose(pts2d, W, H)

            # Store mouth & cheeks for overlays/analytics
            meta["points"].update({
                "lipUp":  tuple(lipUp.astype(int)),
                "lipDn":  tuple(lipDn.astype(int)),
                "lipL":   tuple(lipL.astype(int)),
                "lipR":   tuple(lipR.astype(int)),
                "cheekL": tuple(cheekL.astype(int)),
                "cheekR": tuple(cheekR.astype(int)),
            })

        # ---------- Hands ----------
        if self.hands is not None:
            hs = self.hands.process(rgb)
            if hs.multi_hand_landmarks:
                for i, hand_lm in enumerate(hs.multi_hand_landmarks):
                    # 21 hand landmarks -> pixel coords
                    pts = np.array([[p.x * W, p.y * H] for p in hand_lm.landmark], dtype=np.float32)
        
                    # tight bounding box
                    x0, y0 = pts.min(0).astype(int)
                    x1, y1 = pts.max(0).astype(int)
                    bx, by, bw, bh = int(x0), int(y0), int(x1 - x0), int(y1 - y0)
        
                    # centroid
                    cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())
        
                    # handedness label (subject POV)
                    try:
                        handed = hs.multi_handedness[i].classification[0].label
                        name = "Left" if handed.lower().startswith("left") else "Right"
                    except Exception:
                        name = f"H{i}"
        
                    # append rich hand meta (this is what runner.py already expects)
                    meta["boxes"].append((f"hand{i}", (bx, by, bw, bh)))
                    meta["hands"].append(
                        dict(
                            name=name,
                            cx=cx,
                            cy=cy,
                            box=(bx, by, bw, bh),
                            pts21=pts,              # <--- 21 landmarks for skeleton drawing
                        )
                    )
        
                feats.left_hand  = 1.0 if any(h.get("name") == "Left"  for h in meta["hands"]) else 0.0
                feats.right_hand = 1.0 if any(h.get("name") == "Right" for h in meta["hands"]) else 0.0
        

        return feats, meta
