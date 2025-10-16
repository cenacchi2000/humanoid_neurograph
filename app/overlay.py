# app/overlay.py
from typing import Dict, Any, Tuple, List
import cv2
import numpy as np

GREEN = (90,255,140)
CYAN  = (90,200,255)
ORANGE= (255,180,90)
PINK  = (255,120,200)

def draw_lips_and_cheeks(img, meta: Dict[str,Any], label_text: str|None=None):
    pts = meta.get("points",{})
    lipL, lipR = pts.get("lipL"), pts.get("lipR")
    lipUp, lipDn = pts.get("lipUp"), pts.get("lipDn")
    chL, chR = pts.get("cheekL"), pts.get("cheekR")
    if all(v is not None for v in (lipL, lipR, lipUp, lipDn)):
        # ellipse around mouth
        cx = int(0.5*(lipL[0]+lipR[0])); cy=int(0.5*(lipUp[1]+lipDn[1]))
        axes=(max(6, abs(lipR[0]-lipL[0])//2), max(4, abs(lipDn[1]-lipUp[1])//2+6))
        cv2.ellipse(img,(cx,cy),axes,0,0,360,ORANGE,2)
        txt = label_text or "lips"
        cv2.putText(img, txt, (cx-axes[0], cy+axes[1]+14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, ORANGE, 1, cv2.LINE_AA)
    for c in (chL, chR):
        if c is not None:
            cv2.circle(img, c, 6, PINK, 2)
    return img

def draw_hand_skeleton(img, meta: Dict[str,Any], show_ids=True):
    # lines
    for (pa,pb,name) in meta.get("hand_lines", []):
        cv2.line(img, pa, pb, GREEN, 2)
    # fingertip IDs
    for h in meta.get("hands", []):
        pts = h.get("pts")
        if pts is None: continue
        tips = [4,8,12,16,20]
        for j,k in enumerate(tips, start=1):
            p = tuple(pts[k].astype(int))
            cv2.circle(img, p, 5, (255,255,255), -1)
            if show_ids:
                cv2.putText(img, str(j), (p[0]+5,p[1]-6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (40,40,40), 2, cv2.LINE_AA)
    return img
