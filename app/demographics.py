# app/demographics.py
from typing import Dict, Any
import numpy as np

def estimate(frame_bgr, meta: Dict[str,Any]) -> Dict[str,Any]:
    pts = meta.get("points",{})
    lipL, lipR = pts.get("lipL"), pts.get("lipR")
    irisL, irisR = pts.get("irisL"), pts.get("irisR")

    H, W = frame_bgr.shape[:2]
    # Lighting proxy
    gray = frame_bgr.mean(axis=2)
    lighting = "low" if gray.mean()<70 else ("high" if gray.mean()>180 else "normal")

    # Glasses heuristic: bright frame-like pixels near eyes
    glasses = False
    if irisL and irisR:
        def area(cx,cy,rad=28):
            x0,y0 = max(0,cx-rad), max(0,cy-rad)
            x1,y1 = min(W,cx+rad), min(H,cy+rad)
            return frame_bgr[y0:y1,x0:x1]
        a = area(*irisL); b = area(*irisR)
        def spec(roi):
            if roi.size==0: return 0.0
            g = roi.astype(np.float32).mean(axis=2)
            return float((g>210).mean())
        glasses = (spec(a)+spec(b))/2.0 > 0.06  # small shiny ratio

    # Headwear heuristic: darker/colored band above forehead vs face
    headwear = False
    box_head = next((b for n,b in meta.get("boxes",[]) if n=="head"), None)
    if box_head:
        x,y,w,h = box_head
        band = frame_bgr[max(0,y- int(0.22*h)):y, x:x+w]
        face = frame_bgr[y:y+int(0.4*h), x:x+w]
        if band.size>0 and face.size>0:
            diff = float(abs(band.mean()-face.mean()))
            headwear = diff>22.0

    return {
        "lighting": lighting,
        "distance": "mid",  # (keep simple; could compute face ratio)
        "glasses": bool(glasses),
        "facial_hair": False,  # optional
        "headwear": bool(headwear),
    }

def render_markdown(res: Dict[str,Any]) -> str:
    gl = "yes" if res.get("glasses") else "no"
    hw = "yes" if res.get("headwear") else "no"
    return (
        "**Demographics (appearance)**\n\n"
        f"- Lighting: **{res.get('lighting','?')}**\n"
        f"- Distance: **{res.get('distance','?')}**\n"
        f"- Glasses: **{gl}**\n"
        f"- Headwear: **{hw}**\n"
    )
