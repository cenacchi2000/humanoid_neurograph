# app/runner.py
import time
import collections
from typing import Dict, Optional

import numpy as np
import cv2
import streamlit as st

from .config import (
    DEFAULT_SEED,
    DEFAULT_TARGET_FPS,
    APP_FOOT,
    FRAME_SKIP_MIN_MS,
    ATTN_TARGET_HZ,
    OPENCV_NUM_THREADS,
    POSE_EVERY_N_FRAMES,
)
from .utils import put_label, clamp
from .detectors import make_detector
from .graph import NCPGraph
from .games import GamesPro
from .ui import NeuroGraphUI
from .attention import AttentionComputer
from .pupil import PupilAnalyzer, PupilState
from .msk import MSKAnalyzer
from .emotions import analyze_cognitive
from .tremor import TremorAnalyzer
from .lips import LipsAnalyzer, LipsState

cv2.setNumThreads(OPENCV_NUM_THREADS)

# ---------- helpers ----------

def _draw_face_box(draw, feats, meta):
    for name, (x, y, w, h) in meta.get("boxes", []):
        if name == "head":
            cv2.rectangle(draw, (x, y), (x + w, y + h), (90, 200, 255), 2)
            put_label(draw, (x, y - 8), f"Head | yaw {feats.yaw:+.0f}°")
            return (x, y, w, h)
    return None

def _draw_iris(draw, meta, pupil_state: PupilState):
    cL = meta.get("points", {}).get("irisL")
    cR = meta.get("points", {}).get("irisR")
    rL = meta.get("points", {}).get("irisL_r")
    rR = meta.get("points", {}).get("irisR_r")
    from .pupil import PupilAnalyzer as _PA
    _PA.draw(draw, centerL=cL, centerR=cR, rL=rL, rR=rR, state=pupil_state)

def _draw_hand_skeletons(draw, hands_meta):
    if not hands_meta:
        return
    conn = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
    ]
    for h in hands_meta:
        pts = h.get("pts21")
        name = h.get("name", "?")
        bx, by, bw, bh = h.get("box", (None, None, None, None))
        if None not in (bx, by, bw, bh):
            cv2.rectangle(draw, (bx, by), (bx + bw, by + bh), (90, 255, 140), 2)
        if isinstance(pts, np.ndarray) and pts.shape == (21, 2):
            ptsi = pts.astype(int)
            for a, b in conn:
                cv2.line(draw, tuple(ptsi[a]), tuple(ptsi[b]), (60, 220, 255), 2)
            for p in ptsi:
                cv2.circle(draw, tuple(p), 2, (255, 255, 255), -1)
        put_label(draw, (int(h.get('cx',0))+10, int(h.get('cy',0))-22), name, color=(200, 255, 180))

def _compute_distance(face_box, frame_shape):
    if not face_box: return "unknown"
    _, _, _, h = face_box
    H = frame_shape[0]
    r = h / max(1, H)
    if r >= 0.42: return "very close"
    if r >= 0.30: return "close"
    if r >= 0.18: return "mid"
    if r >= 0.10: return "far"
    return "very far"

def _binary_demographics(frame_bgr, meta, face_box) -> Dict[str, str]:
    pts = meta.get("points", {})
    eyeL = pts.get("pupilL")
    eyeR = pts.get("pupilR")
    H, W = frame_bgr.shape[:2]

    def roi_stats(center, rw=36, rh=22):
        if not center: return 0.0, 0.0
        cx, cy = center
        x0, y0 = int(max(0, cx - rw)), int(max(0, cy - rh))
        x1, y1 = int(min(W - 1, cx + rw)), int(min(H - 1, cy + rh))
        roi = frame_bgr[y0:y1, x0:x1]
        if roi.size == 0: return 0.0, 0.0
        edges = cv2.Canny(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), 60, 120)
        edge_density = float(np.mean(edges > 0))
        spec = float(np.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[:, :, 2] > 235))
        return edge_density, spec

    eL, sL = roi_stats(eyeL)
    eR, sR = roi_stats(eyeR)
    glasses_yes = (max(eL, eR) > 0.18 and max(sL, sR) > 0.06)

    if face_box:
        x, y, w, h = face_box
        top = frame_bgr[max(0, y - int(0.18 * h)): y + int(0.05 * h), x:x + w]
        if top.size > 0:
            hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
            sat_low = float(np.mean(hsv[:, :, 1] < 35))
            edg = float(np.mean(cv2.Canny(cv2.cvtColor(top, cv2.COLOR_BGR2GRAY), 80, 160) > 0))
            headwear_yes = (sat_low > 0.55 and edg > 0.12)
        else:
            headwear_yes = False
    else:
        headwear_yes = False

    return {
        "glasses": "yes" if glasses_yes else "no",
        "headwear": "yes" if headwear_yes else "no",
    }

def _nine_way_expression(feats, lips_state: Optional[LipsState]):
    eopen = clamp(getattr(feats, "eyes_open", 0.0), 0, 1)
    stress = clamp(getattr(feats, "stress", 0.0), 0, 1)
    mouth = clamp(getattr(feats, "mouth_open", 0.0), 0, 1)
    blink = clamp(getattr(feats, "blink", 0.0), 0, 1)
    yaw = abs(clamp(getattr(feats, "yaw", 0.0) / 50.0, -1, 1))
    lips_open = clamp(getattr(lips_state, "gap_norm", 0.0), 0, 1) if lips_state and lips_state.ok else 0.0
    vec = {
        "neutral": (1 - stress) * (1 - mouth) * (0.6 + 0.4 * eopen),
        "happiness": (1 - stress) * (mouth * 0.8 + lips_open * 0.7),
        "sadness": stress * (1 - mouth) * (0.5 + 0.5 * (1 - eopen)),
        "anger": stress * (0.3 + 0.7 * (1 - eopen)),
        "surprise": eopen * 0.7 + mouth * 0.5,
        "fear": stress * eopen * 0.6 + blink * 0.2,
        "disgust": stress * 0.5 + (1 - lips_open) * 0.2,
        "contempt": (1 - eopen) * 0.3 + (1 - mouth) * 0.3 + yaw * 0.2,
        "confusion": yaw * 0.6 + (0.5 - abs(0.5 - mouth)) * 0.3,
    }
    s = sum(vec.values()) + 1e-9
    return {k: float(v / s) for k, v in vec.items()}

def _draw_all_overlays(draw, feats, meta, pupil_state: PupilState, lips_state: Optional[LipsState]):
    face_box = _draw_face_box(draw, feats, meta)
    _draw_iris(draw, meta, pupil_state)
    if lips_state is not None:
        draw[:] = LipsAnalyzer.draw(draw, lips_state, meta)
    _draw_hand_skeletons(draw, meta.get("hands", []))
    return face_box

# ---------- main ----------

def main():
    ui = NeuroGraphUI(default_fps=DEFAULT_TARGET_FPS)
    detector = make_detector()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, ui.target_fps or DEFAULT_TARGET_FPS)

    ncp = NCPGraph(seed=DEFAULT_SEED)
    games = GamesPro(
        cam_size=(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640),
                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)),
        fps_hint=ui.target_fps,
    )
    attn = AttentionComputer(downscale=3)
    pupil = PupilAnalyzer(fps_hint=ui.target_fps)
    msk = MSKAnalyzer()
    trem = TremorAnalyzer(target_fs=ui.target_fps)
    lips = LipsAnalyzer(target_fs=ui.target_fps)

    t_prev = time.time()
    fps_s = collections.deque(maxlen=30)
    blink_times = collections.deque(maxlen=120)

    last_plot_ms = 0
    last_attn_ms = 0
    attn_skip_ms = int(1000 / max(1, ATTN_TARGET_HZ))
    last_edges = last_flow = last_sal = last_mot = None

    frame_idx = 0

    # Cognitive-test controls
    with ui.tests[0]:
        st.markdown("**Finger Rhythm & Tremor**")
        if st.button("Start", key="pro_rhythm_start", use_container_width=True): games.start_rhythm()
        if st.button("Stop",  key="pro_rhythm_stop",  use_container_width=True): games.stop_rhythm()
    with ui.tests[1]:
        st.markdown("**Anti-saccade**")
        if st.button("Start", key="pro_anti_start", use_container_width=True): games.start_anti()
        if st.button("Stop",  key="pro_anti_stop",  use_container_width=True): games.stop_anti()
    with ui.tests[2]:
        st.markdown("**Air Trail Making**")
        if st.button("Start", key="pro_trail_start", use_container_width=True): games.start_trail()
        if st.button("Stop",  key="pro_trail_stop",  use_container_width=True): games.stop_trail()

    while True:
        if not st.session_state.get("run_v7", True):
            time.sleep(0.05)
            continue

        ok, frame = cap.read()
        if not ok:
            st.warning("Cannot read from camera index 0")
            break

        if st.session_state.get("mirror_v7", True):
            frame = cv2.flip(frame, 1)

        feats, meta = detector.detect(frame)

        # pupil dynamics
        t_now = time.time()
        ps = pupil.update(getattr(feats, "pupil_diam", None), t_now)
        if getattr(feats, "blink", 0) > 0:
            blink_times.append(t_now)

        # tremor
        trem_states = trem.update(t_now, meta.get("hands", []))

        # lips / cheeks
        lips_state = lips.update(frame, meta, t_now)

        # MSK (throttled)
        if frame_idx % max(1, POSE_EVERY_N_FRAMES) == 0:
            angles, overlay = msk.process(frame)
        else:
            angles, overlay = msk.rom, np.zeros_like(frame)

        # tests
        games.update_rhythm(frame, meta)
        games.update_anti(frame, meta, feats)
        try:
            games.update_trail(frame, meta)
        except Exception:
            pass

        # overlays
        draw = frame.copy()
        draw = MSKAnalyzer.draw(draw, overlay, angles, msk.rom)
        face_box = _draw_all_overlays(draw, feats, meta, ps, lips_state)

        # per-hand tremor label
        for h in meta.get("hands", []):
            name = h.get("name", "?")
            stt = trem_states.get(name)
            if stt and stt.have_data:
                color = (0, 220, 120) if not stt.is_tremor else (0, 120, 255)
                label = f"{name}: {'TREMOR' if stt.is_tremor else 'steady'} {stt.peak_hz:.1f}Hz  rms={stt.rms_amp_px:.1f}px"
                put_label(draw, (int(h.get('cx',0))+10, int(h.get('cy',0))-6), label, color=color)

        # attention maps (throttled)
        now_ms = int(time.time() * 1000)
        if last_edges is None or (now_ms - last_attn_ms) >= attn_skip_ms:
            maps = attn.compute(frame)
            last_edges, last_flow, last_sal, last_mot = maps["edges"], maps["flow"], maps["sal"], maps["motion"]
            last_attn_ms = now_ms

        # main view (no deprecated args)
        ui.video_ph.image(cv2.cvtColor(draw, cv2.COLOR_BGR2RGB), clamp=True, use_container_width=True)

        # ladder + cards
        H, W = frame.shape[:2]
        strip_h = ui.desired_attn_height(W)
        edges_color = cv2.resize(last_edges, (W, strip_h), interpolation=cv2.INTER_AREA)
        ui.ladder_ph.image(edges_color, caption="Edges • Attention ladder", use_container_width=True)

        card_w, card_h = ui.desired_card_size(W)
        ui.attn1_ph.image(cv2.resize(last_flow, (card_w, card_h)), caption="Motion (Optical flow magnitude)", use_container_width=True)
        ui.attn2_ph.image(cv2.resize(last_sal,  (card_w, card_h)), caption="Saliency (Spectral residual)", use_container_width=True)
        ui.attn3_ph.image(cv2.resize(last_mot,  (card_w, card_h)), caption="Motion accumulation", use_container_width=True)

        # NCP graph
        if now_ms - last_plot_ms >= FRAME_SKIP_MIN_MS:
            try:
                activity = [
                    clamp(feats.eyes_open, 0, 1),
                    clamp(feats.stress, 0, 1),
                    clamp(1.0 if lips_state and lips_state.is_speaking else 0.0, 0, 1),
                    clamp(min(1, abs(feats.yaw)/60.0), 0, 1),
                    clamp(0.5*(feats.left_hand + feats.right_hand), 0, 1),
                    clamp(0.3 + 0.7*feats.eyes_open*(1-feats.stress), 0, 1),
                ]
                activity += [0.2]*10
                activity += [0.1, 0.05, 0.15]
                fig = ncp.figure(activity)
                ui.graph_ph.plotly_chart(fig, use_container_width=True, theme=None, config={"displayModeBar": False})
            except Exception:
                pass
            last_plot_ms = now_ms

        # Expressions + demographics (always visible)
        expr = _nine_way_expression(feats, lips_state)
        primary = max(expr.items(), key=lambda kv: kv[1])[0]
        face_dist = _compute_distance(face_box, frame.shape)
        demo = _binary_demographics(frame, meta, face_box)

        ui.cog_text_ph.markdown(
            f"**Emotion & Cognitive Screen**  \n"
            f"Primary expression: **{primary}**"
        )

        import plotly.graph_objects as go
        bar = go.Figure(go.Bar(x=list(expr.values()), y=list(expr.keys()), orientation="h"))
        bar.update_layout(height=220, margin=dict(l=2, r=8, t=6, b=6))
        bar.update_xaxes(range=[0, 1.0])
        ui.expr_ph.plotly_chart(bar, use_container_width=True, config={"displayModeBar": False})

        ui.demo_ph.markdown(
            "**Demographics (appearance)**  \n"
            f"- Distance: **{face_dist}**  \n"
            f"- Glasses: **{demo['glasses']}**  \n"
            f"- Headwear: **{demo['headwear']}**"
        )

        # status (single compact line)
        dt = max(time.time() - t_prev, 1e-3); t_prev = time.time()
        fps_s.append(1.0/dt)
        lips_txt = f" | lips {lips_state.label}" if lips_state and lips_state.ok else ""
        ui.render_status(
            fps_s, blink_times, feats.yaw, feats.emotion,
            pupil_text=f"Pupil {ps.event} z={ps.zscore:+.1f} roc={ps.roc:+.02f}/s cog={ps.cognitive_index:.2f}{lips_txt}"
        )

        frame_idx += 1
        time.sleep(max(0.0, (1.0 / ui.target_fps) - (time.time() - t_now)))

    cap.release()
    st.markdown(f"<div style='text-align:center;color:#98a2b3'>{APP_FOOT}</div>", unsafe_allow_html=True)


__all__ = ["main"]
