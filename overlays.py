# humanoid_neurograph/app/runner.py
import time, base64, collections
import numpy as np
import cv2
import streamlit as st

from .config import (
    DEFAULT_SEED, DEFAULT_TARGET_FPS, APP_FOOT,
    FRAME_SKIP_MIN_MS, ACTIVITY_EMA_ALPHA, ATTN_TARGET_HZ,
    OPENCV_NUM_THREADS, POSE_EVERY_N_FRAMES
)
from .utils import clamp
from .detectors import make_detector
from .graph import NCPGraph
from .games import GamesPro
from .ui import NeuroGraphUI
from .attention import AttentionComputer
from .pupil import PupilAnalyzer
from .msk import MSKAnalyzer
from .emotions import analyze_cognitive
from .tremor import TremorAnalyzer
from .demographics import estimate as demo_estimate, render_markdown as demo_render
from .status import StatusPanel
from .overlays import draw_overlays

cv2.setNumThreads(OPENCV_NUM_THREADS)

def _ema(prev, new, alpha):
    return [(1 - alpha) * p + alpha * n for p, n in zip(prev, new)]

def main():
    ui = NeuroGraphUI()
    status = StatusPanel(ui.status_ph)
    detector = make_detector()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FPS, ui.target_fps or DEFAULT_TARGET_FPS)

    ncp = NCPGraph(seed=DEFAULT_SEED)
    games = GamesPro(
        cam_size=(int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640),
                  int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)),
        fps_hint=ui.target_fps
    )

    attn  = AttentionComputer(downscale=3)
    pupil = PupilAnalyzer(fps_hint=ui.target_fps)
    msk   = MSKAnalyzer()
    tremor = TremorAnalyzer(target_fs=ui.target_fps)

    t_prev = time.time()
    fps_s = collections.deque(maxlen=30)
    blink_times = collections.deque(maxlen=120)
    activity_ema = [0.0, 0.0, 0.0, 0.0]

    last_plot_ms = 0
    last_attn_ms = 0
    attn_skip_ms = int(1000 / max(1, ATTN_TARGET_HZ))
    last_edges = last_flow = last_sal = last_mot = None

    frame_idx = 0

    # Cognitive test buttons (single set, not recreated per-loop)
    with ui.tests[0]:
        st.markdown("**Finger Rhythm & Tremor**")
        if st.button("Start", key="pro_rhythm_start", use_container_width=True):
            games.start_rhythm()
        if st.button("Stop",  key="pro_rhythm_stop",  use_container_width=True):
            games.stop_rhythm()
    with ui.tests[1]:
        st.markdown("**Anti-saccade**")
        if st.button("Start", key="pro_anti_start", use_container_width=True):
            games.start_anti()
        if st.button("Stop",  key="pro_anti_stop",  use_container_width=True):
            games.stop_anti()
    with ui.tests[2]:
        st.markdown("**Air Trail Making**")
        if st.button("Start", key="pro_trail_start", use_container_width=True):
            games.start_trail()
        if st.button("Stop",  key="pro_trail_stop",  use_container_width=True):
            games.stop_trail()

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

        # ensure hands (for tremor)
        if "hands" not in meta:
            meta["hands"] = []
            for name, box in meta.get("boxes", []):
                if str(name).startswith("hand"):
                    meta["hands"].append({"name": name, "box": box})

        # pupil dynamics
        t_now = time.time()
        ps = pupil.update(getattr(feats, "pupil_diam", None), t_now)
        if getattr(feats, "blink", 0) > 0:
            blink_times.append(t_now)
        pupil_centers = {k: v for k, v in meta.get("points", {}).items() if k in ("pupilL", "pupilR")}
        pupil_text = f"Pupil {ps.event}  z={ps.zscore:+.1f}  roc={ps.roc:+.02f}/s  cog={ps.cognitive_index:.2f}"

        # tremor (accepts frame+meta)
        tremor.update(frame, meta)

        # MSK pose throttled
        if frame_idx % max(1, POSE_EVERY_N_FRAMES) == 0:
            angles, overlay = msk.process(frame)
        else:
            angles, overlay = msk.rom, np.zeros_like(frame)

        # tests update (draws on main frame)
        games.update_rhythm(frame, meta)
        games.update_anti(frame, meta, feats)
        games.update_trail(frame, meta)

        # overlays
        draw = frame.copy()
        draw = MSKAnalyzer.draw(draw, overlay, angles, msk.rom)
        draw_overlays(draw, feats, meta, ps, pupil_centers)

        # attention tiles (throttled)
        now_ms = int(time.time() * 1000)
        if last_edges is None or (now_ms - last_attn_ms) >= attn_skip_ms:
            maps = attn.compute(frame)
            last_edges, last_flow, last_sal, last_mot = (
                maps["edges"], maps["flow"], maps["sal"], maps["motion"]
            )
            last_attn_ms = now_ms

        # main video
        ui.video_ph.image(cv2.cvtColor(draw, cv2.COLOR_BGR2RGB), width="stretch")

        # ladder
        H, W = frame.shape[:2]
        strip_h = ui.desired_attn_height(W)
        edges_color = cv2.resize(last_edges, (W, strip_h), interpolation=cv2.INTER_AREA)
        ui.ladder_ph.image(edges_color, caption="Edges • Attention ladder", width="stretch")

        # 3 attention cards
        card_w = W // 3
        card_h = ui.desired_attn_height(card_w)
        ui.attn1_ph.image(cv2.resize(last_flow, (card_w, card_h)), caption="Motion (Optical flow magnitude)", width="stretch")
        ui.attn2_ph.image(cv2.resize(last_sal,  (card_w, card_h)), caption="Saliency (Spectral residual)",  width="stretch")
        ui.attn3_ph.image(cv2.resize(last_mot,  (card_w, card_h)), caption="Motion accumulation",           width="stretch")

        # activity vector -> NCP graph (stateful)
        activity = [
            clamp(feats.eyes_open * 0.6 + feats.mouth_open * 0.4, 0, 1),
            clamp(0.5 * abs(feats.yaw) / 45.0 + 0.5 * feats.stress, 0, 1),
            clamp(0.6 * feats.speaking + 0.4 * (feats.left_hand or feats.right_hand), 0, 1),
            clamp(0.5 * (feats.left_hand + feats.right_hand), 0, 1),
        ]
        activity_ema = _ema(activity_ema, activity, ACTIVITY_EMA_ALPHA)
        ncp.inject(activity_ema)
        ncp.step(1.0 / max(ui.target_fps, 1))

        if now_ms - last_plot_ms >= FRAME_SKIP_MIN_MS:
            fig = ncp.figure()  # draw current state
            ui.graph_ph.plotly_chart(fig, width="stretch", theme="streamlit")
            last_plot_ms = now_ms

        # right panel
        if st.session_state.get("demo_v7", False):
            res = demo_estimate(frame, meta)
            ui.cog_ph.markdown(demo_render(res))
        else:
            cg = analyze_cognitive(feats, ps, blink_count_window=len(blink_times))
            extra = games.footer_text()
            ui.cog_ph.markdown(
                f"**Emotion & Cognitive Screen**\n\n"
                f"Risk index: **{int(cg['risk']*100)}%**\n"
                f"{cg['summary'] if cg['summary'] else 'No prominent risk cues detected.'}\n\n"
                f"**Test-derived risks** — {extra}"
            )

        # status (single line, updated in place)
        dt = t_now - t_prev
        t_prev = t_now
        fps_s.append(1.0 / max(dt, 1e-3))
        status.update(fps_s, blink_times, feats.yaw, feats.emotion, pupil_text)

        frame_idx += 1
        time.sleep(max(0.0, (1.0 / ui.target_fps) - (time.time() - t_now)))

    cap.release()
    st.markdown(f"<div style='text-align:center;color:#98a2b3'>{APP_FOOT}</div>", unsafe_allow_html=True)

__all__ = ["main"]
