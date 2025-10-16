# app/ui.py
from __future__ import annotations
import streamlit as st
import numpy as np
from collections import deque

TITLE = "Humanoid NeuroGraph — Live Multimodal Neuro-Perceptual Analysis"

class NeuroGraphUI:
    def __init__(self, default_fps: int = 30):
        st.set_page_config(page_title="NeuroGraph v7", layout="wide")
        st.markdown(
            f"<h1 style='margin-top:-12px'>{TITLE}</h1>",
            unsafe_allow_html=True,
        )

        # Top toolbar: FPS slider + toggles (mirror & run/pause stay!)
        cols = st.columns([6, 1.2, 1.2])
        with cols[0]:
            self.target_fps = st.slider("Target FPS", 5, 60, value=default_fps)
        with cols[1]:
            st.session_state.setdefault("run_v7", True)
            run = st.toggle("Run / Pause", value=st.session_state.get("run_v7", True))
            st.session_state["run_v7"] = run
        with cols[2]:
            st.session_state.setdefault("mirror_v7", True)
            mir = st.toggle("Mirror", value=st.session_state.get("mirror_v7", True))
            st.session_state["mirror_v7"] = mir

        # Main two-column layout (video left, graph + right panel on the right)
        left, right = st.columns([1.45, 1.0], gap="large")

        with left:
            self.video_ph = st.empty()
        with right:
            self.graph_ph = st.empty()
            self.cog_text_ph = st.empty()
            self.expr_ph = st.empty()   # 9-way expression bars
            self.demo_ph = st.empty()   # demographics (always visible)

        st.divider()

        # Attention ladder row
        self.ladder_ph = st.empty()

        # Three “cards” row under the ladder
        c1, c2, c3 = st.columns(3, gap="large")
        with c1:
            self.attn1_ph = st.empty()
        with c2:
            self.attn2_ph = st.empty()
        with c3:
            self.attn3_ph = st.empty()

        st.divider()

        # Cognitive tests row (three groups)
        t1, t2, t3 = st.columns(3, gap="large")
        self.tests = (t1, t2, t3)

        st.divider()
        self.status_ph = st.empty()

    # ------- sizing helpers (no deprecation params anywhere) -------
    def desired_attn_height(self, W: int) -> int:
        # a slightly taller strip than before so you can “see” the ladder
        return max(120, int(0.14 * W))

    def desired_card_size(self, W: int) -> tuple[int, int]:
        # 3 cards per row; keep ~6:8 ratio (0.75) as requested earlier
        card_w = max(240, int(0.29 * W))
        card_h = int(card_w * 0.75)
        return card_w, card_h

    # ------- single-line status bar (no spam) -------
    def render_status(
        self,
        fps_deque: deque,
        blink_times: deque,
        yaw_deg: float,
        emotion: str,
        pupil_text: str,
    ):
        fps = np.mean(fps_deque) if fps_deque else 0.0
        blinks = sum(1 for b in blink_times if (blink_times[-1] - b) <= 10.0) if blink_times else 0
        msg = (
            f"**FPS {fps:0.1f}** | **Blinks (win)** {blinks} | **Yaw** {yaw_deg:+.0f}° "
            f"| **Emotion** {emotion} | {pupil_text}"
        )
        self.status_ph.markdown(msg)
