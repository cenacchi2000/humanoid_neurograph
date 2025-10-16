# humanoid_neurograph/app/config.py
from dataclasses import dataclass

APP_TITLE = "Humanoid NeuroGraph — Live Multimodal Neuro-Perceptual Analysis"
APP_SUBTITLE = "MediaPipe + (optional) OpenCV fallback • Live webcam • Neural routing graph • Cognitive tests"
APP_FOOT  = "Demo-grade analysis. Not a medical device. MediaPipe preferred; OpenCV Haar fallback."

# Plotly: prefer WebGL for large point sets
USE_WEBGL = True

# Graph
PACKETS_MAX = 600
EDGE_CONNECT_PROB = (0.25, 0.20, 0.15)

# Runtime defaults
DEFAULT_SEED = 0
DEFAULT_TARGET_FPS = 30

# ---- Attention maps / layout ----
# Compute maps at most this often
ATTN_TARGET_HZ = 8
# Downscale factor for attention computation (integer >=1). 3 is a good balance.
ATTN_DOWNSCALE = 3
# Desired aspect ratio used when sizing attention strips
ATTN_ASPECT = 2.8  # width/height

# Smoothness / throttling
FRAME_SKIP_MIN_MS = 45         # don't redraw graph more often than this
ACTIVITY_EMA_ALPHA = 0.25      # smoothing for activity vector

# Pose / tremor throttles
POSE_EVERY_N_FRAMES = 2        # run pose every N frames (>=1)
TREMOR_FS_TARGET = 30          # tremor sampling rate (Hz), we subsample to this
TREMOR_WINDOW_SEC = 4.0        # FFT window length
TREMOR_BAND = (3.0, 12.0)      # physiological tremor band (Hz)
TREMOR_MIN_RMS_PX = 2.0        # minimum rms amplitude to consider
TREMOR_BAND_RATIO_THR = 2.2    # band power / out-of-band power threshold

# OpenCV threading (set to 1 to reduce jitter on some Macs)
OPENCV_NUM_THREADS = 1

# UI keys used in session_state (keep stable to avoid duplicate widgets)
class UI_KEYS:
    RUN = "run_v7"
    MIRROR = "mirror_v7"
    DEMO = "demo_v7"
    FPS = "fps_v7"
