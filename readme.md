# 🧠 Humanoid NeuroGraph  
**Live Multimodal Neuro-Perceptual Analysis**

A real-time Streamlit platform for analyzing facial expressions, pupil reactivity, lips and articulation, finger and body pose, tremor stability, and deep demographic attributes (age, sex, glasses, beard, mask).  
Developed for research in **human-robot perception**, **affective computing**, and **neuro-behavioral diagnostics**.

---

## 🚀 Features

- **Face & Iris Tracking** – gaze, blink, and pupil dilation analysis  
- **Lips & Articulation** – live lip motion and speech articulation overlay  
- **Hand & Finger Pose** – 21-joint skeletons (MediaPipe Hands)  
- **Full-body MSK Overlay** – joint angles and posture tracking  
- **Cognitive Mini-Games** – finger rhythm, anti-saccade, and trail-making tasks  
- **Emotion & Cognitive Screen** – risk index and current affect state  
- **Demographics (DL)** – age, sex, glasses, beard, mask via InsightFace (CPU)  
- **Attention Visualization** – edge, flow, saliency, and motion tiles  
- **Tremor Analysis** – fine hand oscillation frequency/rms and stability

---

## 🧩 Installation (macOS & Windows)

### ✅ Prerequisites
- **Python 3.10 or 3.11**  
  (MediaPipe and ONNXRuntime are not yet stable on 3.12/3.13)
- **Camera access** enabled in OS settings
- **Git** (optional)

---

### 🧠 macOS Setup

```bash
# 1. Clone
git clone https://github.com/<your-user>/humanoid_neurograph.git
cd humanoid_neurograph

# 2. Optional: use Python 3.10 via pyenv
# brew install pyenv
# pyenv install 3.10.14 && pyenv local 3.10.14

# 3. Virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 4. Upgrade base tools
python -m pip install --upgrade pip setuptools wheel

# 5. Install requirements
pip install -r requirements.txt

# 6. Optional: improve reload speed
xcode-select --install || true
pip install watchdog

# 7. Run the app
streamlit run main.py
