# humanoid_neurograph/app/games.py
from __future__ import annotations
import time, random
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import numpy as np
import cv2

# ------------------------------------------------------------------------------------------
# LEGACY SIMPLE GAMES (kept so older code keeps working)
# ------------------------------------------------------------------------------------------
class Games:
    def __init__(self):
        self.reset()

    def reset(self):
        self.trl = {"active":False, "centers":[], "order":[], "hits":0, "mean_err":0.0, "t0":0.0}
        self.anti = {"active":False, "cue":None, "hits":0, "fa":0, "t0":0.0}
        self.nblink={"active":False, "n":3, "count":0, "hits":0, "fa":0, "t0":0.0}

    def trail_start(self, W=420, H=280):
        self.trl.update(active=True, t0=time.time(), hits=0, order=list(range(8)), mean_err=0.0)
        rnd = np.random.RandomState(0)
        self.trl["centers"] = [(int(60+rnd.randint(0,W-120)), int(60+rnd.randint(0,H-120))) for _ in range(8)]
    def trail_frame(self):
        W,H=420,280
        img = np.zeros((H,W,3), np.uint8)
        if not self.trl["active"]: return img
        for i,c in enumerate(self.trl["centers"]):
            cv2.circle(img, c, 18, (200,180,100), -1)
            cv2.putText(img, str(i+1), (c[0]-6,c[1]+6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
        return img
    def anti_start(self): self.anti.update(active=True, t0=time.time(), hits=0, fa=0)
    def anti_frame(self):
        img = np.zeros((280,420,3), np.uint8)
        if not self.anti["active"]:
            cv2.putText(img,"Start anti-saccade", (60,140), cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)
            return img
        cv2.putText(img, "Look opposite of the dot", (34,30), cv2.FONT_HERSHEY_SIMPLEX,0.6,(200,200,200),1)
        x = 60 if int(time.time()*1.2)%2==0 else 360
        cv2.circle(img,(x,140),16,(0,180,255),-1)
        return img
    def nblink_start(self): self.nblink.update(active=True, t0=time.time(), count=0, hits=0, fa=0)
    def nblink_frame(self):
        img = np.zeros((280,420,3), np.uint8)
        if not self.nblink["active"]:
            cv2.putText(img,"N-Blink", (150,140), cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
            return img
        phase = int(time.time()*0.7)%5
        color = (90,200,90) if phase in (1,3) else (80,80,80)
        cv2.circle(img,(210,140),90,color,-1)
        cv2.putText(img,"WAIT" if phase!=1 else "BLINK",(168,146),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)
        return img


# ------------------------------------------------------------------------------------------
# NEW CLINICAL-STYLE TESTS (GamesPro) — richer analytics; nothing removed above
# ------------------------------------------------------------------------------------------
@dataclass
class TrialLog:
    ts: float
    value: float

@dataclass
class TestResult:
    started: bool = False
    finished: bool = False
    t0: float = 0.0
    t_end: float = 0.0
    metrics: Dict[str, float] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

@dataclass
class RiskSummary:
    parkinson: float = 0.0
    alz_mci: float = 0.0
    ftd_exec: float = 0.0

@dataclass
class RhythmPhaseConfig:
    name: str
    duration: float
    target_hz: float
    instruction: str

@dataclass
class TapEvent:
    ts: float
    phase: int
    y_px: float

def _centroid(box: Tuple[int,int,int,int]) -> Tuple[float,float]:
    x,y,w,h = box
    return (x + 0.5*w, y + 0.5*h)

def _safe_std(a: np.ndarray) -> float:
    if a.size == 0: return 0.0
    return float(np.std(a))

def _hz_from_peaks(ts: List[float]) -> float:
    if len(ts) < 2: return 0.0
    isi = np.diff(np.array(ts))
    m = np.mean(isi)
    return 0.0 if m <= 1e-3 else float(1.0/m)

class GamesPro:
    def __init__(self, cam_size: Tuple[int,int]=(640,480), fps_hint: int=30):
        self.W, self.H = cam_size
        self.fs = max(10, int(fps_hint))
        self.now = time.time

        # Rhythm/tremor
        self.rhythm = TestResult()
        self.rhythm_roi = None
        self.rhythm_peaks: List[float] = []
        self.rhythm_last_y = None
        self.rhythm_last_peak_y = None
        self.rhythm_deriv_ts: List[TrialLog] = []
        self.rhythm_deriv_all: List[TrialLog] = []
        self.rhythm_active = False
        self.rhythm_win = 3.0
        self.rhythm_phases: List[RhythmPhaseConfig] = [
            RhythmPhaseConfig("Warm-up", 4.0, 1.4, "Light taps to get started"),
            RhythmPhaseConfig("Paced", 8.0, 2.0, "Match a steady tapping tempo"),
            RhythmPhaseConfig("Burst", 6.0, 2.6, "Accelerate; keep taps rhythmic"),
        ]
        self.rhythm_phase_idx = -1
        self.rhythm_phase_start = 0.0
        self.rhythm_taps: List[TapEvent] = []

        # Anti-saccade
        self.anti = TestResult()
        self.anti_cue_side = 0
        self.anti_next_jump = 0.0
        self.anti_hits = 0
        self.anti_trials = 0
        self.anti_lat: List[float] = []
        self.anti_max_trials = 18
        self.anti_hold = 0.7

        # Trail
        self.trail = TestResult()
        self.trail_points: List[Tuple[int,int]] = []
        self.trail_target_idx = 0
        self.trail_err = 0
        self.trail_radius = 24

        self.risk = RiskSummary()

    # ---- helpers
    def _hand_centroid(self, meta) -> Optional[Tuple[float,float]]:
        hands = meta.get("hands") or []
        if not hands: return None
        centers = [(_centroid(h["box"]), h) for h in hands]
        cx, cy = self.W/2, self.H/2
        centers.sort(key=lambda ch: (ch[0][0]-cx)**2 + (ch[0][1]-cy)**2)
        return centers[0][0]

    def _gaze_side(self, meta, feats) -> float:
        nose = meta["points"].get("nose")
        pL   = meta["points"].get("pupilL")
        pR   = meta["points"].get("pupilR")
        if nose and pL and pR:
            pc = ((pL[0]+pR[0])//2, (pL[1]+pR[1])//2)
            dx = float(pc[0] - nose[0])
            if abs(dx) > 4:
                return -1.0 if dx < 0 else +1.0
        if abs(feats.yaw) > 5.0:
            return -1.0 if feats.yaw < 0 else +1.0
        return 0.0

    # ---- Test 1: Finger Rhythm & Tremor
    def start_rhythm(self):
        self.rhythm = TestResult(started=True, t0=self.now())
        self.rhythm_peaks.clear()
        self.rhythm_deriv_ts.clear()
        self.rhythm_deriv_all.clear()
        self.rhythm_taps.clear()
        self.rhythm_last_y = None
        self.rhythm_last_peak_y = None
        self.rhythm_active = True
        self.rhythm_phase_idx = 0
        self.rhythm_phase_start = self.rhythm.t0
        w,h = int(self.W*0.28), int(self.H*0.45)
        x0,y0 = self.W - w - 16, (self.H-h)//2
        self.rhythm_roi = (x0,y0,w,h)

    def stop_rhythm(self):
        if not self.rhythm.started or self.rhythm.finished: return
        self.rhythm.finished = True
        self.rhythm.t_end = self.now()
        self.rhythm_active = False
        duration = max(1e-3, self.rhythm.t_end - self.rhythm.t0)
        tap_times = [tap.ts for tap in self.rhythm_taps] or list(self.rhythm_peaks)
        tap_times.sort()
        n_taps = len(tap_times)
        overall_hz = _hz_from_peaks(tap_times) if n_taps >= 2 else float(n_taps) / duration
        isi_vals = np.diff(np.array(tap_times)) if n_taps >= 2 else np.array([])
        overall_isi_std = _safe_std(isi_vals) if isi_vals.size else 0.0
        cadence_cv = float(np.std(isi_vals) / max(1e-3, np.mean(isi_vals))) if isi_vals.size else 0.0

        fatigue_ratio = 1.0
        if tap_times:
            midpoint = self.rhythm.t0 + duration * 0.5
            early = sum(1 for ts in tap_times if ts <= midpoint)
            late = n_taps - early
            fatigue_ratio = float(late) / max(1.0, float(early))

        tremor_vals = np.array([log.value for log in self.rhythm_deriv_all], dtype=float)
        tremor_rms = float(np.sqrt(np.mean(np.square(tremor_vals)))) if tremor_vals.size else 0.0

        phase_details = []
        for idx, phase in enumerate(self.rhythm_phases):
            taps = [tap for tap in self.rhythm_taps if tap.phase == idx]
            phase_times = [tap.ts for tap in taps]
            taps_n = len(phase_times)
            if taps_n >= 2:
                phase_hz = _hz_from_peaks(phase_times)
                phase_isi_std = _safe_std(np.diff(np.array(phase_times)))
            elif taps_n == 1:
                phase_hz = float(taps_n) / max(phase.duration, 1e-3)
                phase_isi_std = 0.0
            else:
                phase_hz = 0.0
                phase_isi_std = 0.0
            accuracy = 1.0 - min(1.0, abs(phase_hz - phase.target_hz) / max(phase.target_hz, 1e-3))
            completion = float(taps_n) / max(1e-3, phase.target_hz * phase.duration)
            phase_details.append({
                "name": phase.name,
                "hz": float(phase_hz),
                "isi_std": float(phase_isi_std),
                "accuracy": float(accuracy),
                "completion": float(completion),
                "taps": float(taps_n),
                "target": float(phase.target_hz),
            })

        phase_accuracy_mean = float(np.mean([p["accuracy"] for p in phase_details])) if phase_details else 0.0
        phase_error_mean = float(np.mean([abs(p["hz"] - p["target"]) / max(p["target"], 1e-3) for p in phase_details])) if phase_details else 0.0

        self.rhythm.metrics = {
            "duration_s": duration,
            "overall_tapping_hz": float(overall_hz),
            "overall_isi_std_s": float(overall_isi_std),
            "cadence_cv": float(cadence_cv),
            "fatigue_ratio": float(fatigue_ratio),
            "tremor_rms_px": float(tremor_rms),
            "phase_accuracy_mean": float(phase_accuracy_mean),
            "tap_count": float(n_taps),
        }
        for idx, detail in enumerate(phase_details, start=1):
            base = f"phase{idx}"
            self.rhythm.metrics[f"{base}_taps"] = detail["taps"]
            self.rhythm.metrics[f"{base}_hz"] = detail["hz"]
            self.rhythm.metrics[f"{base}_isi_std_s"] = detail["isi_std"]
            self.rhythm.metrics[f"{base}_target_hz"] = detail["target"]
            self.rhythm.metrics[f"{base}_completion"] = min(1.0, detail["completion"])

        self.rhythm.notes.append(f"Overall cadence {overall_hz:.2f} Hz with variability CV {cadence_cv:.2f}.")
        for detail in phase_details:
            status = "on pace" if detail["accuracy"] > 0.82 else ("slow" if detail["hz"] < detail["target"] else "fast")
            self.rhythm.notes.append(
                f"{detail['name']}: {detail['hz']:.2f} Hz vs target {detail['target']:.2f} Hz ({int(detail['taps'])} taps, {status})."
            )
        self.rhythm.notes.append(f"Fatigue ratio (late/early taps): {fatigue_ratio:.2f}; tremor RMS: {tremor_rms:.2f} px.")

        vr = 0.0
        if overall_hz < 1.6: vr += 0.25
        if overall_hz < 1.2: vr += 0.20
        if overall_isi_std > 0.12: vr += 0.20
        if cadence_cv > 0.18: vr += 0.20
        if fatigue_ratio < 0.75: vr += 0.10
        if tremor_rms > 3.5: vr += 0.10
        if phase_error_mean > 0.25: vr += 0.15
        self.risk.parkinson = min(1.0, vr)

    def update_rhythm(self, frame_bgr, meta):
        if not self.rhythm_active or not self.rhythm.started: return
        if not self.rhythm_roi: return
        t = self.now()
        if self.rhythm_phase_idx < 0:
            self.rhythm_phase_idx = 0
            self.rhythm_phase_start = t
        if self.rhythm_phase_idx >= len(self.rhythm_phases):
            self.stop_rhythm()
            return
        phase = self.rhythm_phases[self.rhythm_phase_idx]
        phase_elapsed = t - self.rhythm_phase_start
        if phase_elapsed >= phase.duration:
            self.rhythm_phase_idx += 1
            self.rhythm_phase_start = t
            self.rhythm_last_y = None
            self.rhythm_last_peak_y = None
            self.rhythm_deriv_ts.clear()
            if self.rhythm_phase_idx >= len(self.rhythm_phases):
                self.stop_rhythm()
                return
            phase = self.rhythm_phases[self.rhythm_phase_idx]
            phase_elapsed = 0.0
        x0,y0,w,h = self.rhythm_roi
        roi = frame_bgr[y0:y0+h, x0:x0+w]
        if roi.size == 0: return
        gry = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        col = gry.mean(axis=1)
        y = int(np.argmax(cv2.GaussianBlur(col, (9,1), 0)))
        if self.rhythm_last_y is None: self.rhythm_last_y = y
        dy = y - self.rhythm_last_y
        self.rhythm_last_y = y
        log = TrialLog(t, float(dy))
        self.rhythm_deriv_ts.append(log)
        self.rhythm_deriv_all.append(log)
        while self.rhythm_deriv_ts and (t - self.rhythm_deriv_ts[0].ts) > self.rhythm_win:
            self.rhythm_deriv_ts.pop(0)
        if len(self.rhythm_deriv_ts) >= 3:
            d1 = self.rhythm_deriv_ts[-2].value
            d2 = self.rhythm_deriv_ts[-1].value
            if d1 > 0 and d2 <= 0:
                if (not self.rhythm_peaks) or (t - self.rhythm_peaks[-1]) > 0.25:
                    self.rhythm_peaks.append(t)
                    self.rhythm_taps.append(TapEvent(t, self.rhythm_phase_idx, float(y)))
                    self.rhythm_last_peak_y = y

        phase_timestamps = [tap.ts for tap in self.rhythm_taps if tap.phase == self.rhythm_phase_idx]
        if len(phase_timestamps) >= 2:
            phase_hz = _hz_from_peaks(phase_timestamps)
        elif len(phase_timestamps) == 1 and phase_elapsed > 0:
            phase_hz = 1.0 / max(phase_elapsed, 1e-3)
        else:
            phase_hz = 0.0
        prog = min(1.0, phase_elapsed / max(phase.duration, 1e-3))
        remaining = max(0.0, phase.duration - phase_elapsed)

        cv2.rectangle(frame_bgr, (x0,y0), (x0+w,y0+h), (70,230,230), 2)
        title_y = max(22, y0 - 16)
        cv2.putText(frame_bgr,
                    f"Finger Rhythm {self.rhythm_phase_idx+1}/{len(self.rhythm_phases)}",
                    (x0, title_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180,255,255), 2, cv2.LINE_AA)
        instr_y = min(self.H - 12, y0 + 18)
        cv2.putText(frame_bgr,
                    f"{phase.name}: {phase.instruction}",
                    (x0 + 6, instr_y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200,240,255), 2, cv2.LINE_AA)
        bar_y0 = min(self.H - 12, y0 + h + 4)
        bar_y1 = min(self.H - 6, bar_y0 + 6)
        cv2.rectangle(frame_bgr, (x0, bar_y0), (x0 + w, bar_y1), (80, 90, 90), 1)
        cv2.rectangle(frame_bgr, (x0, bar_y0), (x0 + int(w * prog), bar_y1), (70, 230, 230), -1)
        stats_y = min(self.H - 8, bar_y1 + 12)
        cv2.putText(frame_bgr,
                    f"Target {phase.target_hz:.1f} Hz | current {phase_hz:.1f} Hz | {remaining:.1f}s",
                    (x0, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (180,255,255), 2, cv2.LINE_AA)

    # ---- Test 2: Anti-saccade
    def start_anti(self):
        self.anti = TestResult(started=True, t0=self.now())
        self.anti_hits = 0; self.anti_trials = 0; self.anti_lat.clear()
        self.anti_next_jump = 0.0
        self.anti_cue_side = 0

    def stop_anti(self):
        if not self.anti.started or self.anti.finished: return
        self.anti.finished = True
        self.anti.t_end = self.now()
        acc = (self.anti_hits / max(1, self.anti_trials))
        lat = float(np.median(self.anti_lat)) if self.anti_lat else 0.0
        self.anti.metrics = {"accuracy": acc, "median_latency_s": lat, "trials": float(self.anti_trials)}
        r = 0.0
        if acc < 0.7: r += 0.5
        if lat > 0.45: r += 0.4
        self.risk.ftd_exec = min(1.0, r)

    def update_anti(self, frame_bgr, meta, feats):
        if not self.anti.started or self.anti.finished: return
        t = self.now()
        if t >= self.anti_next_jump:
            self.anti_trials += 1
            self.anti_cue_side = -1 if (random.random()<0.5) else +1
            self.anti_next_jump = t + self.anti_hold
            self.anti_cue_t = t
        y = int(self.H*0.28)
        x = int(self.W*0.18) if self.anti_cue_side<0 else int(self.W*0.82)
        cv2.circle(frame_bgr, (x, y), 16, (0,180,255), -1)
        cv2.putText(frame_bgr, "ANTI-SACCADE: look to the OPPOSITE side",
                    (int(self.W*0.18), y-24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (210,210,210), 2, cv2.LINE_AA)
        side = self._gaze_side(meta, feats)
        if side != 0:
            if (side == -self.anti_cue_side):
                lat = t - self.anti_cue_t
                if lat > 0.12:
                    self.anti_hits += 1
                    self.anti_lat.append(lat)
                    self.anti_next_jump = t
        if self.anti_trials >= self.anti_max_trials:
            self.stop_anti()

    # ---- Test 3: Air Trail Making
    def start_trail(self):
        self.trail = TestResult(started=True, t0=self.now())
        rnd = np.random.RandomState(1)
        pts = []
        W,H = self.W, self.H
        for _ in range(10):
            pts.append((int(0.18*W + rnd.rand()*0.64*W),
                        int(0.42*H + rnd.rand()*0.22*H)))
        self.trail_points = pts
        self.trail_target_idx = 0
        self.trail_err = 0

    def stop_trail(self):
        if not self.trail.started or self.trail.finished: return
        self.trail.finished = True
        self.trail.t_end = self.now()
        dur = self.trail.t_end - self.trail.t0
        self.trail.metrics = {"duration_s": dur, "errors": float(self.trail_err)}
        r = 0.0
        if dur > 22: r += 0.5
        if self.trail_err >= 3: r += 0.4
        self.risk.alz_mci = min(1.0, r)

    def update_trail(self, frame_bgr, meta):
        if not self.trail.started or self.trail.finished: return
        for i, c in enumerate(self.trail_points):
            color = (60,180,120) if i < self.trail_target_idx else (200,180,100)
            cv2.circle(frame_bgr, c, self.trail_radius, color, 2)
            cv2.putText(frame_bgr, str(i+1), (c[0]-6, c[1]+6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        tgt_idx = self.trail_target_idx
        tgt = self.trail_points[tgt_idx]
        cv2.circle(frame_bgr, tgt, self.trail_radius+4, (255,210,60), 2)
        hc = self._hand_centroid(meta)
        if hc:
            cv2.circle(frame_bgr, (int(hc[0]), int(hc[1])), 6, (180,140,255), -1)
            if (hc[0]-tgt[0])**2 + (hc[1]-tgt[1])**2 <= (self.trail_radius**2):
                self.trail_target_idx += 1
                if self.trail_target_idx >= len(self.trail_points):
                    self.stop_trail()
            else:
                for j, c in enumerate(self.trail_points):
                    if j == tgt_idx: continue
                    if (hc[0]-c[0])**2 + (hc[1]-c[1])**2 <= (self.trail_radius**2):
                        self.trail_err += 1
                        break

    def risk_summary(self) -> RiskSummary:
        return self.risk

    def footer_text(self) -> str:
        r = self.risk_summary()
        return (f"PD risk ~{int(100*r.parkinson)}% • "
                f"AD/MCI risk ~{int(100*r.alz_mci)}% • "
                f"Exec/FTD risk ~{int(100*r.ftd_exec)}%")

__all__ = ["Games", "GamesPro", "TestResult", "RiskSummary"]
