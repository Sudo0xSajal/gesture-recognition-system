"""
integrations/realtime/run_webcam.py
====================================
STEP 1 → STEP 6 — Complete Real-Time Gesture System

Full pipeline, live on webcam:
  Step 1: RGB camera captures frame (OpenCV)
  Step 2: MediaPipe detects 21 hand landmarks
  Step 3: AI model classifies gesture (CNN / LSTM / SVM)
  Step 4: Decision engine reads clinical meaning
  Step 5: IoT command + caregiver alert triggered
  Step 6: Voice/visual feedback + event log

Supports all three model types from Ma'am's document:
  cnn      — image-based CNN (most accurate for single frame)
  lstm     — RNN/LSTM (best for motion gestures: wave, shake)
  svm      — landmark-based SVM (fastest, works on any CPU)

Run
---
    bash realtime.sh -c log/<exp>/best_model.pt      -m cnn
    bash realtime.sh -c log/<exp>/best_model.pt      -m lstm
    bash realtime.sh -c log/<exp>/svm_model.joblib   -m svm
    bash realtime.sh -c log/<exp>/svm_model.joblib   -m svm --save-log events.json
"""

import sys
import json
import time
import argparse
import collections
import numpy as np
from pathlib import Path
from datetime import datetime

import cv2
import mediapipe as mp
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))

from utils.config import (
    GESTURE_NAMES, GESTURE_ACTIONS, CAMERA_CONFIG,
    INFERENCE_CONFIG, NUM_CLASSES,
)


# =============================================================================
# TEMPORAL SMOOTHER
# =============================================================================
class TemporalSmoother:
    """
    Majority-vote or EWA smoothing over a rolling window of predictions.
    Prevents single-frame noise from triggering false clinical alerts.

    is_stable() gate: only fires when N consecutive frames agree AND
    mean confidence exceeds the threshold. This is the key safety check
    that prevents false positive emergency alarms.
    """

    def __init__(self, window: int = 7, method: str = "majority",
                 alpha: float = 0.4):
        self.window   = window
        self.method   = method
        self.alpha    = alpha
        self.buf      = collections.deque(maxlen=window)
        self.ewa      = np.zeros(NUM_CLASSES, dtype=np.float32)

    def update(self, gid: int, probs: np.ndarray, conf: float):
        self.buf.append((gid, conf))
        if self.method == "ewa":
            self.ewa = self.alpha * probs + (1 - self.alpha) * self.ewa

    def get(self) -> tuple:
        if not self.buf:
            return 0, 0.0
        if self.method == "majority":
            counts = collections.Counter(g for g, _ in self.buf)
            gid    = counts.most_common(1)[0][0]
            confs  = [c for g, c in self.buf if g == gid]
            return gid, float(np.mean(confs))
        else:
            gid = int(np.argmax(self.ewa))
            return gid, float(self.ewa[gid])

    def is_stable(self, threshold: float) -> bool:
        """True when buffer is full, unanimous, and confident enough."""
        if len(self.buf) < self.window:
            return False
        ids = [g for g, _ in self.buf]
        if len(set(ids)) > 1:
            return False
        return float(np.mean([c for _, c in self.buf])) >= threshold

    def clear(self):
        self.buf.clear()
        self.ewa = np.zeros(NUM_CLASSES, dtype=np.float32)


# =============================================================================
# EVENT LOGGER
# =============================================================================
class EventLogger:
    """Logs stable gesture events to JSON file for caregiver integration."""

    def __init__(self, log_path: Path = None):
        self.log_path = log_path
        self.events   = []
        self.last_gid = -1

    def log(self, gid: int, conf: float) -> bool:
        if gid == self.last_gid:
            return False
        action = GESTURE_ACTIONS.get(gid, {})
        event  = {
            "timestamp":    datetime.now().isoformat(),
            "gesture_id":   gid,
            "gesture_name": GESTURE_NAMES[gid],
            "confidence":   round(conf, 4),
            "action":       action.get("message",  ""),
            "alert":        action.get("alert",    False),
            "iot_command":  action.get("iot",      None),
            "priority":     action.get("priority", "NORMAL"),
        }
        self.events.append(event)
        self.last_gid = gid
        if self.log_path:
            with open(self.log_path, "w") as f:
                json.dump(self.events, f, indent=2)
        return True

    def reset(self):
        self.last_gid = -1


# =============================================================================
# MODEL LOADER
# =============================================================================
def _normalise(vec: np.ndarray) -> np.ndarray:
    pts   = vec.reshape(21, 3)
    pts   = pts - pts[0]
    scale = np.max(np.abs(pts)) + 1e-8
    return (pts / scale).flatten().astype(np.float32)


def _load_model(checkpoint: Path, model_type: str):
    if model_type in ("cnn", "mobilenet"):
        from models.cnn_model import GestureCNN, MobileNetV2Transfer
        cls = GestureCNN if model_type == "cnn" else MobileNetV2Transfer
        model = cls()
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state["model_state"])
        model.eval()
        return model
    elif model_type == "lstm":
        from models.lstm_model import LSTMModel
        model = LSTMModel()
        state = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state["model_state"])
        model.eval()
        return model
    elif model_type == "svm":
        from models.svm_classifier import SVMClassifier
        return SVMClassifier.load(checkpoint)
    raise ValueError(f"Unknown model type: {model_type}")


def _infer(model, model_type: str, vec: np.ndarray, bgr: np.ndarray,
           lstm_buf: collections.deque = None) -> tuple:
    """Run one inference step. Returns (gesture_id, probs, confidence)."""

    if model_type == "svm":
        probs = model.predict_proba(vec.reshape(1, -1))[0]

    elif model_type in ("cnn", "mobilenet"):
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
            ToTensorV2(),
        ])
        rgb   = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        img_t = transform(image=rgb)["image"].unsqueeze(0)
        with torch.no_grad():
            probs = F.softmax(model(img_t), dim=1).squeeze(0).numpy()

    elif model_type == "lstm":
        lstm_buf.append(vec)
        if len(lstm_buf) < lstm_buf.maxlen:
            return None, None, 0.0
        seq = torch.from_numpy(np.stack(list(lstm_buf))).unsqueeze(0)
        with torch.no_grad():
            probs = F.softmax(model(seq), dim=1).squeeze(0).numpy()

    gid  = int(np.argmax(probs))
    conf = float(probs[gid])
    return gid, probs, conf


# =============================================================================
# HUD DRAWING
# =============================================================================
def _draw_hud(frame, gid, gname, conf, stable, fps, recording):
    h, w   = frame.shape[:2]
    col    = (50,200,50) if conf >= 0.75 else (50,200,200) if conf >= 0.6 else (50,50,200)
    action = GESTURE_ACTIONS.get(gid, {})
    alert  = action.get("alert", False)
    priority = action.get("priority", "NORMAL")

    # Top bar
    cv2.rectangle(frame, (0,0), (w,85), (15,15,15), -1)
    cv2.circle(frame, (18,42), 9, col if stable else (80,80,80), -1)
    cv2.putText(frame, f"[{gid:02d}] {gname}",
                (35,35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, col, 2)
    cv2.putText(frame, f"Conf: {conf*100:.1f}%   Stable: {'YES' if stable else 'NO'}",
                (35,65), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (190,190,190), 1)

    # Alert banner
    if stable and alert:
        bc = (0,0,180) if priority == "HIGH" else (0,100,180)
        cv2.rectangle(frame, (0,85), (w,118), bc, -1)
        cv2.putText(frame, action.get("message","")[:70],
                    (8,110), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255,255,255), 1)

    # FPS + REC
    cv2.putText(frame, f"FPS:{fps:.1f}",
                (w-100,28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160,160,160), 1)
    if recording:
        cv2.circle(frame, (w-18,52), 7, (0,0,220), -1)
        cv2.putText(frame, "REC", (w-52,58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,220), 1)

    # Confidence bar
    cv2.rectangle(frame, (0,h-9), (int(w*conf),h), col, -1)
    cv2.rectangle(frame, (0,h-9), (w,h), (60,60,60), 1)

    # IoT
    iot = action.get("iot")
    if stable and iot:
        cv2.rectangle(frame, (0,h-32), (w,h-9), (20,10,40), -1)
        cv2.putText(frame, f"IoT: {iot}",
                    (8,h-14), cv2.FONT_HERSHEY_SIMPLEX, 0.47, (200,150,255), 1)

    # Key guide
    cv2.rectangle(frame, (0,h-50), (w,h-32), (15,15,15), -1)
    cv2.putText(frame, "q:quit  r:rec  s:save  c:clear",
                (8,h-35), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120,120,120), 1)
    return frame


# =============================================================================
# MAIN LOOP
# =============================================================================
def run_webcam(checkpoint: Path, model_type: str, camera_id: int = 0,
               window: int = None, threshold: float = None,
               method: str = "majority", save_video: Path = None,
               save_log: Path = None):

    window    = window    or INFERENCE_CONFIG["smoothing_window"]
    threshold = threshold or INFERENCE_CONFIG["confidence_thresh"]

    print(f"\n[Webcam] Model: {model_type}  Checkpoint: {checkpoint}")
    model = _load_model(checkpoint, model_type)

    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False, max_num_hands=1,
        min_detection_confidence=CAMERA_CONFIG["min_detection_conf"],
        min_tracking_confidence=CAMERA_CONFIG["min_tracking_conf"],
        model_complexity=1,
    )

    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_CONFIG["width"])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_CONFIG["height"])
    cap.set(cv2.CAP_PROP_FPS,          CAMERA_CONFIG["fps"])
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_id}")

    fw, fh = int(cap.get(3)), int(cap.get(4))
    print(f"[Webcam] {fw}×{fh}  window={window}  threshold={threshold}")

    writer    = None
    recording = bool(save_video)
    if save_video:
        writer = cv2.VideoWriter(str(save_video),
                                 cv2.VideoWriter_fourcc(*"mp4v"), 20.0, (fw, fh))

    smoother     = TemporalSmoother(window, method)
    event_logger = EventLogger(save_log)
    lstm_buf     = collections.deque(maxlen=15) if model_type == "lstm" else None

    fps_buf   = collections.deque(maxlen=30)
    prev_time = time.time()
    gid, conf, gname = 0, 0.0, GESTURE_NAMES[0]

    print("[Webcam] Running — press q to quit\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res   = mp_hands.process(rgb)

        if res.multi_hand_landmarks:
            for hl in res.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hl, mp.solutions.hands.HAND_CONNECTIONS,
                    mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp.solutions.drawing_styles.get_default_hand_connections_style(),
                )
            lms = res.multi_hand_landmarks[0].landmark
            vec = _normalise(np.array([[lm.x,lm.y,lm.z] for lm in lms],
                                       dtype=np.float32).flatten())

            gid_raw, probs, conf_raw = _infer(model, model_type, vec, frame, lstm_buf)

            if gid_raw is not None:
                smoother.update(gid_raw, probs, conf_raw)
                gid, conf = smoother.get()
                gname     = GESTURE_NAMES[gid]

                if smoother.is_stable(threshold):
                    logged = event_logger.log(gid, conf)
                    if logged:
                        action = GESTURE_ACTIONS.get(gid, {})
                        ts     = datetime.now().strftime("%H:%M:%S")
                        print(f"[{ts}] [{gid:02d}] {gname:22s}  "
                              f"conf={conf:.3f}  → {action.get('message','')[:55]}")
                        if action.get("iot"):
                            print(f"  [IoT] → {action['iot']}")
        else:
            smoother.clear()
            event_logger.reset()

        now = time.time()
        fps_buf.append(1.0 / max(now - prev_time, 1e-6))
        prev_time = now

        frame = _draw_hud(frame, gid, gname, conf,
                          smoother.is_stable(threshold),
                          float(np.mean(fps_buf)), recording)

        if writer and recording:
            writer.write(frame)

        cv2.imshow("Gesture Recognition System", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("r"):
            recording = not recording
            print(f"[Webcam] Recording {'STARTED' if recording else 'STOPPED'}")
        elif key == ord("s"):
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            cv2.imwrite(f"frame_{ts}.jpg", frame)
            print(f"[Webcam] Frame saved: frame_{ts}.jpg")
        elif key == ord("c"):
            smoother.clear()
            event_logger.reset()
            print("[Webcam] Buffer cleared")

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    mp_hands.close()
    print(f"\n[Webcam] Done.  Events: {len(event_logger.events)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--checkpoint", required=True, type=Path)
    ap.add_argument("-m", "--model",      required=True,
                    choices=["cnn","mobilenet","lstm","svm"])
    ap.add_argument("--camera",     type=int,   default=0)
    ap.add_argument("--window",     type=int,   default=None)
    ap.add_argument("--threshold",  type=float, default=None)
    ap.add_argument("--method",     default="majority", choices=["majority","ewa"])
    ap.add_argument("--save-video", type=Path,  default=None)
    ap.add_argument("--save-log",   type=Path,  default=None)
    args = ap.parse_args()
    run_webcam(args.checkpoint, args.model, args.camera,
               args.window, args.threshold, args.method,
               args.save_video, args.save_log)


if __name__ == "__main__":
    main()
