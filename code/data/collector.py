"""
code/data/collector.py
======================
STEP 1 + STEP 2 — Sensing & Hand Detection (Ma'am's Document)

Interactive webcam tool for recording labeled gesture samples.
Uses MediaPipe to detect and validate hand landmarks before saving each frame,
ensuring only high-quality samples enter the dataset.

What it saves per frame
-----------------------
  frame_XXXXXX.jpg          — JPEG image (640×480)
  landmarks_XXXXXX.json     — 21 MediaPipe hand landmarks (x, y, z each)

Directory layout
----------------
  <dataset_root>/<participant>/<session>/<gesture_id>/
      frame_000001.jpg
      landmarks_000001.json
      ...

Run
---
    python code/data/collector.py --participant p001
    python code/data/collector.py --participant p001 --session morning

Keyboard controls
-----------------
    0–9       select gesture class 0–9
    a–o       select gesture class 10–24
    r         start / stop recording
    q         quit
"""

import cv2
import json
import time
import threading
import queue
import argparse
import numpy as np
import mediapipe as mp
from pathlib import Path
from datetime import datetime

from utils.config import (
    GESTURE_NAMES, GESTURE_ACTIONS, CAMERA_CONFIG,
    NUM_CLASSES, RAW_DATASET_PATH,
)


# ── Keyboard map: char → gesture ID ──────────────────────────────────────────
_KEY_MAP: dict[str, int] = {str(i): i for i in range(10)}
_KEY_MAP.update({chr(ord("a") + i): 10 + i for i in range(15)})


# =============================================================================
# QUALITY SCORER
# =============================================================================
def _quality_score(frame: np.ndarray, hand_result) -> float:
    """
    Score a frame 0–1 based on:
      - Brightness: how close to 127 the mean pixel is (0 = black/white, 1 = ideal)
      - Contrast:   stddev / 50 capped at 1 (higher = sharper)
      - Visibility: 1.0 if all landmarks are inside frame, 0.6 if any are outside

    Only frames scoring above CAMERA_CONFIG['quality_threshold'] are saved.
    """
    gray      = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = 1.0 - abs(float(gray.mean()) - 127.0) / 127.0
    contrast   = min(float(gray.std()) / 50.0, 1.0)
    visibility = 1.0

    if hand_result and hand_result.multi_hand_landmarks:
        for lm in hand_result.multi_hand_landmarks[0].landmark:
            if not (0.0 <= lm.x <= 1.0 and 0.0 <= lm.y <= 1.0):
                visibility = 0.6
                break

    return (brightness + contrast + visibility) / 3.0


# =============================================================================
# ASYNC FILE SAVER
# =============================================================================
class _AsyncSaver:
    """
    Writes JPEG + JSON pairs to disk on a background thread so the main
    capture loop never blocks on I/O.
    """

    def __init__(self, max_queue: int = 500):
        self._q    = queue.Queue(maxsize=max_queue)
        self._stop = threading.Event()
        self._t    = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _run(self):
        while not self._stop.is_set() or not self._q.empty():
            try:
                fn = self._q.get(timeout=0.1)
                fn()
                self._q.task_done()
            except queue.Empty:
                continue

    def enqueue(self, frame: np.ndarray, lm_data: dict,
                save_dir: Path, idx: int):
        img_path = save_dir / f"frame_{idx:06d}.jpg"
        lm_path  = save_dir / f"landmarks_{idx:06d}.json"
        frame_cp = frame.copy()

        def _write():
            cv2.imwrite(str(img_path), frame_cp,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            with open(lm_path, "w") as f:
                json.dump(lm_data, f)

        self._q.put(_write)

    def finish(self):
        self._stop.set()
        self._q.join()


# =============================================================================
# DATA COLLECTOR
# =============================================================================
class DataCollector:
    """
    Full gesture data collection application.

    Implements Ma'am's Step 1 (Sensing) and Step 2 (Hand Detection):
    - Opens RGB camera (Step 1)
    - Runs MediaPipe Hands to detect 21 landmark keypoints per frame (Step 2)
    - Applies quality scoring to reject blurry / badly lit frames
    - Saves passing frames + landmark JSON asynchronously

    Parameters
    ----------
    participant : participant ID string, e.g. 'p001'
    session     : optional session name (default = timestamp)
    """

    def __init__(self, participant: str, session: str = None):
        self.root        = RAW_DATASET_PATH
        self.participant = participant
        self.session     = session or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.cur_gid     = 0
        self.recording   = False
        self.counters: dict[int, int] = {}
        self._saver      = _AsyncSaver()

        # MediaPipe Hands  (Step 2 — Hand & Body Detection)
        self._mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=CAMERA_CONFIG["min_detection_conf"],
            min_tracking_confidence=CAMERA_CONFIG["min_tracking_conf"],
            model_complexity=1,
        )

    # ── Camera init ──────────────────────────────────────────────────────────
    def _open_camera(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(CAMERA_CONFIG["device_id"])
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  CAMERA_CONFIG["width"])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_CONFIG["height"])
        cap.set(cv2.CAP_PROP_FPS,          CAMERA_CONFIG["fps"])
        cap.set(cv2.CAP_PROP_AUTOFOCUS,    1 if CAMERA_CONFIG["autofocus"] else 0)
        if not cap.isOpened():
            raise RuntimeError("Cannot open camera. Check device_id in CAMERA_CONFIG.")
        return cap

    # ── Build landmark JSON dict ──────────────────────────────────────────────
    @staticmethod
    def _build_landmark_dict(hand_landmarks, gesture_id: int,
                              participant: str, session: str,
                              quality: float) -> dict:
        """
        Serialise MediaPipe hand landmarks to JSON.

        Returns a dict matching exactly what data/preprocess.py expects:
          {"landmarks": [{id, x, y, z, visibility}, ...], "gesture_id": int, ...}
        """
        return {
            "landmarks": [
                {
                    "id":         i,
                    "x":          float(lm.x),
                    "y":          float(lm.y),
                    "z":          float(lm.z),
                    "visibility": float(getattr(lm, "visibility", 1.0)),
                }
                for i, lm in enumerate(hand_landmarks.landmark)
            ],
            "gesture_id":  gesture_id,
            "gesture_name": GESTURE_NAMES[gesture_id],
            "participant": participant,
            "session":     session,
            "timestamp":   time.time(),
            "quality":     round(quality, 4),
        }

    # ── HUD overlay ──────────────────────────────────────────────────────────
    def _draw_hud(self, frame: np.ndarray, score: float) -> np.ndarray:
        h, w   = frame.shape[:2]
        gname  = GESTURE_NAMES.get(self.cur_gid, "?")
        count  = self.counters.get(self.cur_gid, 0)
        action = GESTURE_ACTIONS.get(self.cur_gid, {}).get("message", "")
        rec    = "● REC" if self.recording else "  READY"
        col    = (0, 0, 220) if self.recording else (0, 200, 0)

        # Top bar
        cv2.rectangle(frame, (0, 0), (w, 80), (18, 18, 18), -1)
        cv2.putText(frame, f"{rec}  [{self.cur_gid:02d}] {gname}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, col, 2)
        cv2.putText(frame, f"Frames: {count}   Quality: {score:.2f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (180, 180, 180), 1)

        # Clinical meaning
        cv2.rectangle(frame, (0, 80), (w, 110), (10, 40, 10), -1)
        cv2.putText(frame, f"Meaning: {action[:70]}",
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 255, 180), 1)

        # Quality bar
        bar = int(w * score)
        cv2.rectangle(frame, (0, h - 8), (bar, h),
                      (0, 200, 0) if score >= 0.6 else (0, 0, 200), -1)

        # Key guide
        cv2.rectangle(frame, (0, h - 30), (w, h - 8), (18, 18, 18), -1)
        cv2.putText(frame,
                    "0-9 / a-o: gesture   r: rec/stop   q: quit",
                    (10, h - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
                    (140, 140, 140), 1)
        return frame

    # ── Main loop ─────────────────────────────────────────────────────────────
    def run(self):
        cap = self._open_camera()
        print("\n[DataCollector] Started.")
        print(f"  Dataset: {self.root.name}  Participant: {self.participant}  "
              f"Session: {self.session}")
        print("  Keys: 0-9 / a-o = gesture   r = record/stop   q = quit\n")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[DataCollector] Camera read failed.")
                break
            frame = cv2.flip(frame, 1)

            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self._mp_hands.process(rgb)
            score   = _quality_score(frame, results)

            # Draw MediaPipe skeleton
            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    mp.solutions.drawing_utils.draw_landmarks(
                        frame, hand_lms,
                        mp.solutions.hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                        mp.solutions.drawing_styles.get_default_hand_connections_style(),
                    )

            # Save if recording, hand detected, and quality is good
            if (self.recording
                    and results.multi_hand_landmarks
                    and score >= CAMERA_CONFIG["quality_threshold"]):
                hand0 = results.multi_hand_landmarks[0]
                gid   = self.cur_gid
                idx   = self.counters.get(gid, 0) + 1
                self.counters[gid] = idx

                save_dir = (self.root / "raw" / self.participant
                            / self.session / str(gid))
                save_dir.mkdir(parents=True, exist_ok=True)

                lm_data = self._build_landmark_dict(
                    hand0, gid, self.participant, self.session, score)
                self._saver.enqueue(frame, lm_data, save_dir, idx)

            frame = self._draw_hud(frame, score)
            cv2.imshow("Gesture Data Collector", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                self.recording = not self.recording
                status = "STARTED" if self.recording else "STOPPED"
                print(f"  Recording {status} — [{self.cur_gid}] "
                      f"{GESTURE_NAMES[self.cur_gid]}")
            else:
                ch = chr(key) if key < 128 else ""
                if ch in _KEY_MAP:
                    new_gid = _KEY_MAP[ch]
                    if new_gid != self.cur_gid:
                        self.cur_gid   = new_gid
                        self.recording = False
                        print(f"  Selected [{self.cur_gid}] "
                              f"{GESTURE_NAMES[self.cur_gid]}")

        # ── Cleanup ───────────────────────────────────────────────────────────
        self._saver.finish()
        cap.release()
        cv2.destroyAllWindows()
        self._mp_hands.close()

        print("\n[DataCollector] Session summary:")
        total = 0
        for gid in range(NUM_CLASSES):
            cnt = self.counters.get(gid, 0)
            if cnt > 0:
                print(f"  [{gid:02d}] {GESTURE_NAMES[gid]:20s}: {cnt:4d} frames")
                total += cnt
        print(f"\n  Total saved: {total} frames")
        print(f"  Saved to: {self.root / 'raw' / self.participant / self.session}\n")


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Gesture data collector")
    ap.add_argument("-p", "--participant", required=True,
                    help="Participant ID, e.g. p001")
    ap.add_argument("-s", "--session",     default=None,
                    help="Session name (default: timestamp)")
    args = ap.parse_args()
    DataCollector(args.participant, args.session).run()
