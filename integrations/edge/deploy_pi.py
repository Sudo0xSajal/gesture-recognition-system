"""
integrations/edge/deploy_pi.py
================================
Deploy trained model to Raspberry Pi via SSH.

Ma'am's Document Reference
---------------------------
Hardware: Raspberry Pi 4 Model B
Software: TFLite runtime, MediaPipe, OpenCV
Purpose:  Standalone bedside gesture recognition without a PC

What this script does
---------------------
1. Connects to the Pi via SSH (password or private key)
2. Creates /home/pi/gesture_model/ directory on the Pi
3. Uploads model.tflite
4. Uploads config.json (gesture names + clinical actions)
5. Uploads pi_inference.py (self-contained inference script)
6. Installs required Python packages
7. Optionally starts inference immediately (nohup background process)

The Pi inference script (embedded below) is completely standalone —
it does not need the project directory structure.

Run
---
    python integrations/edge/deploy_pi.py \\
        --model-path integrations/edge/model.tflite \\
        --host 192.168.1.100 \\
        --password raspberry

    bash deploy_edge.sh -c log/<exp>/best_model.pt -m landmark \\
         --host 192.168.1.100 --password raspberry --run
"""

import sys
import io
import json
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))
from utils.config import GESTURE_NAMES, GESTURE_ACTIONS


# =============================================================================
# PI INFERENCE SCRIPT  (embedded — uploaded to Pi as pi_inference.py)
# This script runs COMPLETELY STANDALONE on the Raspberry Pi.
# It has NO dependency on the project directory.
# =============================================================================
PI_INFERENCE_SCRIPT = '''#!/usr/bin/env python3
"""
pi_inference.py
===============
Standalone gesture recognition inference for Raspberry Pi 4.
Uploaded and run by deploy_pi.py.

NO PROJECT IMPORTS NEEDED — this script is self-contained.

Pipeline (Ma\'am\'s document):
  1. OpenCV opens camera
  2. MediaPipe detects 21 hand landmarks
  3. TFLite model classifies gesture
  4. Decision engine maps gesture → clinical action
  5. Result displayed + optional GPIO alert

Hardware: Raspberry Pi 4 + Camera Module V2 (or USB webcam)

Run
---
    python3 pi_inference.py
    python3 pi_inference.py --model model.tflite --config config.json
    python3 pi_inference.py --quantized   # for INT8 quantised models

Press q to quit.
"""

import cv2
import json
import time
import numpy as np
import collections
from datetime import datetime

try:
    import mediapipe as mp
    MEDIAPIPE_OK = True
except ImportError:
    MEDIAPIPE_OK = False
    print("[WARN] MediaPipe not installed: pip3 install mediapipe")

try:
    import tflite_runtime.interpreter as tflite
    TFLITE_OK = True
except ImportError:
    try:
        import tensorflow.lite as tflite
        TFLITE_OK = True
    except ImportError:
        TFLITE_OK = False
        print("[WARN] TFLite not installed: pip3 install tflite-runtime")

try:
    import RPi.GPIO as GPIO
    GPIO_OK = True
except ImportError:
    GPIO_OK = False  # Not on Pi or no GPIO needed


# ── GPIO alert pin (change to match your circuit) ────────────────────────────
ALERT_PIN = 18
BUZZER_PIN = 23

if GPIO_OK:
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(ALERT_PIN,  GPIO.OUT, initial=GPIO.LOW)
    GPIO.setup(BUZZER_PIN, GPIO.OUT, initial=GPIO.LOW)


def trigger_alert(priority: str = "NORMAL"):
    """Activate GPIO alert (LED/buzzer) for nurse call."""
    if not GPIO_OK:
        return
    GPIO.output(ALERT_PIN, GPIO.HIGH)
    if priority == "HIGH":
        GPIO.output(BUZZER_PIN, GPIO.HIGH)
    time.sleep(0.5)
    GPIO.output(ALERT_PIN,  GPIO.LOW)
    GPIO.output(BUZZER_PIN, GPIO.LOW)


def normalise(vec):
    """Centre on wrist and scale to [-1, 1]."""
    pts   = vec.reshape(21, 3)
    pts   = pts - pts[0]
    scale = np.max(np.abs(pts)) + 1e-8
    return (pts / scale).flatten().astype(np.float32)


def load_config(config_path: str = "config.json") -> dict:
    """Load gesture names and actions from config.json."""
    with open(config_path) as f:
        return json.load(f)


class TemporalSmoother:
    """5-frame majority-vote temporal smoother."""
    def __init__(self, window: int = 5):
        self.buf = collections.deque(maxlen=window)
    def update(self, gid: int, conf: float):
        self.buf.append((gid, conf))
    def get(self):
        if not self.buf:
            return 0, 0.0
        counts = collections.Counter(g for g, _ in self.buf)
        gid    = counts.most_common(1)[0][0]
        confs  = [c for g, c in self.buf if g == gid]
        return gid, float(np.mean(confs))
    def is_stable(self, threshold: float = 0.75) -> bool:
        if len(self.buf) < self.buf.maxlen:
            return False
        ids = [g for g, _ in self.buf]
        return len(set(ids)) == 1 and np.mean([c for _, c in self.buf]) >= threshold
    def clear(self):
        self.buf.clear()


def run(model_path: str = "model.tflite",
        config_path: str = "config.json",
        camera_id: int = 0,
        is_quantized: bool = False):

    if not TFLITE_OK:
        print("[ERROR] TFLite not available. Run: pip3 install tflite-runtime")
        return
    if not MEDIAPIPE_OK:
        print("[ERROR] MediaPipe not available. Run: pip3 install mediapipe")
        return

    # ── Load config ───────────────────────────────────────────────────────────
    cfg             = load_config(config_path)
    gesture_names   = cfg["gesture_names"]
    gesture_actions = cfg.get("gesture_actions", {})

    # ── Load TFLite model ─────────────────────────────────────────────────────
    print(f"[Pi] Loading model: {model_path}")
    interp = tflite.Interpreter(model_path=model_path)
    interp.allocate_tensors()
    inp_det = interp.get_input_details()
    out_det = interp.get_output_details()

    # INT8 quantisation parameters (for dequantising output)
    if is_quantized:
        out_scale, out_zero_point = out_det[0]["quantization"]
        inp_scale, inp_zero_point = inp_det[0]["quantization"]
        print(f"[Pi] INT8 quantized — inp_scale={inp_scale:.6f}  out_scale={out_scale:.6f}")
    else:
        out_scale = out_zero_point = 0
        inp_scale = inp_zero_point = 0

    # ── MediaPipe ─────────────────────────────────────────────────────────────
    mp_hands = mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
        model_complexity=0,          # Fastest mode for Pi
    )

    # ── Camera ────────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS,          30)
    if not cap.isOpened():
        print(f"[Pi] ERROR: Cannot open camera {camera_id}")
        return

    smoother    = TemporalSmoother(window=5)
    fps_buf     = collections.deque(maxlen=15)
    prev_time   = time.time()
    last_logged = -1
    event_log   = []

    print("[Pi] Running. Press q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res   = mp_hands.process(rgb)
        h, w  = frame.shape[:2]

        gname, conf, gid = "...", 0.0, 0

        if res.multi_hand_landmarks:
            # Draw skeleton
            for hand_lms in res.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(
                    frame, hand_lms, mp.solutions.hands.HAND_CONNECTIONS)

            # Landmark vector
            lms = res.multi_hand_landmarks[0].landmark
            vec = np.array([[lm.x, lm.y, lm.z] for lm in lms],
                           dtype=np.float32).flatten()
            vec = normalise(vec)

            # TFLite inference
            if is_quantized:
                inp = np.clip(
                    np.round(vec / inp_scale + inp_zero_point),
                    -128, 127
                ).astype(np.int8).reshape(inp_det[0]["shape"])
            else:
                inp = vec.reshape(inp_det[0]["shape"])

            interp.set_tensor(inp_det[0]["index"], inp)
            interp.invoke()
            output = interp.get_tensor(out_det[0]["index"])

            if is_quantized:
                logits = (output.astype(np.float32) - out_zero_point) * out_scale
            else:
                logits = output.astype(np.float32)

            # Softmax
            logits = logits.flatten()
            probs  = np.exp(logits - logits.max())
            probs /= probs.sum()
            gid    = int(np.argmax(probs))
            conf   = float(probs[gid])
            gname  = gesture_names.get(str(gid), str(gid))

            smoother.update(gid, conf)
            gid, conf = smoother.get()
            gname     = gesture_names.get(str(gid), str(gid))

            # Stable gate
            if smoother.is_stable(0.75) and gid != last_logged:
                action   = gesture_actions.get(str(gid), {})
                msg      = action.get("message", "")
                is_alert = action.get("alert", False)
                priority = action.get("priority", "NORMAL")
                ts       = datetime.now().strftime("%H:%M:%S")
                print(f"[{ts}] [{gid:02d}] {gname}  conf={conf:.3f}  → {msg}")
                if is_alert:
                    trigger_alert(priority)
                last_logged = gid
                event_log.append({"ts": ts, "gesture": gname,
                                   "conf": round(conf, 4)})
        else:
            smoother.clear()
            last_logged = -1

        # FPS
        now = time.time()
        fps_buf.append(1.0 / max(now - prev_time, 1e-6))
        prev_time = now
        fps = float(np.mean(fps_buf))

        # Overlay
        col = (50,200,50) if conf >= 0.75 else (50,200,200) if conf >= 0.6 else (50,50,200)
        cv2.rectangle(frame, (0,0), (w,60), (15,15,15), -1)
        cv2.putText(frame, f"[{gid:02d}] {gname}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.80, col, 2)
        cv2.putText(frame, f"Conf: {conf*100:.1f}%  FPS: {fps:.1f}",
                    (10,55), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (180,180,180), 1)
        bar_w = int(w * conf)
        cv2.rectangle(frame, (0,h-8), (bar_w,h), col, -1)

        cv2.imshow("Gesture Recognition — Pi", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    mp_hands.close()
    if GPIO_OK:
        GPIO.cleanup()
    print(f"[Pi] Done. Total gestures logged: {len(event_log)}")


if __name__ == "__main__":
    import argparse as ap_mod
    parser = ap_mod.ArgumentParser()
    parser.add_argument("--model",     default="model.tflite")
    parser.add_argument("--config",    default="config.json")
    parser.add_argument("--camera",    type=int, default=0)
    parser.add_argument("--quantized", action="store_true")
    a = parser.parse_args()
    run(a.model, a.config, a.camera, a.quantized)
'''


# =============================================================================
# DEPLOYMENT
# =============================================================================
def deploy(
    model_path:   Path,
    host:         str,
    username:     str    = "pi",
    password:     str    = None,
    key_path:     str    = None,
    port:         int    = 22,
    remote_dir:   str    = "/home/pi/gesture_model",
    run_after:    bool   = False,
    is_quantized: bool   = False,
) -> None:
    """
    Deploy TFLite model + config + inference script to Raspberry Pi.

    Parameters
    ----------
    model_path   : Path  — local .tflite file
    host         : str   — Pi IP address or hostname
    username     : str   — SSH username (default 'pi')
    password     : str   — SSH password (None if using key auth)
    key_path     : str   — path to SSH private key (None if using password)
    port         : int   — SSH port (default 22)
    remote_dir   : str   — target directory on Pi
    run_after    : bool  — start inference after deployment
    is_quantized : bool  — add --quantized flag to pi_inference.py
    """
    try:
        import paramiko
    except ImportError:
        raise ImportError("Install paramiko: pip install paramiko")

    print(f"\n[Deploy] Connecting to {username}@{host}:{port} ...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    if key_path:
        ssh.connect(host, port=port, username=username,
                    key_filename=key_path, timeout=30)
    else:
        ssh.connect(host, port=port, username=username,
                    password=password, timeout=30)
    print("[Deploy] Connected ✓")

    def run_cmd(cmd: str) -> tuple:
        stdin, stdout, stderr = ssh.exec_command(cmd)
        out = stdout.read().decode().strip()
        err = stderr.read().decode().strip()
        return out, err

    # Step 1: Create remote directory
    print(f"\n[Deploy] Step 1: Creating {remote_dir}")
    run_cmd(f"mkdir -p {remote_dir}")

    sftp = ssh.open_sftp()

    # Step 2: Upload TFLite model
    remote_model = f"{remote_dir}/model.tflite"
    size_kb = model_path.stat().st_size / 1024
    print(f"[Deploy] Step 2: Uploading model ({size_kb:.1f} KB) → {remote_model}")
    sftp.put(str(model_path), remote_model)

    # Step 3: Upload config.json
    config = {
        "gesture_names": {str(k): v for k, v in GESTURE_NAMES.items()},
        "gesture_actions": {
            str(k): v for k, v in GESTURE_ACTIONS.items()
        },
        "num_classes": len(GESTURE_NAMES),
    }
    config_bytes = json.dumps(config, indent=2).encode()
    remote_config = f"{remote_dir}/config.json"
    print(f"[Deploy] Step 3: Uploading config.json → {remote_config}")
    sftp.putfo(io.BytesIO(config_bytes), remote_config)

    # Step 4: Upload pi_inference.py
    remote_script = f"{remote_dir}/pi_inference.py"
    print(f"[Deploy] Step 4: Uploading pi_inference.py → {remote_script}")
    sftp.putfo(io.BytesIO(PI_INFERENCE_SCRIPT.encode()), remote_script)
    run_cmd(f"chmod +x {remote_script}")

    sftp.close()

    # Step 5: Install dependencies
    print("\n[Deploy] Step 5: Installing dependencies on Pi ...")
    packages = "python3-opencv python3-numpy"
    out, _ = run_cmd(f"sudo apt-get install -y {packages} 2>&1 | tail -3")
    print(f"  apt: {out[:80]}")

    pip_pkgs = "mediapipe tflite-runtime"
    out, _ = run_cmd(f"pip3 install {pip_pkgs} 2>&1 | tail -3")
    print(f"  pip: {out[:80]}")

    print("\n[Deploy] Deployment complete ✓")
    print(f"  Model:  {remote_model}  ({size_kb:.1f} KB)")
    print(f"  Config: {remote_config}")
    print(f"  Script: {remote_script}")

    # Step 6: Run inference (optional)
    if run_after:
        qflag = "--quantized" if is_quantized else ""
        cmd   = (f"cd {remote_dir} && "
                 f"nohup python3 pi_inference.py {qflag} "
                 f"> inference.log 2>&1 &")
        print(f"\n[Deploy] Starting inference on Pi: {cmd}")
        run_cmd(cmd)
        print("[Deploy] Inference started in background (PID in inference.log)")

    ssh.close()

    print(f"\n[Deploy] To start inference manually:")
    print(f"  ssh {username}@{host}")
    print(f"  cd {remote_dir}")
    q = "--quantized " if is_quantized else ""
    print(f"  python3 pi_inference.py {q}--model model.tflite --config config.json")


def main():
    ap = argparse.ArgumentParser(description="Deploy gesture model to Raspberry Pi")
    ap.add_argument("--model-path", required=True, type=Path,
                    help="Path to local .tflite file")
    ap.add_argument("--host",       required=True,
                    help="Pi IP address or hostname")
    ap.add_argument("--username",   default="pi")
    ap.add_argument("--password",   default=None)
    ap.add_argument("--key",        default=None,
                    help="Path to SSH private key file")
    ap.add_argument("--port",       type=int, default=22)
    ap.add_argument("--remote-dir", default="/home/pi/gesture_model")
    ap.add_argument("--run",        action="store_true",
                    help="Start inference after deployment")
    ap.add_argument("--quantized",  action="store_true",
                    help="Tell pi_inference.py the model is INT8 quantized")
    args = ap.parse_args()

    deploy(
        model_path=args.model_path,
        host=args.host,
        username=args.username,
        password=args.password,
        key_path=args.key,
        port=args.port,
        remote_dir=args.remote_dir,
        run_after=args.run,
        is_quantized=args.quantized,
    )


if __name__ == "__main__":
    main()
