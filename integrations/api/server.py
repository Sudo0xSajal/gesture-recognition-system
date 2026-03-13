"""
integrations/api/server.py
===========================
STEP 5 — Decision System: REST API

Flask REST server supporting all three model types:
  CNN  — predicts from base64 image
  LSTM — predicts from base64 image (MediaPipe extracts sequence internally)
  SVM  — predicts from 63-dim landmark array (fastest, no GPU needed)

Endpoints
---------
  GET  /health                  → server status
  GET  /gestures                → all 25 gesture names + clinical actions
  GET  /gesture_action/<id>     → clinical action for one gesture ID
  POST /predict                 → base64 image → prediction (CNN / LSTM)
  POST /predict_landmarks       → 63-dim array → prediction (SVM / LSTM)
  POST /batch_predict           → list of base64 images → list of predictions

Run
---
    # CNN
    bash serve.sh -c log/<exp>/best_model.pt -m cnn

    # LSTM
    bash serve.sh -c log/<exp>/best_model.pt -m lstm

    # SVM
    bash serve.sh -c log/<exp>/svm_model.joblib -m svm
"""

import sys
import json
import base64
import argparse
import threading
import numpy as np
from pathlib import Path

import cv2
import mediapipe as mp
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "code"))

from utils.config import (
    GESTURE_NAMES, GESTURE_ACTIONS, CAMERA_CONFIG,
    INFERENCE_CONFIG, NUM_CLASSES,
)


# =============================================================================
# INFERENCE ENGINE  (supports CNN, LSTM, SVM)
# =============================================================================
class InferenceEngine:
    """Thread-safe inference engine for CNN, LSTM, and SVM models."""

    def __init__(self, checkpoint: Path, model_type: str, device: str = "cpu"):
        self.model_type = model_type
        self.device     = device
        self._lock      = threading.Lock()

        if model_type in ("cnn", "mobilenet"):
            self._load_pytorch(checkpoint, model_type, device)
        elif model_type == "lstm":
            self._load_pytorch(checkpoint, "lstm", device)
        elif model_type == "svm":
            self._load_svm(checkpoint)
        else:
            raise ValueError(f"Unknown model_type '{model_type}'")

        # MediaPipe for landmark extraction
        self._mp_hands = mp.solutions.hands.Hands(
            static_image_mode=True, max_num_hands=1,
            min_detection_confidence=CAMERA_CONFIG["min_detection_conf"],
            min_tracking_confidence=CAMERA_CONFIG["min_tracking_conf"],
            model_complexity=1,
        )
        print(f"[Server] Model '{model_type}' loaded from {checkpoint}")

    def _load_pytorch(self, checkpoint: Path, model_type: str, device: str):
        from models.cnn_model import GestureCNN, MobileNetV2Transfer
        from models.lstm_model import LSTMModel
        cls_map = {"cnn": GestureCNN, "mobilenet": MobileNetV2Transfer, "lstm": LSTMModel}
        self.model = cls_map[model_type]()
        state = torch.load(checkpoint, map_location=device)
        self.model.load_state_dict(state["model_state"])
        self.model.eval()
        self.model.to(device)

    def _load_svm(self, checkpoint: Path):
        from models.svm_classifier import SVMClassifier
        self.clf = SVMClassifier.load(checkpoint)

    def _normalise(self, vec: np.ndarray) -> np.ndarray:
        pts   = vec.reshape(21, 3)
        pts   = pts - pts[0]
        scale = np.max(np.abs(pts)) + 1e-8
        return (pts / scale).flatten().astype(np.float32)

    def _mediapipe(self, bgr: np.ndarray) -> np.ndarray:
        """BGR image → 63-dim normalised landmark vector, or None."""
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        res = self._mp_hands.process(rgb)
        if not res.multi_hand_landmarks:
            return None
        lms = res.multi_hand_landmarks[0].landmark
        vec = np.array([[lm.x, lm.y, lm.z] for lm in lms],
                       dtype=np.float32).flatten()
        return self._normalise(vec)

    def _image_preprocess(self, bgr: np.ndarray) -> torch.Tensor:
        import albumentations as A
        from albumentations.pytorch import ToTensorV2
        transform = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
            ToTensorV2(),
        ])
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return transform(image=rgb)["image"].unsqueeze(0).to(self.device)

    def predict_from_image(self, bgr: np.ndarray) -> dict:
        """Predict from BGR image (all model types)."""
        with self._lock:
            if self.model_type in ("cnn", "mobilenet"):
                img_t = self._image_preprocess(bgr)
                with torch.no_grad():
                    probs = F.softmax(self.model(img_t), dim=1).squeeze(0).cpu().numpy()

            elif self.model_type == "lstm":
                vec = self._mediapipe(bgr)
                if vec is None:
                    return {"error": "No hand detected", "gesture_id": -1, "confidence": 0.0}
                seq = torch.from_numpy(vec).unsqueeze(0).repeat(15, 1).unsqueeze(0)
                with torch.no_grad():
                    probs = F.softmax(self.model(seq), dim=1).squeeze(0).cpu().numpy()

            elif self.model_type == "svm":
                vec = self._mediapipe(bgr)
                if vec is None:
                    return {"error": "No hand detected", "gesture_id": -1, "confidence": 0.0}
                probs = self.clf.predict_proba(vec.reshape(1, -1))[0]

        return self._build_result(probs)

    def predict_from_landmarks(self, landmarks: list) -> dict:
        """Predict from raw 63-dim landmark list (SVM and LSTM)."""
        if len(landmarks) != 63:
            return {"error": f"Expected 63 values, got {len(landmarks)}",
                    "gesture_id": -1, "confidence": 0.0}
        vec = self._normalise(np.array(landmarks, dtype=np.float32))

        with self._lock:
            if self.model_type == "svm":
                probs = self.clf.predict_proba(vec.reshape(1, -1))[0]

            elif self.model_type == "lstm":
                seq = torch.from_numpy(vec).unsqueeze(0).repeat(15, 1).unsqueeze(0)
                with torch.no_grad():
                    probs = F.softmax(self.model(seq), dim=1).squeeze(0).cpu().numpy()

            else:
                return {"error": "predict_landmarks not supported for CNN models. "
                                 "Use /predict with a base64 image instead.",
                        "gesture_id": -1, "confidence": 0.0}

        return self._build_result(probs)

    def _build_result(self, probs: np.ndarray) -> dict:
        gid    = int(np.argmax(probs))
        conf   = float(probs[gid])
        action = GESTURE_ACTIONS.get(gid, {})
        top5   = np.argsort(probs)[::-1][:5]
        return {
            "gesture_id":     gid,
            "gesture_name":   GESTURE_NAMES[gid],
            "confidence":     conf,
            "confidence_pct": f"{conf*100:.1f}%",
            "action":         action.get("message",  ""),
            "alert":          action.get("alert",    False),
            "iot_command":    action.get("iot",      None),
            "priority":       action.get("priority", "NORMAL"),
            "top5": [
                {"gesture_id": int(i), "gesture_name": GESTURE_NAMES[int(i)],
                 "confidence": round(float(probs[i]), 4)}
                for i in top5
            ],
        }


# =============================================================================
# FLASK APP
# =============================================================================
app    = Flask(__name__)
engine: InferenceEngine = None


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_type": engine.model_type,
                    "num_classes": NUM_CLASSES})


@app.route("/gestures", methods=["GET"])
def gestures():
    return jsonify({
        str(gid): {
            "name":   name,
            "action": GESTURE_ACTIONS.get(gid, {}).get("message", ""),
            "alert":  GESTURE_ACTIONS.get(gid, {}).get("alert", False),
            "iot":    GESTURE_ACTIONS.get(gid, {}).get("iot", None),
        }
        for gid, name in GESTURE_NAMES.items()
    })


@app.route("/gesture_action/<int:gid>", methods=["GET"])
def gesture_action(gid: int):
    if gid not in GESTURE_NAMES:
        return jsonify({"error": f"Unknown gesture_id {gid}"}), 404
    return jsonify({"gesture_id": gid, "gesture_name": GESTURE_NAMES[gid],
                    **GESTURE_ACTIONS.get(gid, {})})


@app.route("/predict", methods=["POST"])
def predict():
    """Predict from base64-encoded image. Works with CNN and LSTM models."""
    data = request.get_json(force=True)
    if "image" not in data:
        return jsonify({"error": "Missing 'image' field"}), 400
    try:
        buf = np.frombuffer(base64.b64decode(data["image"]), np.uint8)
        bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if bgr is None:
            return jsonify({"error": "Could not decode image"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    return jsonify(engine.predict_from_image(bgr))


@app.route("/predict_landmarks", methods=["POST"])
def predict_landmarks():
    """Predict from 63-dim landmark array. Works with SVM and LSTM models."""
    data = request.get_json(force=True)
    if "landmarks" not in data:
        return jsonify({"error": "Missing 'landmarks' field"}), 400
    return jsonify(engine.predict_from_landmarks(data["landmarks"]))


@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    """Predict for multiple base64 images at once."""
    data = request.get_json(force=True)
    if "images" not in data:
        return jsonify({"error": "Missing 'images' field"}), 400
    results = []
    for idx, img_b64 in enumerate(data["images"]):
        try:
            buf = np.frombuffer(base64.b64decode(img_b64), np.uint8)
            bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            res = engine.predict_from_image(bgr)
        except Exception as e:
            res = {"error": str(e), "gesture_id": -1, "confidence": 0.0}
        results.append({"index": idx, **res})
    return jsonify({"results": results, "count": len(results)})


def main():
    global engine
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--checkpoint", required=True, type=Path)
    ap.add_argument("-m", "--model",      required=True,
                    choices=["cnn","mobilenet","lstm","svm"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--port",   type=int, default=5000)
    ap.add_argument("--host",   default="0.0.0.0")
    ap.add_argument("--debug",  action="store_true")
    args = ap.parse_args()

    print(f"\n[Server] Gesture Recognition API")
    print(f"  Model: {args.model}  |  Checkpoint: {args.checkpoint}")
    print(f"  URL:   http://{args.host}:{args.port}\n")

    engine = InferenceEngine(args.checkpoint, args.model, args.device)
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
