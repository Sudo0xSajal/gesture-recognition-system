"""
code/test.py
============
Quick single-sample inference — verify a checkpoint immediately after training.

Run
---
    # CNN or LSTM — from image file
    python code/test.py --model cnn  --checkpoint log/<exp>/best_model.pt --image hand.jpg
    python code/test.py --model lstm --checkpoint log/<exp>/best_model.pt --image hand.jpg

    # SVM — from landmark JSON (or image — MediaPipe runs automatically)
    python code/test.py --model svm  --checkpoint log/<exp>/svm_model.joblib --image hand.jpg
    python code/test.py --model svm  --checkpoint log/<exp>/svm_model.joblib --landmarks lm.json
"""

import json
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
import cv2
import mediapipe as mp

from utils.config import GESTURE_NAMES, GESTURE_ACTIONS, CAMERA_CONFIG


def _normalise(vec: np.ndarray) -> np.ndarray:
    pts   = vec.reshape(21, 3)
    pts   = pts - pts[0]
    scale = np.max(np.abs(pts)) + 1e-8
    return (pts / scale).flatten().astype(np.float32)


def _extract_landmarks(img_path: Path) -> np.ndarray:
    img = cv2.imread(str(img_path))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    with mp.solutions.hands.Hands(
        static_image_mode=True, max_num_hands=1,
        min_detection_confidence=CAMERA_CONFIG["min_detection_conf"],
    ) as hands:
        result = hands.process(rgb)
    if not result.multi_hand_landmarks:
        raise RuntimeError("No hand detected in image.")
    lms = result.multi_hand_landmarks[0].landmark
    vec = np.array([[lm.x, lm.y, lm.z] for lm in lms], dtype=np.float32).flatten()
    return _normalise(vec)


def _load_landmark_json(path: Path) -> np.ndarray:
    with open(path) as f:
        data = json.load(f)
    vec = np.array([[lm["x"], lm["y"], lm["z"]] for lm in data["landmarks"]],
                   dtype=np.float32).flatten()
    return _normalise(vec)


def _image_tensor(path: Path) -> torch.Tensor:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    img = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    t   = A.Compose([A.Resize(224,224),
                     A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
                     ToTensorV2()])(image=img)["image"]
    return t.unsqueeze(0)


def predict_cnn(checkpoint: Path, model_type: str, img_path: Path) -> dict:
    from models.cnn_model import GestureCNN, MobileNetV2Transfer
    cls_map = {"cnn": GestureCNN, "mobilenet": MobileNetV2Transfer}
    model = cls_map[model_type]()
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state["model_state"])
    model.eval()
    img_t = _image_tensor(img_path)
    with torch.no_grad():
        probs = F.softmax(model(img_t), dim=1).squeeze(0).numpy()
    return probs


def predict_lstm(checkpoint: Path, img_path: Path) -> dict:
    from models.lstm_model import LSTMModel
    vec   = _extract_landmarks(img_path)
    model = LSTMModel()
    state = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state["model_state"])
    model.eval()
    seq = torch.from_numpy(vec).unsqueeze(0).repeat(15, 1).unsqueeze(0)  # (1,15,63)
    with torch.no_grad():
        probs = F.softmax(model(seq), dim=1).squeeze(0).numpy()
    return probs


def predict_svm(checkpoint: Path, vec: np.ndarray) -> np.ndarray:
    from models.svm_classifier import SVMClassifier
    clf   = SVMClassifier.load(checkpoint)
    probs = clf.predict_proba(vec.reshape(1, -1))[0]
    return probs


def _print_result(probs: np.ndarray, model_type: str):
    gid  = int(np.argmax(probs))
    conf = float(probs[gid])
    action = GESTURE_ACTIONS.get(gid, {})
    top5   = np.argsort(probs)[::-1][:5]
    print(f"\n{'='*52}")
    print(f"  Model:      {model_type}")
    print(f"  Gesture:    [{gid:02d}] {GESTURE_NAMES[gid]}")
    print(f"  Confidence: {conf:.4f}  ({conf*100:.1f}%)")
    print(f"  Action:     {action.get('message','')}")
    print(f"  Alert:      {action.get('alert',False)}  |  IoT: {action.get('iot',None)}")
    print(f"\n  Top-5:")
    for i in top5:
        print(f"    [{i:02d}] {GESTURE_NAMES[i]:24s}  {probs[i]:.4f}")
    print("="*52)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model",      required=True,
                    choices=["cnn","mobilenet","lstm","svm"])
    ap.add_argument("-c", "--checkpoint", required=True, type=Path)
    ap.add_argument("--image",     type=Path, default=None)
    ap.add_argument("--landmarks", type=Path, default=None)
    args = ap.parse_args()

    if args.image is None and args.landmarks is None:
        ap.error("Provide --image or --landmarks")

    if args.model in ("cnn", "mobilenet"):
        if args.image is None:
            ap.error("CNN model needs --image")
        probs = predict_cnn(args.checkpoint, args.model, args.image)

    elif args.model == "lstm":
        if args.image is None:
            ap.error("LSTM model needs --image (MediaPipe extracts landmarks)")
        probs = predict_lstm(args.checkpoint, args.image)

    elif args.model == "svm":
        if args.landmarks:
            vec = _load_landmark_json(args.landmarks)
        else:
            vec = _extract_landmarks(args.image)
        probs = predict_svm(args.checkpoint, vec)

    _print_result(probs, args.model)


if __name__ == "__main__":
    main()
