"""
code/models/svm_classifier.py
==============================
STEP 3 — Gesture Recognition: SVM (Support Vector Machine)

SVM with RBF kernel trained on 63-dim normalised MediaPipe landmark vectors.
This is one of the three algorithms specified in Ma'am's document.

Why SVM for gesture recognition?
---------------------------------
  - Works well on small datasets (< 1000 samples per class)
  - 63-dim landmark vectors are already a good feature space
  - RBF kernel handles non-linear decision boundaries between similar gestures
    (e.g. "Hand to Mouth" vs "Eating Gesture" are close in landmark space)
  - Fastest inference: ~1ms per sample on any CPU
  - No GPU needed
  - Interpretable: support vectors show which hand poses are hardest to classify

Architecture
------------
  Input: 63-dim normalised MediaPipe landmark vector
  Kernel: RBF (Radial Basis Function)  k(x,y) = exp(-gamma * ||x-y||^2)
  C: regularisation strength (default 10.0)
  Output: gesture class ID (0-24) + probability vector via Platt scaling

Training: scikit-learn SVC with svm_probability=True
Inference: predict() returns class ID, predict_proba() returns 25 probabilities

Saved as: log/<exp>/svm_model.joblib

Run
---
    python code/train_svm.py --task hgrd --exp-name exp1
    python code/train_svm.py --task hgrd --exp-name exp1 --kernel linear
    python code/train_svm.py --task hgrd --exp-name exp1 --C 50 --gamma 0.001
"""

import pickle
import numpy as np
from pathlib import Path

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report
import joblib

from utils.config import (
    GESTURE_NAMES, NUM_CLASSES, TRAINING_CONFIG, get_action,
)


class SVMClassifier:
    """
    SVM gesture classifier wrapped in a scikit-learn Pipeline.

    The Pipeline applies StandardScaler first (zero mean, unit variance)
    then SVC. Scaling is important for RBF SVM — without it, features with
    larger ranges dominate the kernel computation.

    Parameters
    ----------
    kernel      : str   — 'rbf' | 'linear' | 'poly'
    C           : float — regularisation (higher = less regularisation)
    gamma       : str/float — RBF bandwidth ('scale' | 'auto' | float)
    probability : bool  — enable predict_proba via Platt scaling
    """

    def __init__(
        self,
        kernel:      str         = None,
        C:           float       = None,
        gamma = None,
        probability: bool        = None,
    ):
        kernel      = kernel      or TRAINING_CONFIG["svm_kernel"]
        C           = C           or TRAINING_CONFIG["svm_C"]
        gamma       = gamma       or TRAINING_CONFIG["svm_gamma"]
        probability = probability if probability is not None else TRAINING_CONFIG["svm_probability"]

        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("svm",    SVC(
                kernel=kernel,
                C=C,
                gamma=gamma,
                probability=probability,
                class_weight="balanced",   # handles class imbalance like FocalLoss
                decision_function_shape="ovr",
                random_state=42,
            )),
        ])
        self.is_trained = False
        self.classes_   = list(range(NUM_CLASSES))

    # ── Training ──────────────────────────────────────────────────────────────
    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVMClassifier":
        """
        Train the SVM on landmark feature vectors.

        Parameters
        ----------
        X : (N, 63) float32 — normalised landmark vectors
        y : (N,)    int     — gesture class labels
        """
        print(f"[SVM] Training on {len(X)} samples  "
              f"kernel={self.pipeline['svm'].kernel}  "
              f"C={self.pipeline['svm'].C}")
        self.pipeline.fit(X, y)
        self.is_trained = True
        print("[SVM] Training complete")
        return self

    # ── Inference ─────────────────────────────────────────────────────────────
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class IDs. X: (N, 63). Returns (N,) int array."""
        return self.pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        Returns (N, 25) float array via Platt scaling (probability=True required).
        """
        if not self.pipeline["svm"].probability:
            raise RuntimeError("Set svm_probability=True in TRAINING_CONFIG")
        return self.pipeline.predict_proba(X)

    def predict_single(self, vec: np.ndarray) -> dict:
        """
        Predict one 63-dim landmark vector and return full result dict.

        Parameters
        ----------
        vec : (63,) float32 — normalised landmark vector

        Returns
        -------
        dict:
          gesture_id, gesture_name, confidence, action, alert, iot_command, top5
        """
        x     = vec.reshape(1, -1)
        probs = self.predict_proba(x)[0]         # (25,)
        gid   = int(np.argmax(probs))
        conf  = float(probs[gid])
        action = get_action(gid)

        top5_idx  = np.argsort(probs)[::-1][:5]
        return {
            "gesture_id":   gid,
            "gesture_name": GESTURE_NAMES[gid],
            "confidence":   conf,
            "confidence_pct": f"{conf*100:.1f}%",
            "action":       action.get("message", ""),
            "alert":        action.get("alert", False),
            "iot_command":  action.get("iot", None),
            "priority":     action.get("priority", "NORMAL"),
            "top5": [
                {
                    "gesture_id":   int(i),
                    "gesture_name": GESTURE_NAMES[int(i)],
                    "confidence":   round(float(probs[i]), 4),
                }
                for i in top5_idx
            ],
        }

    # ── Evaluation ─────────────────────────────────────────────────────────────
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """
        Evaluate on a labeled dataset and print per-class accuracy.

        Returns
        -------
        dict: overall_accuracy, per_class_accuracy, classification_report
        """
        preds   = self.predict(X)
        overall = accuracy_score(y, preds)
        names   = [GESTURE_NAMES[i] for i in range(NUM_CLASSES)]
        report  = classification_report(y, preds, target_names=names,
                                        zero_division=0, output_dict=True)
        report_str = classification_report(y, preds, target_names=names,
                                           zero_division=0)

        # Per-class accuracy
        per_class = {}
        for gid in range(NUM_CLASSES):
            mask    = y == gid
            if mask.sum() == 0:
                per_class[gid] = None
                continue
            per_class[gid] = round(float((preds[mask] == gid).mean()), 4)

        print(f"\n[SVM] Overall accuracy: {overall:.4f}  ({overall*100:.2f}%)")
        print(report_str)

        return {
            "overall_accuracy":   round(overall, 6),
            "macro_f1":           round(report["macro avg"]["f1-score"], 4),
            "weighted_f1":        round(report["weighted avg"]["f1-score"], 4),
            "per_class_accuracy": {GESTURE_NAMES[i]: v
                                   for i, v in per_class.items()},
        }

    # ── Save / Load ────────────────────────────────────────────────────────────
    def save(self, path: Path) -> None:
        """Save trained pipeline to disk."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pipeline, path)
        size_kb = Path(path).stat().st_size / 1024
        print(f"[SVM] Saved → {path}  ({size_kb:.1f} KB)")

    @classmethod
    def load(cls, path: Path) -> "SVMClassifier":
        """Load a saved SVM pipeline from disk."""
        inst = cls.__new__(cls)
        inst.pipeline   = joblib.load(path)
        inst.is_trained = True
        inst.classes_   = list(range(NUM_CLASSES))
        print(f"[SVM] Loaded from {path}")
        return inst
