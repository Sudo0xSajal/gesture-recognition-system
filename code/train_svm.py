"""
code/train_svm.py
=================
COMPLETE SVM Training — Step 3 of Ma'am's Document
Algorithm: Support Vector Machine (SVM)

This ONE file contains everything needed to train the SVM model:
  - Feature loading (63-dim normalised landmark vectors from .npz cache)
  - StandardScaler for feature normalisation
  - SVC with RBF kernel training
  - Validation evaluation
  - Final test evaluation with per-class accuracy
  - Model saving to .joblib file
  - Per-class accuracy bar chart
  - Summary JSON

Why SVM for gesture recognition?
----------------------------------
SVM works on the 63-dim MediaPipe landmark vectors directly.
It does NOT need images. It does NOT need a GPU.
It trains in seconds to minutes instead of hours.

It is the best choice when:
  - You have limited training data (< 500 samples per class)
  - You are deploying on Raspberry Pi (no GPU, limited CPU)
  - You want instant predictions (~1ms per sample)
  - You want a simple baseline to compare against CNN and LSTM

How SVM works:
  - Finds the maximum-margin boundary between gesture classes
  - RBF kernel: k(x,y) = exp(-gamma * ||x-y||^2)
    measures similarity in the 63-dim landmark space
  - class_weight='balanced' handles the same class imbalance
    problem that FocalLoss handles for CNN and LSTM

Run
---
    python code/train_svm.py --exp-name exp1
    python code/train_svm.py --exp-name exp1 --kernel linear
    python code/train_svm.py --exp-name exp1 --C 50 --gamma 0.001

Or via shell:
    bash train_svm.sh -e exp1 -k rbf -C 10

Output saved to:
    log/exp1_svm_gesture/
        svm_model.joblib        <- load this for inference
        summary.json            <- accuracy numbers
        eval_results.json       <- per-class accuracy
        per_class_accuracy.png  <- bar chart
"""

import sys
import json
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.config import (
    get_split_file, get_log_dir,
    PREPROCESSED_DATASET_PATH,
    GESTURE_NAMES, NUM_CLASSES,
)


# =============================================================================
# PART 1 — DATA LOADING
# SVM uses precomputed 63-dim feature vectors, not raw images
# =============================================================================

def load_landmark_features(task: str, split: str) -> tuple:
    """
    Load precomputed 63-dim landmark feature vectors for SVM training.

    First tries to load from the .npz cache built by preprocess.py
    (fast — a single numpy file load).

    Falls back to reading each JSON file individually if no cache exists.

    Returns:
      X : (N, 63) float32 — normalised landmark feature vectors
      y : (N,)    int     — gesture class labels
    """
    import json as json_mod

    root       = PREPROCESSED_DATASET_PATH
    cache_path = root / "splits" / f"{split}_features.npz"

    if cache_path.exists():
        data = np.load(cache_path)
        X    = data["features"].astype(np.float32)
        y    = data["labels"].astype(np.int32)
        print(f"  [Data] {split}: {len(X)} samples from cache")
        return X, y

    # No cache — read from split file and JSON files directly
    print(f"  [Data] {split}: no cache found, reading JSON files...")
    split_file = get_split_file(task, split)
    if not split_file.exists():
        raise FileNotFoundError(
            f"Split file not found: {split_file}\n"
            f"Run: python code/data/preprocess.py --task {task}  first"
        )

    samples = []
    with open(split_file) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 2:
                samples.append((Path(parts[0]), int(parts[1])))

    features = []
    labels   = []
    skipped  = 0

    for path, gid in samples:
        try:
            with open(path) as f:
                data = json_mod.load(f)
            lms = data["landmarks"]
            vec = np.array([[lm["x"], lm["y"], lm["z"]] for lm in lms],
                           dtype=np.float32).flatten()
            # Normalise: centre on wrist, scale to [-1,1]
            pts   = vec.reshape(21, 3)
            pts   = pts - pts[0]
            scale = np.max(np.abs(pts)) + 1e-8
            vec   = (pts / scale).flatten()
            features.append(vec)
            labels.append(gid)
        except Exception:
            skipped += 1

    print(f"  [Data] {split}: {len(features)} samples loaded ({skipped} skipped)")
    return (np.array(features, dtype=np.float32),
            np.array(labels,   dtype=np.int32))


# =============================================================================
# PART 2 — SVM MODEL
# Pipeline: StandardScaler -> SVC with RBF kernel
# =============================================================================

def build_svm_pipeline(kernel: str, C: float, gamma) -> Pipeline:
    """
    Build an sklearn Pipeline: StandardScaler -> SVC.

    Why StandardScaler?
      The RBF kernel computes ||x - y||^2 between all pairs of samples.
      If features have different scales (e.g. x-coord in [0,1] but z-coord
      in [-0.1, 0.1]), the kernel will be dominated by large-scale features.
      StandardScaler makes all 63 features have mean=0 and std=1 so every
      landmark dimension contributes equally.

    Why class_weight='balanced'?
      Same reason as FocalLoss for CNN/LSTM: emergency gestures may have
      fewer training samples. 'balanced' automatically weights each class
      inversely proportional to its count, so rare classes are not ignored.

    Why probability=True?
      Enables predict_proba() via Platt scaling so we get confidence scores
      (needed for the stability gate in run_webcam.py).

    Parameters
    ----------
    kernel : 'rbf' | 'linear' | 'poly'
    C      : regularisation — higher C = less regularisation = tighter fit
    gamma  : RBF bandwidth — 'scale' uses 1/(n_features * X.var())
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=True,          # enable confidence scores
            class_weight="balanced",   # handle class imbalance
            decision_function_shape="ovr",  # one-vs-rest for 25 classes
            random_state=42,
        )),
    ])


# =============================================================================
# PART 3 — TRAINING
# =============================================================================

def train_svm(pipeline: Pipeline, X_train: np.ndarray,
              y_train: np.ndarray) -> Pipeline:
    """
    Train the SVM on all training landmark vectors.

    SVM training finds support vectors — the boundary samples between
    gesture classes. The RBF kernel maps the 63-dim input to an infinite-
    dimensional feature space where a linear separator exists.

    Time: ~30 seconds to 5 minutes depending on dataset size and C value.
    Higher C = more support vectors = slower training but potentially higher accuracy.
    """
    kernel = pipeline["svm"].kernel
    C      = pipeline["svm"].C
    gamma  = pipeline["svm"].gamma
    n      = len(X_train)

    print(f"\n[SVM] Training on {n} samples")
    print(f"[SVM] kernel={kernel}  C={C}  gamma={gamma}")
    print(f"[SVM] This may take 1-5 minutes for large datasets...")

    pipeline.fit(X_train, y_train)

    n_sv = pipeline["svm"].n_support_.sum()
    print(f"[SVM] Training complete — {n_sv} support vectors found")

    return pipeline


# =============================================================================
# PART 4 — EVALUATION
# =============================================================================

def evaluate_svm(pipeline: Pipeline, X: np.ndarray, y: np.ndarray,
                 split_name: str) -> dict:
    """
    Evaluate the trained SVM on val or test set.
    Prints classification report with precision, recall, F1 per gesture class.
    Returns dict with accuracy and per-class scores.
    """
    preds   = pipeline.predict(X)
    overall = accuracy_score(y, preds)
    names   = [GESTURE_NAMES[i] for i in range(NUM_CLASSES)]

    report_str = classification_report(y, preds, target_names=names, zero_division=0)
    report_d   = classification_report(y, preds, target_names=names,
                                       zero_division=0, output_dict=True)

    print(f"\n{'='*60}")
    print(f"  SVM {split_name.upper()} ACCURACY: {overall:.4f}  ({overall*100:.2f}%)")
    print(f"  Macro F1: {report_d['macro avg']['f1-score']:.4f}")
    print(f"\n{report_str}")
    print("="*60)

    per_class = {
        GESTURE_NAMES[i]: round(
            report_d.get(GESTURE_NAMES[i], {}).get("recall", 0.0), 4)
        for i in range(NUM_CLASSES)
    }

    return {
        "accuracy":    round(overall, 6),
        "macro_f1":    round(report_d["macro avg"]["f1-score"], 4),
        "weighted_f1": round(report_d["weighted avg"]["f1-score"], 4),
        "per_class_accuracy": per_class,
    }


def plot_per_class_accuracy(per_class: dict, log_dir: Path):
    """Save per-class accuracy bar chart to PNG."""
    names  = list(per_class.keys())
    values = [v or 0.0 for v in per_class.values()]
    colors = ["#4CAF50" if v >= 0.90 else "#FFC107" if v >= 0.75 else "#F44336"
              for v in values]

    fig, ax = plt.subplots(figsize=(14, 8))
    bars = ax.barh(names, values, color=colors)
    ax.axvline(0.90, color="#4CAF50", linestyle="--", alpha=0.6, label="90%")
    ax.axvline(0.75, color="#FFC107", linestyle="--", alpha=0.6, label="75%")

    for bar, val in zip(bars, values):
        ax.text(min(val + 0.01, 0.95), bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=8)

    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Accuracy per Gesture Class")
    ax.set_title("SVM — Per-Class Accuracy (25 Gesture Classes)", fontsize=13)
    ax.legend(); plt.tight_layout()
    out = log_dir / "per_class_accuracy.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Per-class accuracy saved -> {out}")


# =============================================================================
# PART 5 — MAIN
# =============================================================================

def train(args):
    """Complete SVM training pipeline."""

    log_dir = get_log_dir(args.exp_name, "svm", args.task)
    print(f"\n[SVM] Task: {args.task}  Log: {log_dir}")

    # Parse gamma (could be 'scale', 'auto', or a float)
    gamma = args.gamma
    if gamma not in ("scale", "auto"):
        try:
            gamma = float(gamma)
        except ValueError:
            gamma = "scale"

    # Load features
    print(f"\n[SVM] Loading features...")
    X_train, y_train = load_landmark_features(args.task, "train")
    X_val,   y_val   = load_landmark_features(args.task, "val")
    X_test,  y_test  = load_landmark_features(args.task, "test")

    print(f"\n[SVM] Feature shapes:")
    print(f"  Train: X={X_train.shape}  y={y_train.shape}")
    print(f"  Val:   X={X_val.shape}    y={y_val.shape}")
    print(f"  Test:  X={X_test.shape}   y={y_test.shape}")

    # Build and train SVM
    pipeline = build_svm_pipeline(args.kernel, args.C, gamma)
    pipeline = train_svm(pipeline, X_train, y_train)

    # Evaluate on val and test
    print(f"\n[SVM] Evaluating on validation set...")
    val_results  = evaluate_svm(pipeline, X_val,  y_val,  "validation")

    print(f"\n[SVM] Evaluating on test set...")
    test_results = evaluate_svm(pipeline, X_test, y_test, "test")

    # Save model
    model_path = log_dir / "svm_model.joblib"
    joblib.dump(pipeline, model_path)
    size_kb = model_path.stat().st_size / 1024
    print(f"\n[SVM] Model saved -> {model_path}  ({size_kb:.1f} KB)")

    # Save eval results
    with open(log_dir / "eval_results.json", "w") as f:
        json.dump(test_results, f, indent=2)

    # Save summary
    summary = {
        "model":           "svm",
        "task":            args.task,
        "kernel":          pipeline["svm"].kernel,
        "C":               pipeline["svm"].C,
        "gamma":           str(pipeline["svm"].gamma),
        "n_support_vectors": int(pipeline["svm"].n_support_.sum()),
        "val_accuracy":    val_results["accuracy"],
        "test_accuracy":   test_results["accuracy"],
        "macro_f1":        test_results["macro_f1"],
        "weighted_f1":     test_results["weighted_f1"],
    }
    with open(log_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Plot per-class accuracy
    plot_per_class_accuracy(test_results["per_class_accuracy"], log_dir)

    print(f"\n{'='*65}")
    print(f"  SVM TRAINING COMPLETE")
    print(f"  Val  accuracy : {val_results['accuracy']:.4f}")
    print(f"  Test accuracy : {test_results['accuracy']:.4f}")
    print(f"  Macro F1      : {test_results['macro_f1']:.4f}")
    print(f"  Model saved   : {model_path}")
    print(f"{'='*65}\n")

    return test_results["accuracy"]


def main():
    ap = argparse.ArgumentParser(
        description="Train SVM gesture classifier — fully supervised"
    )
    ap.add_argument("-t", "--task",      default="gesture",
                    help="Experiment task label (used in log directory name)")
    ap.add_argument("-e", "--exp-name",  default="exp")
    ap.add_argument("-k", "--kernel",    default="rbf",
                    choices=["rbf", "linear", "poly"],
                    help="SVM kernel type")
    ap.add_argument("-C", "--C",         type=float, default=10.0,
                    help="Regularisation strength (higher = less regularisation)")
    ap.add_argument("--gamma",           default="scale",
                    help="RBF kernel coefficient: 'scale' | 'auto' | float")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
