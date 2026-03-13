"""
code/evaluate.py
================
Full evaluation for CNN, LSTM, and SVM models.

Produces:
  - Overall accuracy
  - Per-class accuracy for all 25 gestures
  - Precision, Recall, F1 (sklearn classification_report)
  - Confusion matrix heatmap (PNG)
  - Per-class accuracy bar chart (PNG)
  - Confidence histogram (PNG) — CNN and LSTM only
  - eval_results.json

Run
---
    python code/evaluate.py --task hgrd --model cnn  --checkpoint log/<exp>/best_model.pt
    python code/evaluate.py --task hgrd --model lstm --checkpoint log/<exp>/best_model.pt
    python code/evaluate.py --task hgrd --model svm  --checkpoint log/<exp>/svm_model.joblib
"""

import json
import argparse
import numpy as np
from pathlib import Path

import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from utils.config import GESTURE_NAMES, NUM_CLASSES


# =============================================================================
# LOAD MODEL
# =============================================================================
def load_pytorch_model(checkpoint: Path, model_type: str,
                        device: str = "cpu") -> torch.nn.Module:
    from models.cnn_model import GestureCNN, MobileNetV2Transfer
    from models.lstm_model import LSTMModel
    cls_map = {"cnn": GestureCNN, "mobilenet": MobileNetV2Transfer, "lstm": LSTMModel}
    if model_type not in cls_map:
        raise ValueError(f"Unknown model_type '{model_type}'")
    model = cls_map[model_type]()
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()
    model.to(device)
    print(f"[Evaluate] Loaded {model_type} — "
          f"epoch={state.get('epoch','?')}  val_acc={state.get('val_acc',0):.4f}")
    return model


def load_svm_model(checkpoint: Path):
    from models.svm_classifier import SVMClassifier
    return SVMClassifier.load(checkpoint)


# =============================================================================
# INFERENCE LOOPS
# =============================================================================
@torch.no_grad()
def run_pytorch_inference(model, loader, device: str) -> tuple:
    preds, labels, confs = [], [], []
    for batch in loader:
        x, y = batch
        x    = x.to(device)
        logits = model(x)
        probs  = F.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        preds.extend(pred.cpu().numpy().tolist())
        labels.extend(y.numpy().tolist())
        confs.extend(conf.cpu().numpy().tolist())
    return preds, labels, confs


def run_svm_inference(clf, X: np.ndarray, y: np.ndarray) -> tuple:
    preds  = clf.predict(X).tolist()
    labels = y.tolist()
    probs  = clf.predict_proba(X)
    confs  = probs.max(axis=1).tolist()
    return preds, labels, confs


# =============================================================================
# PLOTS
# =============================================================================
def plot_confusion_matrix(preds, labels, out_path: Path) -> None:
    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_norm = np.where(cm.sum(1, keepdims=True) > 0,
                           cm / cm.sum(1, keepdims=True), 0.0)
    names = [GESTURE_NAMES[i][:8] for i in range(NUM_CLASSES)]
    fig, ax = plt.subplots(figsize=(20, 16))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=names, yticklabels=names,
                linewidths=0.3, ax=ax, vmin=0, vmax=1)
    ax.set_title("Confusion Matrix (row-normalised)", fontsize=14)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.tick_params(axis="y", rotation=0,  labelsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Confusion matrix → {out_path}")


def plot_per_class_accuracy(per_class: dict, out_path: Path) -> None:
    names  = [GESTURE_NAMES[i] for i in range(NUM_CLASSES)]
    values = [per_class.get(GESTURE_NAMES[i], 0) or 0
              for i in range(NUM_CLASSES)]
    colors = ["#4CAF50" if v >= 0.9 else "#FFC107" if v >= 0.75 else "#F44336"
              for v in values]
    fig, ax = plt.subplots(figsize=(14, 7))
    bars = ax.barh(names, values, color=colors)
    ax.axvline(0.9,  color="#4CAF50", linestyle="--", alpha=0.5, label="90%")
    ax.axvline(0.75, color="#FFC107", linestyle="--", alpha=0.5, label="75%")
    for bar, val in zip(bars, values):
        ax.text(min(val + 0.01, 0.95), bar.get_y() + bar.get_height()/2,
                f"{val:.2f}", va="center", fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("Accuracy"); ax.set_title("Per-Class Accuracy — 25 Gestures")
    ax.legend(); plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  Per-class accuracy → {out_path}")


def plot_confidence_histogram(preds, labels, confs, out_path: Path) -> None:
    correct = [c for p, l, c in zip(preds, labels, confs) if p == l]
    wrong   = [c for p, l, c in zip(preds, labels, confs) if p != l]
    bins = np.linspace(0, 1, 25)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(correct, bins=bins, alpha=0.7, color="#4CAF50",
            label=f"Correct (n={len(correct)})")
    ax.hist(wrong,   bins=bins, alpha=0.7, color="#F44336",
            label=f"Wrong (n={len(wrong)})")
    ax.axvline(0.75, color="#2196F3", linestyle="--", label="Threshold=0.75")
    ax.set_xlabel("Confidence"); ax.set_ylabel("Count")
    ax.set_title("Prediction Confidence Distribution")
    ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight"); plt.close()
    print(f"  Confidence histogram → {out_path}")


# =============================================================================
# MAIN EVALUATE FUNCTION
# =============================================================================
def evaluate(task: str, model_type: str, checkpoint: Path,
             device: str = "cpu") -> dict:

    out_dir = checkpoint.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model + run inference ────────────────────────────────────────────
    if model_type == "svm":
        clf    = load_svm_model(checkpoint)
        from data.dataset import load_svm_data
        X, y   = load_svm_data(task, "test")
        preds, labels, confs = run_svm_inference(clf, X, y)
    else:
        model  = load_pytorch_model(checkpoint, model_type, device)
        from data.dataset import create_dataloaders
        loaders = create_dataloaders(task, model_type, num_workers=0)
        preds, labels, confs = run_pytorch_inference(model, loaders["test"], device)

    # ── Metrics ───────────────────────────────────────────────────────────────
    overall = sum(p == l for p, l in zip(preds, labels)) / len(labels)
    names   = [GESTURE_NAMES[i] for i in range(NUM_CLASSES)]
    report  = classification_report(labels, preds, target_names=names,
                                    zero_division=0, output_dict=True)
    print(f"\n{'='*60}")
    print(f"  Model: {model_type}  Task: {task}")
    print(f"  Overall Accuracy: {overall:.4f}  ({overall*100:.2f}%)")
    print(f"  Macro F1:         {report['macro avg']['f1-score']:.4f}")
    print(classification_report(labels, preds, target_names=names, zero_division=0))
    print("="*60)

    per_class = {}
    for gid in range(NUM_CLASSES):
        mask = [l == gid for l in labels]
        if sum(mask) == 0:
            per_class[GESTURE_NAMES[gid]] = None
            continue
        correct = sum(p == l for p, l in zip(preds, labels) if l == gid)
        per_class[GESTURE_NAMES[gid]] = round(correct / sum(mask), 4)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_confusion_matrix(preds, labels, out_dir / "confusion_matrix.png")
    plot_per_class_accuracy(per_class,   out_dir / "per_class_accuracy.png")
    plot_confidence_histogram(preds, labels, confs,
                              out_dir / "confidence_histogram.png")

    results = {
        "model_type":         model_type,
        "task":               task,
        "overall_accuracy":   round(overall, 6),
        "macro_f1":           round(report["macro avg"]["f1-score"],    4),
        "weighted_f1":        round(report["weighted avg"]["f1-score"], 4),
        "per_class_accuracy": per_class,
        "checkpoint":         str(checkpoint),
    }
    with open(out_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Evaluate] Results → {out_dir}/eval_results.json")
    return results


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-t", "--task",       required=True, choices=["hgrd","custom"])
    ap.add_argument("-m", "--model",      required=True,
                    choices=["cnn","mobilenet","lstm","svm"])
    ap.add_argument("-c", "--checkpoint", required=True, type=Path)
    ap.add_argument("--device",           default="cpu")
    args = ap.parse_args()
    evaluate(args.task, args.model, args.checkpoint, device=args.device)
