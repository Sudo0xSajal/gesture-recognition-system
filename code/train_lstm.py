"""
code/train_lstm.py
==================
COMPLETE LSTM/RNN Training — Step 3 of Ma'am's Document
Algorithm: Recurrent Neural Network (RNN) — implemented as LSTM

This ONE file contains everything needed to train the LSTM model:
  - Dataset loading (temporal sequences of 15 landmark frames)
  - LSTMModel architecture (Bidirectional LSTM + attention)
  - FocalLoss definition
  - Full training loop (forward, BPTT, optimizer step)
  - Validation after every epoch
  - Early stopping
  - Best model checkpoint saving
  - Training curve plot
  - Final test evaluation with per-class report

Why LSTM/RNN for gesture recognition?
--------------------------------------
Some gestures are MOVEMENTS, not static poses:
  - Hand Wave (#3)    — hand moves back and forth over time
  - Hand Shake (#19)  — rapid trembling motion
  - Circle Motion (#14) — finger draws a circle
  - Two Finger Swipe (#16) — finger swipes left to right

A CNN only sees ONE frame and cannot tell if the hand is moving or still.
An LSTM sees 15 consecutive frames and learns the MOTION PATTERN over time.
RNN stands for Recurrent Neural Network — LSTM is the most popular variant.

Run
---
    python code/train_lstm.py --task hgrd --exp-name exp1
    python code/train_lstm.py --task hgrd --exp-name exp1 --lr 3e-4 --epochs 150
    python code/train_lstm.py --task hgrd --exp-name exp1 --gpu 0

Or via shell:
    bash train_lstm.sh -c 0 -t hgrd -e exp1 -l 3e-4 -s 42

Output saved to:
    log/exp1_s42_lstm_hgrd/
        best_model.pt
        metrics.json
        summary.json
        training_history.png
        eval_results.json
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.metrics import classification_report

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.config import (
    get_split_file, get_log_dir,
    GESTURE_NAMES, NUM_CLASSES, MODEL_CONFIG,
)


# =============================================================================
# PART 1 — DATASET
# Builds temporal sequences from consecutive landmark frames
# =============================================================================

def load_landmark_vector(json_path: Path) -> np.ndarray:
    """
    Load one MediaPipe JSON file -> 63-dim normalised landmark vector.
    If the file is missing or malformed, returns a zero vector.

    Normalisation:
      1. Subtract wrist position -> hand centred at origin
      2. Divide by max absolute value -> scale to [-1, 1]
    This makes the vector invariant to where the hand is on screen,
    so the model learns hand SHAPE not hand POSITION.
    """
    import json as json_mod
    try:
        with open(json_path) as f:
            data = json_mod.load(f)
        landmarks = data["landmarks"]
        vec = np.array([[lm["x"], lm["y"], lm["z"]] for lm in landmarks],
                       dtype=np.float32).flatten()           # (63,)
        pts   = vec.reshape(21, 3)
        pts   = pts - pts[0]                                 # centre on wrist
        scale = np.max(np.abs(pts)) + 1e-8
        return (pts / scale).flatten().astype(np.float32)   # (63,)
    except Exception:
        return np.zeros(63, dtype=np.float32)


class GestureLSTMDataset(Dataset):
    """
    Builds temporal sequences of 15 consecutive landmark frames for LSTM.

    Each sample is a (seq_len, 63) array representing the hand's motion
    over seq_len consecutive video frames.

    The dataset is built by reading the split .txt file, grouping frames
    by (gesture, session folder), then creating overlapping windows of
    seq_len frames with 50% overlap (sliding window).

    If a session has fewer than seq_len frames, the first frame is
    repeated to pad the sequence to the required length.

    Returns: (sequence tensor [seq_len, 63], gesture_label int)
    """

    def __init__(self, split_file: Path, seq_len: int = 15,
                 augment: bool = False):
        if not split_file.exists():
            raise FileNotFoundError(
                f"\nSplit file not found: {split_file}"
                f"\nFix: run  python code/data/preprocess.py --task <task>  first"
            )
        self.seq_len = seq_len
        self.augment = augment

        # Read all (path, label) pairs from the split file
        raw_samples = []
        with open(split_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    raw_samples.append((Path(parts[0]), int(parts[1])))

        # Group consecutive frames by (gesture_id, parent_folder)
        # so we build sequences only from frames of the same gesture session
        sessions = {}
        for path, gid in raw_samples:
            key = (gid, str(path.parent))
            sessions.setdefault(key, []).append(path)

        # Build sliding-window sequences
        self.sequences = []
        for (gid, _), paths in sessions.items():
            paths_sorted = sorted(paths)          # sort by filename for time order
            n = len(paths_sorted)

            if n < seq_len:
                # Pad by repeating the first frame at the start
                pad     = [paths_sorted[0]] * (seq_len - n)
                windows = [pad + paths_sorted]
            else:
                # Sliding window with 50% overlap
                step    = max(1, seq_len // 2)
                windows = [
                    paths_sorted[i: i + seq_len]
                    for i in range(0, n - seq_len + 1, step)
                ]

            for window in windows:
                self.sequences.append((window, gid))

        print(f"  [Dataset] {split_file.name}: {len(self.sequences)} sequences "
              f"(seq_len={seq_len})")

    def __len__(self):
        return len(self.sequences)

    def get_labels(self):
        return [s[1] for s in self.sequences]

    def __getitem__(self, idx):
        paths, label = self.sequences[idx]

        # Load each frame's landmark vector -> stack into (seq_len, 63)
        vecs = [load_landmark_vector(p) for p in paths]
        seq  = np.stack(vecs, axis=0).astype(np.float32)    # (seq_len, 63)

        # Light augmentation: add small Gaussian noise to coordinates
        if self.augment:
            noise = np.random.normal(0, 0.008, seq.shape).astype(np.float32)
            seq   = seq + noise

        return torch.from_numpy(seq), label


def build_dataloaders(task: str, seq_len: int, batch_size: int, num_workers: int):
    """
    Create train, val, and test DataLoaders for LSTM training.
    Uses WeightedRandomSampler for equal class representation per batch.
    """
    train_ds = GestureLSTMDataset(get_split_file(task, "train"),
                                   seq_len, augment=True)
    val_ds   = GestureLSTMDataset(get_split_file(task, "val"),
                                   seq_len, augment=False)
    test_ds  = GestureLSTMDataset(get_split_file(task, "test"),
                                   seq_len, augment=False)

    # Weighted sampler: equal class probability per batch
    labels       = train_ds.get_labels()
    class_counts = np.bincount(labels, minlength=NUM_CLASSES).astype(float)
    class_counts = np.where(class_counts == 0, 1.0, class_counts)
    sample_weights = [1.0 / class_counts[l] for l in labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights),
                                    replacement=True)

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=pin, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin)

    return train_loader, val_loader, test_loader


# =============================================================================
# PART 2 — MODEL ARCHITECTURE
# Bidirectional LSTM with Temporal Attention
# =============================================================================

class TemporalAttention(nn.Module):
    """
    Temporal attention layer — learns which frames in the sequence
    are most important for classifying the gesture.

    Example: For 'Hand Wave' the peak of the wave is most informative.
    For 'Chest Hold' the moment of contact is most informative.

    How it works:
      - Projects each timestep's LSTM output to a scalar score
      - Applies softmax to get a probability distribution over timesteps
      - Weighted sum of all timestep outputs -> single context vector

    Input:  (batch, seq_len, hidden_size)
    Output: (batch, hidden_size)
    """
    def __init__(self, hidden_size: int):
        super().__init__()
        self.attention = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, lstm_out: torch.Tensor) -> torch.Tensor:
        """
        lstm_out: (N, seq_len, hidden_size)
        returns:  (N, hidden_size) — attended context vector
        """
        scores  = self.attention(lstm_out)        # (N, seq_len, 1)
        weights = F.softmax(scores, dim=1)        # (N, seq_len, 1)
        context = (weights * lstm_out).sum(dim=1) # (N, hidden_size)
        return context


class LSTMModel(nn.Module):
    """
    Bidirectional LSTM (RNN) for temporal gesture sequence classification.

    Input:  (batch, seq_len=15, input_size=63)  — 15 frames of 63 landmarks
    Output: (batch, 25)                          — gesture class logits

    Architecture:
      Input projection: Linear(63 -> 128) + LayerNorm
        -> keeps landmark dimensions in a good range for LSTM

      BiLSTM Layer 1: hidden=128, input=128
        -> reads sequence forward AND backward, output size = 256

      BiLSTM Layer 2: hidden=128, input=256
        -> deeper temporal modelling, output size = 256

      Temporal Attention: over 256-dim outputs
        -> weighted sum across the 15 time steps

      Classifier:
        Linear(256 -> 128) -> LayerNorm -> GELU -> Dropout(0.3)
        Linear(128 -> 25)

    Why Bidirectional?
      Reading both forward and backward captures context from both
      the start AND end of the gesture motion, giving better accuracy.

    Total parameters: ~900K
    """
    def __init__(self, input_size: int = 63,
                 hidden_size: int = 128, num_layers: int = 2,
                 num_classes: int = NUM_CLASSES, dropout: float = 0.3):
        super().__init__()

        # Project input landmarks to a better-sized feature space
        self.input_proj = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
        )

        # Bidirectional LSTM: output size = hidden_size * 2 = 256
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,    # input shape: (batch, seq, features)
            bidirectional=True,  # reads forward and backward
            dropout=dropout if num_layers > 1 else 0.0,
        )

        lstm_out_size = hidden_size * 2   # 256 because bidirectional

        # Temporal attention over all 15 timesteps
        self.attention = TemporalAttention(lstm_out_size)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_size, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self):
        """Initialise LSTM forget gate bias to 1.0 to help remember long sequences."""
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
                # Forget gate bias = 1.0 (prevents forgetting too quickly)
                n = param.size(0)
                param.data[n // 4: n // 2].fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LSTM.
        x: (N, seq_len=15, 63) — sequence of landmark frames
        returns: (N, 25) — class logits
        """
        # Project each frame's 63 landmarks to 128 dims
        x = self.input_proj(x)          # (N, 15, 128)

        # Run through bidirectional LSTM
        lstm_out, _ = self.lstm(x)      # (N, 15, 256)

        # Temporal attention: which frames matter most?
        context = self.attention(lstm_out)  # (N, 256)

        # Classify
        return self.classifier(context)     # (N, 25)


# =============================================================================
# PART 3 — LOSS FUNCTION
# (same FocalLoss as CNN — handles class imbalance)
# =============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss — same as used for CNN.
    See train_cnn.py PART 3 for full explanation.
    alpha=0.25, gamma=2.0 focuses training on hard/rare gestures.
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        return (self.alpha * (1.0 - pt) ** self.gamma * ce).mean()


# =============================================================================
# PART 4 — TRAINING LOOP
# =============================================================================

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def run_train_epoch(model, loader, criterion, optimizer,
                    scaler, device, use_amp) -> tuple:
    """
    One full pass through the LSTM training set.

    For each batch of sequences:
      1. Forward:  (N,15,63) -> LSTM -> (N,25) logits
      2. Loss:     FocalLoss(logits, true_labels)
      3. Backward: compute gradients through time (BPTT)
      4. Clip:     max_norm=1.0 prevents exploding gradients
      5. Update:   AdamW updates all LSTM, attention, classifier weights

    Returns: (average_loss, accuracy)
    """
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for sequences, labels in loader:
        sequences = sequences.to(device, non_blocking=True)  # (N, 15, 63)
        labels    = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=use_amp):
            logits = model(sequences)               # (N, 25)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        preds       = logits.detach().argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    return total_loss / len(loader), correct / total


@torch.no_grad()
def run_eval_epoch(model, loader, criterion, device, use_amp) -> tuple:
    """
    Evaluate LSTM on validation or test set — no weight updates.
    Returns: (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for sequences, labels in loader:
        sequences = sequences.to(device, non_blocking=True)
        labels    = labels.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            logits = model(sequences)
            loss   = criterion(logits, labels)

        total_loss += loss.item()
        preds       = logits.argmax(dim=1)
        correct    += (preds == labels).sum().item()
        total      += labels.size(0)

    return total_loss / len(loader), correct / total


@torch.no_grad()
def run_test_evaluation(model, loader, device, log_dir: Path) -> dict:
    """Final per-class accuracy report on test set. Saves eval_results.json."""
    model.eval()
    all_preds  = []
    all_labels = []

    for sequences, labels in loader:
        sequences = sequences.to(device)
        preds     = model(sequences).argmax(dim=1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.numpy().tolist())

    overall    = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    names      = [GESTURE_NAMES[i] for i in range(NUM_CLASSES)]
    report_str = classification_report(all_labels, all_preds,
                                       target_names=names, zero_division=0)
    report_d   = classification_report(all_labels, all_preds,
                                       target_names=names, zero_division=0,
                                       output_dict=True)

    print(f"\n{'='*60}")
    print(f"  LSTM TEST ACCURACY: {overall:.4f}  ({overall*100:.2f}%)")
    print(f"  Macro F1: {report_d['macro avg']['f1-score']:.4f}")
    print(f"\n{report_str}")
    print("="*60)

    results = {
        "test_accuracy": round(overall, 6),
        "macro_f1":      round(report_d["macro avg"]["f1-score"], 4),
        "weighted_f1":   round(report_d["weighted avg"]["f1-score"], 4),
        "per_class_accuracy": {
            GESTURE_NAMES[i]: round(
                report_d.get(GESTURE_NAMES[i], {}).get("recall", 0.0), 4)
            for i in range(NUM_CLASSES)
        },
    }
    with open(log_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def save_checkpoint(model, optimizer, epoch, val_acc, path: Path):
    torch.save({
        "epoch":       epoch,
        "val_acc":     val_acc,
        "model_state": model.state_dict(),
        "opt_state":   optimizer.state_dict(),
    }, path)


def plot_training_curves(history: list, best_epoch: int, log_dir: Path):
    epochs  = [r["epoch"]      for r in history]
    tr_acc  = [r["train_acc"]  for r in history]
    va_acc  = [r["val_acc"]    for r in history]
    tr_loss = [r["train_loss"] for r in history]
    va_loss = [r["val_loss"]   for r in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs, tr_acc,  label="Train Acc", color="#2196F3", linewidth=2)
    ax1.plot(epochs, va_acc,  label="Val Acc",   color="#4CAF50", linewidth=2)
    ax1.axvline(best_epoch, color="#F44336", linestyle="--",
                alpha=0.7, label=f"Best epoch {best_epoch}")
    ax1.set_title("LSTM Accuracy", fontsize=13)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, tr_loss, label="Train Loss", color="#2196F3", linewidth=2)
    ax2.plot(epochs, va_loss, label="Val Loss",   color="#4CAF50", linewidth=2)
    ax2.set_title("LSTM Loss", fontsize=13)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Focal Loss")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.suptitle("LSTM (RNN) Training", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = log_dir / "training_history.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Training curves saved -> {out}")


# =============================================================================
# PART 5 — MAIN
# =============================================================================

def train(args):
    """Complete LSTM training pipeline."""
    set_seed(args.seed)

    if args.gpu >= 0 and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = torch.device("cuda")
        print(f"\n[LSTM] Using GPU {args.gpu}: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print(f"\n[LSTM] Using CPU")

    log_dir = get_log_dir(f"{args.exp_name}_s{args.seed}", "lstm", args.task)
    print(f"[LSTM] Task: {args.task}  Seed: {args.seed}  Log: {log_dir}")

    # Load data
    print(f"\n[LSTM] Loading dataset (seq_len={args.seq_len})...")
    train_loader, val_loader, test_loader = build_dataloaders(
        args.task, args.seq_len, args.batch_size, args.num_workers
    )

    # Build model
    print(f"\n[LSTM] Building model...")
    model = LSTMModel(
        input_size=63, hidden_size=128, num_layers=2,
        num_classes=NUM_CLASSES, dropout=0.3,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[LSTM] BiLSTM + Attention — {n_params:,} parameters")

    # Loss, optimizer, scheduler
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6, verbose=True
    )
    use_amp = args.use_amp and device.type == "cuda"
    scaler  = GradScaler(enabled=use_amp)

    print(f"[LSTM] lr={args.lr}  batch={args.batch_size}  "
          f"max_epochs={args.epochs}  patience={args.patience}  AMP={use_amp}\n")

    # Training loop
    best_val_acc = 0.0
    best_epoch   = 0
    patience_cnt = 0
    history      = []

    print(f"{'='*65}")
    print(f"  TRAINING LSTM/RNN — max {args.epochs} epochs")
    print(f"{'='*65}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc = run_train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp
        )
        va_loss, va_acc = run_eval_epoch(
            model, val_loader, criterion, device, use_amp
        )

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]

        print(f"  Epoch {epoch:4d}/{args.epochs}"
              f"  tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.4f}"
              f"  va_loss={va_loss:.4f}  va_acc={va_acc:.4f}"
              f"  lr={lr_now:.1e}  [{elapsed:.1f}s]")

        history.append({
            "epoch":      epoch,
            "train_loss": round(tr_loss, 6),
            "train_acc":  round(tr_acc,  6),
            "val_loss":   round(va_loss, 6),
            "val_acc":    round(va_acc,  6),
            "lr":         lr_now,
        })
        with open(log_dir / "metrics.json", "w") as f:
            json.dump(history, f, indent=2)

        scheduler.step(va_acc)

        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_epoch   = epoch
            patience_cnt = 0
            save_checkpoint(model, optimizer, epoch, va_acc,
                            log_dir / "best_model.pt")
            print(f"  ✅  New best val_acc = {va_acc:.4f}  -> best_model.pt saved")
        else:
            patience_cnt += 1

        if epoch % 10 == 0:
            save_checkpoint(model, optimizer, epoch, va_acc,
                            log_dir / f"ckpt_epoch{epoch:04d}.pt")

        if patience_cnt >= args.patience:
            print(f"\n[LSTM] Early stopping at epoch {epoch}")
            break

    # Final test evaluation
    print(f"\n[LSTM] Loading best model (epoch {best_epoch})...")
    ckpt = torch.load(log_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])

    test_results = run_test_evaluation(model, test_loader, device, log_dir)

    summary = {
        "model":         "lstm",
        "task":          args.task,
        "seed":          args.seed,
        "best_val_acc":  round(best_val_acc, 6),
        "best_epoch":    best_epoch,
        "test_accuracy": test_results["test_accuracy"],
        "macro_f1":      test_results["macro_f1"],
        "total_epochs":  len(history),
    }
    with open(log_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    plot_training_curves(history, best_epoch, log_dir)

    print(f"\n{'='*65}")
    print(f"  LSTM TRAINING COMPLETE")
    print(f"  Best val accuracy : {best_val_acc:.4f}  (epoch {best_epoch})")
    print(f"  Test accuracy     : {test_results['test_accuracy']:.4f}")
    print(f"  Macro F1          : {test_results['macro_f1']:.4f}")
    print(f"  Checkpoint        : {log_dir / 'best_model.pt'}")
    print(f"{'='*65}\n")

    return best_val_acc


def main():
    ap = argparse.ArgumentParser(
        description="Train LSTM/RNN gesture model — fully supervised"
    )
    ap.add_argument("-t", "--task",      required=True, choices=["hgrd", "custom"])
    ap.add_argument("-e", "--exp-name",  default="exp")
    ap.add_argument("-s", "--seed",      type=int,   default=42)
    ap.add_argument("-c", "--gpu",       type=int,   default=0)
    ap.add_argument("-l", "--lr",        type=float, default=3e-4)
    ap.add_argument("--seq-len",         type=int,   default=15,
                    help="Number of consecutive frames per sequence")
    ap.add_argument("--batch-size",      type=int,   default=32)
    ap.add_argument("--epochs",          type=int,   default=150)
    ap.add_argument("--patience",        type=int,   default=20)
    ap.add_argument("--num-workers",     type=int,   default=4)
    ap.add_argument("--use-amp",         action="store_true", default=True)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
