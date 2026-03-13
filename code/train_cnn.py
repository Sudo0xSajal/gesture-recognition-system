"""
code/train_cnn.py
=================
COMPLETE CNN Training — Step 3 of Ma'am's Document
Algorithm: Convolutional Neural Network (CNN)

This ONE file contains everything needed to train the CNN model:
  - Dataset loading (224x224 images)
  - Data augmentation pipeline
  - GestureCNN model architecture (ResBlock stack)
  - FocalLoss definition
  - Full training loop (forward pass, backprop, optimizer step)
  - Validation after every epoch
  - Early stopping
  - Best model checkpoint saving
  - Training curve plot
  - Final test evaluation with per-class report

CNN is best for SINGLE-FRAME gestures: Thumb Up, Open Palm, Chest Hold, etc.
It looks at ONE image and classifies which of the 25 gestures it is.

Run
---
    python code/train_cnn.py --task hgrd --exp-name exp1
    python code/train_cnn.py --task hgrd --exp-name exp1 --lr 3e-4 --epochs 150
    python code/train_cnn.py --task hgrd --exp-name exp1 --gpu 0
    python code/train_cnn.py --task custom --exp-name myexp

Or via shell:
    bash train_cnn.sh -c 0 -t hgrd -e exp1 -l 3e-4 -s 42

Output saved to:
    log/exp1_s42_cnn_hgrd/
        best_model.pt           <- load this for inference
        metrics.json            <- loss and accuracy per epoch
        summary.json            <- final accuracy numbers
        training_history.png    <- accuracy and loss plots
        eval_results.json       <- per-class accuracy on test set
"""

import os
import sys
import json
import time
import random
import argparse
from pathlib import Path

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import classification_report

sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils.config import (
    get_split_file, get_log_dir,
    GESTURE_NAMES, NUM_CLASSES,
)


# =============================================================================
# PART 1 — DATASET
# How images are loaded and prepared for training
# =============================================================================

def get_train_transforms():
    """
    Augmentation applied during training only.
    These transforms make the model robust to:
      - Hand at different angles (Rotate, Affine)
      - Different room lighting (BrightnessContrast, CLAHE)
      - Camera noise and blur (GaussNoise, GaussianBlur)
      - Partial occlusion by bedsheets (CoarseDropout)
    Validation and test sets use NO augmentation.
    """
    return A.Compose([
        A.Resize(224, 224),
        A.Rotate(limit=25, p=0.6),
        A.Affine(scale=(0.75, 1.25), translate_percent=0.15, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(0.3, 0.3, p=0.5),
        A.HueSaturationValue(20, 30, 20, p=0.3),
        A.CLAHE(clip_limit=4.0, p=0.3),
        A.GaussNoise(var_limit=(5, 30), p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.CoarseDropout(max_holes=4, max_height=32, max_width=32, p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms():
    """Only resize + normalise for validation and test sets."""
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


class GestureImageDataset(Dataset):
    """
    Loads 224x224 RGB gesture images for CNN training.

    Reads split .txt file (created by preprocess.py).
    Each line in the .txt file is:
        /path/to/landmarks_000001.json   <gesture_id>

    The image file is found by replacing 'landmarks_' with 'frame_'
    and '.json' with '.jpg'.

    Returns: (image_tensor shape [3,224,224], gesture_label int)
    """

    def __init__(self, split_file: Path, augment: bool = False):
        if not split_file.exists():
            raise FileNotFoundError(
                f"\nSplit file not found: {split_file}"
                f"\nFix: run  python code/data/preprocess.py --task <task>  first"
            )
        self.samples = []
        with open(split_file) as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    self.samples.append((Path(parts[0]), int(parts[1])))

        self.transform = get_train_transforms() if augment else get_val_transforms()
        print(f"  [Dataset] {split_file.name}: {len(self.samples)} images")

    def __len__(self):
        return len(self.samples)

    def get_labels(self):
        """Returns list of all gesture labels — used for WeightedRandomSampler."""
        return [s[1] for s in self.samples]

    def __getitem__(self, idx):
        lm_path, label = self.samples[idx]

        # Derive image path from landmark json path
        img_path = Path(str(lm_path)
                        .replace("landmarks_", "frame_")
                        .replace(".json", ".jpg"))

        if img_path.exists():
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            # Fallback: blank image (should not happen after proper collection)
            img = np.zeros((224, 224, 3), dtype=np.uint8)

        tensor = self.transform(image=img)["image"]
        return tensor, label


def build_dataloaders(task: str, batch_size: int, num_workers: int):
    """
    Create train, val, and test DataLoaders.

    WeightedRandomSampler ensures every gesture class has equal probability
    of appearing each batch. Without this, the CNN would see 'Thumb Up'
    (a common everyday gesture) far more than 'Both Hands Raise' (emergency).
    With equal sampling, the CNN learns all 25 classes equally well.
    """
    train_ds = GestureImageDataset(get_split_file(task, "train"), augment=True)
    val_ds   = GestureImageDataset(get_split_file(task, "val"),   augment=False)
    test_ds  = GestureImageDataset(get_split_file(task, "test"),  augment=False)

    # Weighted sampler: make every class equally likely per batch
    labels       = train_ds.get_labels()
    class_counts = np.bincount(labels, minlength=NUM_CLASSES).astype(float)
    class_counts = np.where(class_counts == 0, 1.0, class_counts)
    class_weights = 1.0 / class_counts
    sample_weights = [class_weights[l] for l in labels]
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
# The CNN that classifies gesture images
# =============================================================================

class ResBlock(nn.Module):
    """
    Residual block: two conv layers + skip connection.

    The skip connection (input added back to output) prevents vanishing
    gradients in deep networks. Without it, gradients shrink to near-zero
    by the time they reach early layers and training stalls.

    in_ch  -> conv3x3 -> BN -> ReLU -> conv3x3 -> BN -> + skip -> ReLU -> out_ch
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        # Skip connection: if shape changes, use 1x1 conv to match
        if in_ch != out_ch or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        else:
            self.skip = nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv2(self.conv1(x)) + self.skip(x))


class GestureCNN(nn.Module):
    """
    CNN for 25-class gesture image classification.

    Input:  (batch, 3, 224, 224)  — RGB image tensor
    Output: (batch, 25)           — raw class scores (logits)

    Architecture (feature extraction):
      Stage 1: 3   -> 32  filters  (224x224)  first conv
      Stage 2: 32  -> 64  filters  (112x112)  stride 2
      Stage 3: 64  -> 128 filters  (56x56)    stride 2
      Stage 4: 128 -> 256 filters  (28x28)    stride 2
      Stage 5: 256 -> 512 filters  (14x14)    stride 2
      GlobalAvgPool -> (512,)
    Classifier:
      Linear(512 -> 256) -> BN -> ReLU -> Dropout(0.4)
      Linear(256 -> 25)

    Total parameters: ~4.2 million
    """
    def __init__(self, num_classes: int = NUM_CLASSES, dropout: float = 0.4):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.stage2 = ResBlock(32,  64,  stride=2)
        self.stage3 = ResBlock(64,  128, stride=2)
        self.stage4 = ResBlock(128, 256, stride=2)
        self.stage5 = ResBlock(256, 512, stride=2)
        self.gap    = nn.AdaptiveAvgPool2d(1)      # -> (batch, 512, 1, 1)
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through all stages.
        Input x: (N, 3, 224, 224)
        Output:  (N, 25) logits
        """
        x = self.stage1(x)  # (N, 32,  224, 224)
        x = self.stage2(x)  # (N, 64,  112, 112)
        x = self.stage3(x)  # (N, 128,  56,  56)
        x = self.stage4(x)  # (N, 256,  28,  28)
        x = self.stage5(x)  # (N, 512,  14,  14)
        x = self.gap(x)     # (N, 512,   1,   1)
        x = x.flatten(1)    # (N, 512)
        return self.classifier(x)  # (N, 25)


# =============================================================================
# PART 3 — LOSS FUNCTION
# How training error is measured
# =============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss — handles class imbalance in gesture datasets.

    Problem: some gestures (Thumb Up) have many training samples,
    emergency gestures (Chest Hold, Both Hands Raise) have fewer.
    Standard CrossEntropy treats all samples equally so the model
    learns easy common gestures well but ignores hard rare ones.

    Focal Loss multiplies each sample's loss by (1 - confidence)^gamma.
    - If model is already confident (pt high) → (1-pt)^2 is small → less weight
    - If model is wrong or uncertain (pt low) → (1-pt)^2 is large → more weight
    - This forces the model to focus on the hard/rare emergency gestures

    Formula: FL(pt) = -alpha * (1-pt)^gamma * log(pt)
    alpha=0.25, gamma=2.0 — standard values from Lin et al. 2017
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  (N, 25) raw model output
        targets: (N,)    integer class labels
        """
        # Standard cross entropy per sample (no reduction yet)
        ce = F.cross_entropy(logits, targets, reduction="none")
        # Convert loss to probability: pt = e^(-ce)
        pt = torch.exp(-ce)
        # Apply focal weighting
        focal_loss = self.alpha * (1.0 - pt) ** self.gamma * ce
        return focal_loss.mean()


# =============================================================================
# PART 4 — TRAINING LOOP
# The actual training, epoch by epoch
# =============================================================================

def set_seed(seed: int):
    """Fix all random seeds so results are reproducible across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_train_epoch(model, loader, criterion, optimizer,
                    scaler, device, use_amp) -> tuple:
    """
    One full pass through the training set.

    For each batch:
      1. Forward pass: images -> CNN -> logits
      2. Compute FocalLoss(logits, true_labels)
      3. Backward pass: compute gradients
      4. Gradient clipping: prevent exploding gradients
      5. Optimizer step: update CNN weights
      6. AMP scaler step: handles float16 precision safely

    Returns: (average_loss_this_epoch, accuracy_this_epoch)
    """
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Forward pass (AMP uses float16 on CUDA for speed)
        with autocast(enabled=use_amp):
            logits = model(images)           # (N, 25)
            loss   = criterion(logits, labels)

        # Backward pass
        scaler.scale(loss).backward()

        # Gradient clipping (max_norm=1.0 prevents exploding gradients)
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update weights
        scaler.step(optimizer)
        scaler.update()

        # Track metrics
        total_loss += loss.item()
        predictions = logits.detach().argmax(dim=1)
        correct     += (predictions == labels).sum().item()
        total       += labels.size(0)

    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def run_eval_epoch(model, loader, criterion, device, use_amp) -> tuple:
    """
    One full pass through val or test set — no weight updates.
    model.eval() disables dropout and batch norm uses running statistics.
    Returns: (average_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            logits = model(images)
            loss   = criterion(logits, labels)

        total_loss += loss.item()
        predictions = logits.argmax(dim=1)
        correct     += (predictions == labels).sum().item()
        total       += labels.size(0)

    return total_loss / len(loader), correct / total


@torch.no_grad()
def run_test_evaluation(model, loader, device, log_dir: Path) -> dict:
    """
    Final evaluation on test set with full per-class accuracy report.
    Saves eval_results.json with accuracy for all 25 gesture classes.
    """
    model.eval()
    all_preds  = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        preds  = model(images).argmax(dim=1).cpu().numpy()
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
    print(f"  CNN TEST ACCURACY: {overall:.4f}  ({overall*100:.2f}%)")
    print(f"  Macro F1: {report_d['macro avg']['f1-score']:.4f}")
    print(f"\n{report_str}")
    print("="*60)

    per_class = {
        GESTURE_NAMES[i]: round(
            report_d.get(GESTURE_NAMES[i], {}).get("recall", 0.0), 4)
        for i in range(NUM_CLASSES)
    }

    results = {
        "test_accuracy": round(overall, 6),
        "macro_f1":      round(report_d["macro avg"]["f1-score"], 4),
        "weighted_f1":   round(report_d["weighted avg"]["f1-score"], 4),
        "per_class_accuracy": per_class,
    }
    with open(log_dir / "eval_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


def save_checkpoint(model, optimizer, epoch, val_acc, path: Path):
    """Save model weights + optimizer state to disk."""
    torch.save({
        "epoch":       epoch,
        "val_acc":     val_acc,
        "model_state": model.state_dict(),
        "opt_state":   optimizer.state_dict(),
    }, path)


def plot_training_curves(history: list, best_epoch: int, log_dir: Path):
    """Save training and validation accuracy/loss curves to PNG."""
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
    ax1.set_title("CNN Accuracy", fontsize=13)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1); ax1.legend(); ax1.grid(alpha=0.3)

    ax2.plot(epochs, tr_loss, label="Train Loss", color="#2196F3", linewidth=2)
    ax2.plot(epochs, va_loss, label="Val Loss",   color="#4CAF50", linewidth=2)
    ax2.set_title("CNN Loss", fontsize=13)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Focal Loss")
    ax2.legend(); ax2.grid(alpha=0.3)

    plt.suptitle(f"GestureCNN Training", fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = log_dir / "training_history.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Training curves saved -> {out}")


# =============================================================================
# PART 5 — MAIN: puts everything together
# =============================================================================

def train(args):
    """
    Full CNN training pipeline from data loading to saved model.
    """
    set_seed(args.seed)

    # Device setup
    if args.gpu >= 0 and torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        device = torch.device("cuda")
        print(f"\n[CNN] Using GPU {args.gpu}: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print(f"\n[CNN] Using CPU")

    log_dir = get_log_dir(f"{args.exp_name}_s{args.seed}", "cnn", args.task)
    print(f"[CNN] Task: {args.task}  Seed: {args.seed}  Log: {log_dir}")

    # --- Load data ---
    print(f"\n[CNN] Loading dataset...")
    train_loader, val_loader, test_loader = build_dataloaders(
        args.task, args.batch_size, args.num_workers
    )

    # --- Build model ---
    print(f"\n[CNN] Building model...")
    model = GestureCNN(num_classes=NUM_CLASSES, dropout=0.4).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[CNN] GestureCNN — {n_params:,} parameters")

    # --- Loss, optimizer, scheduler ---
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(),
                                   lr=args.lr, weight_decay=1e-4)
    # ReduceLROnPlateau: cuts LR by half if val_acc does not improve for 5 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6, verbose=True
    )
    use_amp = args.use_amp and device.type == "cuda"
    scaler  = GradScaler(enabled=use_amp)

    print(f"[CNN] lr={args.lr}  batch={args.batch_size}  "
          f"max_epochs={args.epochs}  patience={args.patience}  AMP={use_amp}\n")

    # --- Training loop ---
    best_val_acc = 0.0
    best_epoch   = 0
    patience_cnt = 0
    history      = []

    print(f"{'='*65}")
    print(f"  TRAINING CNN — max {args.epochs} epochs")
    print(f"{'='*65}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Forward + backward on training set
        tr_loss, tr_acc = run_train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, use_amp
        )

        # Evaluate on validation set (no gradients)
        va_loss, va_acc = run_eval_epoch(
            model, val_loader, criterion, device, use_amp
        )

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]["lr"]

        print(f"  Epoch {epoch:4d}/{args.epochs}"
              f"  tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.4f}"
              f"  va_loss={va_loss:.4f}  va_acc={va_acc:.4f}"
              f"  lr={lr_now:.1e}  [{elapsed:.1f}s]")

        # Save metrics to JSON
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

        # Step LR scheduler based on validation accuracy
        scheduler.step(va_acc)

        # Save best model
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            best_epoch   = epoch
            patience_cnt = 0
            save_checkpoint(model, optimizer, epoch, va_acc,
                            log_dir / "best_model.pt")
            print(f"  ✅  New best val_acc = {va_acc:.4f}  -> best_model.pt saved")
        else:
            patience_cnt += 1

        # Save periodic checkpoint every 10 epochs
        if epoch % 10 == 0:
            save_checkpoint(model, optimizer, epoch, va_acc,
                            log_dir / f"ckpt_epoch{epoch:04d}.pt")

        # Early stopping: stop if no improvement for patience epochs
        if patience_cnt >= args.patience:
            print(f"\n[CNN] Early stopping at epoch {epoch}"
                  f" — no improvement for {args.patience} epochs")
            break

    # --- Load best model and evaluate on test set ---
    print(f"\n[CNN] Loading best model (epoch {best_epoch}, "
          f"val_acc={best_val_acc:.4f})...")
    ckpt = torch.load(log_dir / "best_model.pt", map_location=device)
    model.load_state_dict(ckpt["model_state"])

    test_results = run_test_evaluation(model, test_loader, device, log_dir)

    # Save summary
    summary = {
        "model":         "cnn",
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
    print(f"  CNN TRAINING COMPLETE")
    print(f"  Best val accuracy : {best_val_acc:.4f}  (epoch {best_epoch})")
    print(f"  Test accuracy     : {test_results['test_accuracy']:.4f}")
    print(f"  Macro F1          : {test_results['macro_f1']:.4f}")
    print(f"  Checkpoint        : {log_dir / 'best_model.pt'}")
    print(f"{'='*65}\n")

    return best_val_acc


def main():
    ap = argparse.ArgumentParser(
        description="Train CNN gesture model — fully supervised"
    )
    ap.add_argument("-t", "--task",       default="gesture",
                    help="Experiment task label (used in log directory name)")
    ap.add_argument("-e", "--exp-name",   default="exp",
                    help="Experiment name (e.g. exp1)")
    ap.add_argument("-s", "--seed",       type=int,   default=42)
    ap.add_argument("-c", "--gpu",        type=int,   default=0,
                    help="GPU index (-1 for CPU)")
    ap.add_argument("-l", "--lr",         type=float, default=3e-4)
    ap.add_argument("--batch-size",       type=int,   default=32)
    ap.add_argument("--epochs",           type=int,   default=150)
    ap.add_argument("--patience",         type=int,   default=20)
    ap.add_argument("--num-workers",      type=int,   default=4)
    ap.add_argument("--use-amp",          action="store_true", default=True)
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
