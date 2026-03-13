"""
train_cnn.py — Complete CNN Training Pipeline
==============================================
This file integrates with your existing:
  - config.py (central configuration)
  - dataset.py (data loading)
  - cnn_model.py (model architectures)

Features:
  - Data augmentation pipeline
  - GestureCNN and MobileNetV2 training
  - Focal Loss for class imbalance
  - Full training loop with validation
  - Early stopping and checkpointing
  - Training curve visualization
  - Test evaluation with per-class metrics

Usage
-----
    python train_cnn.py                    # train with config defaults
    python train_cnn.py --backbone mobilenetv2 --lr 1e-3 --epochs 50
    python train_cnn.py --resume checkpoints/best_model.pth
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
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ── Import your existing modules ───────────────────────────────────── #
sys.path.insert(0, str(Path(__file__).parent))
from config import GestureConfig
from dataset import create_dataloaders, GestureDataset
from cnn_model import build_model, GestureCNN, MobileNetV2Transfer

cfg = GestureConfig(mode="train")


# =============================================================================
# PART 1 — LOSS FUNCTION (Focal Loss for class imbalance)
# =============================================================================

class FocalLoss(nn.Module):
    """
    Focal Loss — handles class imbalance in gesture datasets.
    
    Standard CrossEntropy treats all samples equally, so the model learns
    easy common gestures well but ignores rare ones.
    
    Focal Loss multiplies each sample's loss by (1 - confidence)^gamma:
    - If model is confident (pt high) → (1-pt)^gamma is small → less weight
    - If model is wrong (pt low) → (1-pt)^gamma is large → more weight
    
    This forces the model to focus on the hard/rare gestures.
    
    Args:
        alpha: class weight (can be tensor for per-class weights)
        gamma: focusing parameter (default 2.0 works well)
        reduction: 'mean' or 'sum'
    """
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        logits:  (N, num_classes) raw model output
        targets: (N,) integer class labels
        """
        # Standard cross entropy per sample
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # Convert to probability: pt = exp(-ce_loss)
        pt = torch.exp(-ce_loss)
        
        # Apply focal weighting
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# =============================================================================
# PART 2 — TRAINING LOOP UTILITIES
# =============================================================================

def set_seed(seed: int):
    """Fix all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Seed] Random seed set to {seed}")


def get_scheduler(optimizer, scheduler_type: str, **kwargs):
    """Factory function for learning rate schedulers."""
    if scheduler_type == "cosine":
        return CosineAnnealingLR(optimizer, T_max=kwargs.get('T_max', 50))
    elif scheduler_type == "step":
        return StepLR(optimizer, step_size=kwargs.get('step_size', 10), 
                      gamma=kwargs.get('gamma', 0.1))
    elif scheduler_type == "plateau":
        return ReduceLROnPlateau(optimizer, mode='max', factor=0.5, 
                                  patience=5, min_lr=1e-6)
    else:
        return None


def train_one_epoch(model, loader, criterion, optimizer, 
                    scaler, device, use_amp) -> tuple:
    """
    One full pass through the training set.
    
    Returns:
        avg_loss: average loss for the epoch
        accuracy: classification accuracy for the epoch
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = loader  # tqdm can be added here if desired
    
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Forward pass with mixed precision
        with autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        
        # Backward pass
        scaler.scale(loss).backward()
        
        # Gradient clipping (prevents exploding gradients)
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        scaler.step(optimizer)
        scaler.update()
        
        # Track metrics
        total_loss += loss.item()
        predictions = logits.detach().argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device, use_amp) -> tuple:
    """
    One full pass through validation set — no gradient computation.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        with autocast(enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)
        
        total_loss += loss.item()
        predictions = logits.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy


def save_checkpoint(model, optimizer, scheduler, epoch, val_acc, 
                    val_loss, path: Path, is_best: bool = False):
    """Save complete training state."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'val_acc': val_acc,
        'val_loss': val_loss,
    }, path)
    if is_best:
        print(f"  ✅ Best model saved → {path}")


def plot_training_history(history: list, best_epoch: int, save_path: Path):
    """
    Plot training and validation accuracy/loss curves.
    """
    epochs = [h['epoch'] for h in history]
    train_acc = [h['train_acc'] for h in history]
    val_acc = [h['val_acc'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy plot
    ax1.plot(epochs, train_acc, 'b-', label='Train Accuracy', linewidth=2)
    ax1.plot(epochs, val_acc, 'g-', label='Validation Accuracy', linewidth=2)
    ax1.axvline(best_epoch, color='r', linestyle='--', alpha=0.7, 
                label=f'Best Epoch ({best_epoch})')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)
    
    # Loss plot
    ax2.plot(epochs, train_loss, 'b-', label='Train Loss', linewidth=2)
    ax2.plot(epochs, val_loss, 'g-', label='Validation Loss', linewidth=2)
    ax2.axvline(best_epoch, color='r', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('Model Loss')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'Training History - {cfg.backbone}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close()
    print(f"[Plot] Training curves saved → {save_path}")


@torch.no_grad()
def evaluate_test(model, test_loader, device, log_dir: Path, class_names: dict):
    """
    Comprehensive test evaluation with per-class metrics.
    """
    model.eval()
    all_preds = []
    all_labels = []
    
    # Collect all predictions
    for images, labels in test_loader:
        images = images.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
    
    # Calculate metrics
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    # Get class names in order
    target_names = [class_names.get(i, f"Class_{i}") for i in range(cfg.num_classes)]
    
    # Generate classification report
    report_str = classification_report(
        all_labels, all_preds, 
        target_names=target_names,
        zero_division=0
    )
    
    report_dict = classification_report(
        all_labels, all_preds,
        target_names=target_names,
        output_dict=True,
        zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix - Test Set')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(log_dir / 'confusion_matrix.png', dpi=120)
    plt.close()
    
    # Print results
    print("\n" + "="*70)
    print(f"  TEST SET EVALUATION")
    print("="*70)
    print(f"  Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Macro F1-Score: {report_dict['macro avg']['f1-score']:.4f}")
    print(f"  Weighted F1-Score: {report_dict['weighted avg']['f1-score']:.4f}")
    print("\n  Per-Class Performance:")
    print("-"*70)
    
    per_class_results = {}
    for i, name in enumerate(target_names):
        recall = report_dict.get(name, {}).get('recall', 0)
        precision = report_dict.get(name, {}).get('precision', 0)
        f1 = report_dict.get(name, {}).get('f1-score', 0)
        support = report_dict.get(name, {}).get('support', 0)
        print(f"    {name:<12} | prec: {precision:.3f} | recall: {recall:.3f} | "
              f"f1: {f1:.3f} | support: {int(support):3d}")
        per_class_results[name] = {
            'precision': round(precision, 4),
            'recall': round(recall, 4),
            'f1-score': round(f1, 4),
            'support': int(support)
        }
    
    results = {
        'test_accuracy': round(accuracy, 6),
        'macro_f1': round(report_dict['macro avg']['f1-score'], 4),
        'weighted_f1': round(report_dict['weighted avg']['f1-score'], 4),
        'per_class': per_class_results
    }
    
    # Save results
    with open(log_dir / 'test_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save full report
    with open(log_dir / 'classification_report.txt', 'w') as f:
        f.write(report_str)
    
    return results


# =============================================================================
# PART 3 — MAIN TRAINING FUNCTION
# =============================================================================

def create_experiment_dir(base_dir: str, exp_name: str, backbone: str, seed: int) -> Path:
    """Create and return experiment directory."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(base_dir) / f"{exp_name}_{backbone}_s{seed}_{timestamp}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def train(args):
    """
    Main training pipeline.
    """
    # Setup
    set_seed(args.seed)
    
    # Device configuration
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        print(f"\n[Device] Using GPU {args.gpu}: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("\n[Device] Using CPU")
    
    # Create experiment directory
    exp_dir = create_experiment_dir(
        args.log_dir, args.exp_name, args.backbone, args.seed
    )
    print(f"[Log] Experiment directory: {exp_dir}")
    
    # Save configuration
    with open(exp_dir / 'config.json', 'w') as f:
        config_dict = {k: str(v) for k, v in cfg.__dict__.items() 
                      if not k.startswith('_')}
        json.dump(config_dict, f, indent=2)
    
    # ── 1. Load Data ───────────────────────────────────────────── #
    print("\n[1/5] Loading datasets...")
    loaders = create_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        use_weighted_sampler=args.use_weighted_sampler
    )
    
    train_loader = loaders['train']
    val_loader = loaders['val']
    test_loader = loaders['test']
    
    # ── 2. Build Model ─────────────────────────────────────────── #
    print(f"\n[2/5] Building model: {args.backbone}")
    model = build_model(
        backbone=args.backbone,
        num_classes=cfg.num_classes,
        dropout=args.dropout,
        pretrained=args.pretrained
    ).to(device)
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"    Total parameters: {total_params:,}")
    print(f"    Trainable parameters: {trainable_params:,}")
    
    # ── 3. Loss, Optimizer, Scheduler ──────────────────────────── #
    print("\n[3/5] Setting up training components...")
    
    # Loss function (Focal Loss or CrossEntropy)
    if args.use_focal_loss:
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
        print(f"    Using Focal Loss (alpha={args.focal_alpha}, gamma={args.focal_gamma})")
    else:
        criterion = nn.CrossEntropyLoss()
        print("    Using CrossEntropy Loss")
    
    # Optimizer
    if args.optimizer.lower() == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=args.lr, 
            weight_decay=args.weight_decay
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=args.lr, 
            weight_decay=args.weight_decay
        )
    
    # Scheduler
    scheduler = get_scheduler(
        optimizer, 
        args.scheduler,
        T_max=args.epochs,
        step_size=10,
        gamma=0.1
    )
    
    # Mixed precision training
    use_amp = args.use_amp and device.type == 'cuda'
    scaler = GradScaler(enabled=use_amp)
    print(f"    Mixed precision: {'ON' if use_amp else 'OFF'}")
    
    # ── 4. Resume from checkpoint if specified ─────────────────── #
    start_epoch = 1
    best_val_acc = 0.0
    best_val_loss = float('inf')
    history = []
    
    if args.resume:
        if Path(args.resume).exists():
            print(f"\n[Resume] Loading checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler and checkpoint.get('scheduler_state_dict'):
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_acc = checkpoint.get('val_acc', 0.0)
            best_val_loss = checkpoint.get('val_loss', float('inf'))
            print(f"    Resumed from epoch {checkpoint['epoch']}")
        else:
            print(f"    Checkpoint not found: {args.resume}, starting fresh")
    
    # ── 5. Training Loop ───────────────────────────────────────── #
    print(f"\n[4/5] Starting training for {args.epochs} epochs...")
    print("="*70)
    
    patience_counter = 0
    
    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start = time.time()
        
        # Training
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, 
            scaler, device, use_amp
        )
        
        # Validation
        val_loss, val_acc = validate_one_epoch(
            model, val_loader, criterion, device, use_amp
        )
        
        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print progress
        print(f"  Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.2e} | Time: {epoch_time:.1f}s")
        
        # Update history
        history.append({
            'epoch': epoch,
            'train_loss': round(train_loss, 6),
            'train_acc': round(train_acc, 6),
            'val_loss': round(val_loss, 6),
            'val_acc': round(val_acc, 6),
            'lr': current_lr
        })
        
        # Save metrics
        with open(exp_dir / 'metrics.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        # Scheduler step
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_acc)
            else:
                scheduler.step()
        
        # Save best model
        is_best = False
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_epoch = epoch
            patience_counter = 0
            is_best = True
            
            save_checkpoint(
                model, optimizer, scheduler, epoch, 
                val_acc, val_loss, exp_dir / 'best_model.pth', 
                is_best=True
            )
        else:
            patience_counter += 1
        
        # Save checkpoint periodically
        if epoch % args.save_every == 0:
            save_checkpoint(
                model, optimizer, scheduler, epoch,
                val_acc, val_loss, exp_dir / f'checkpoint_epoch{epoch:04d}.pth'
            )
        
        # Early stopping
        if patience_counter >= args.patience:
            print(f"\n[Early Stopping] No improvement for {args.patience} epochs")
            break
    
    print("="*70)
    print(f"[Training Complete] Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
    
    # ── 6. Plot training curves ────────────────────────────────── #
    plot_training_history(history, best_epoch, exp_dir / 'training_history.png')
    
    # ── 7. Final test evaluation ───────────────────────────────── #
    print("\n[5/5] Evaluating best model on test set...")
    
    # Load best model
    best_model_path = exp_dir / 'best_model.pth'
    if best_model_path.exists():
        checkpoint = torch.load(best_model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"    Loaded best model from epoch {checkpoint['epoch']}")
    
    # Evaluate
    test_results = evaluate_test(
        model, test_loader, device, exp_dir, cfg.class_names or {}
    )
    
    # Save summary
    summary = {
        'experiment': args.exp_name,
        'backbone': args.backbone,
        'seed': args.seed,
        'best_val_acc': round(best_val_acc, 6),
        'best_epoch': best_epoch,
        'test_accuracy': test_results['test_accuracy'],
        'macro_f1': test_results['macro_f1'],
        'weighted_f1': test_results['weighted_f1'],
        'total_epochs': len(history),
        'config': {
            'batch_size': args.batch_size,
            'lr': args.lr,
            'optimizer': args.optimizer,
            'scheduler': args.scheduler,
            'focal_loss': args.use_focal_loss
        }
    }
    
    with open(exp_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Final output
    print("\n" + "="*70)
    print("  TRAINING COMPLETE - SUMMARY")
    print("="*70)
    print(f"  Experiment: {args.exp_name}")
    print(f"  Backbone: {args.backbone}")
    print(f"  Best validation accuracy: {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")
    print(f"  Test accuracy: {test_results['test_accuracy']:.4f} ({test_results['test_accuracy']*100:.2f}%)")
    print(f"  Test Macro F1: {test_results['macro_f1']:.4f}")
    print(f"\n  Results saved to: {exp_dir}")
    print("="*70)
    
    return best_val_acc, test_results['test_accuracy']


# =============================================================================
# PART 4 — COMMAND LINE INTERFACE
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Train CNN for Hand Gesture Recognition")
    
    # Experiment settings
    parser.add_argument('--exp-name', type=str, default='gesture_cnn',
                        help='Experiment name for logging')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--log-dir', type=str, default='./experiments',
                        help='Base directory for experiment logs')
    
    # Model settings
    parser.add_argument('--backbone', type=str, default=cfg.backbone,
                        choices=['mobilenetv2', 'gesturecnn'],
                        help='Model architecture')
    parser.add_argument('--pretrained', action='store_true', default=cfg.pretrained,
                        help='Use pretrained weights (for MobileNetV2)')
    parser.add_argument('--dropout', type=float, default=cfg.dropout_rate,
                        help='Dropout rate in classifier')
    
    # Training settings
    parser.add_argument('--batch-size', type=int, default=cfg.batch_size,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=cfg.num_epochs,
                        help='Maximum number of epochs')
    parser.add_argument('--lr', type=float, default=cfg.learning_rate,
                        help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=cfg.weight_decay,
                        help='Weight decay for optimizer')
    parser.add_argument('--optimizer', type=str, default='adamw',
                        choices=['adam', 'adamw'],
                        help='Optimizer type')
    parser.add_argument('--scheduler', type=str, default=cfg.scheduler,
                        choices=['cosine', 'step', 'plateau', 'none'],
                        help='Learning rate scheduler')
    
    # Loss function
    parser.add_argument('--use-focal-loss', action='store_true', default=True,
                        help='Use Focal Loss instead of CrossEntropy')
    parser.add_argument('--focal-alpha', type=float, default=0.25,
                        help='Alpha parameter for Focal Loss')
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                        help='Gamma parameter for Focal Loss')
    
    # Data loading
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--use-weighted-sampler', action='store_true', default=True,
                        help='Use weighted sampler for class balance')
    
    # Training control
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID to use (-1 for CPU)')
    parser.add_argument('--patience', type=int, default=cfg.early_stop_patience,
                        help='Early stopping patience')
    parser.add_argument('--save-every', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--use-amp', action='store_true', default=True,
                        help='Use automatic mixed precision')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    args = parser.parse_args()
    
    # Train
    train(args)


if __name__ == '__main__':
    main()