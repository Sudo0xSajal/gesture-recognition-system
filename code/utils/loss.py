"""
code/utils/loss.py
==================
Supervised loss functions for CNN and LSTM training.

Three losses available — all fully supervised (every sample has a label):
  FocalLoss          — handles class imbalance (rare emergency gestures)
  LabelSmoothingLoss — prevents overconfidence (important for clinical safety)
  CrossEntropyLoss   — standard baseline

No semi-supervised loss, no contrastive loss, no DHC.
SVM uses scikit-learn's built-in loss internally — not defined here.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# 1. FOCAL LOSS
#    Best choice for this dataset: emergency gestures (Chest Hold, Both Hands
#    Raise) are naturally rarer than everyday gestures (Thumb Up, Open Palm).
#    Focal Loss down-weights easy well-classified samples so the model focuses
#    on hard misclassifications.
#    Reference: Lin et al. "Focal Loss for Dense Object Detection" ICCV 2017
# =============================================================================
class FocalLoss(nn.Module):
    """
    FL(pt) = -alpha * (1 - pt)^gamma * log(pt)

    Parameters
    ----------
    alpha  : float — overall loss scale (default 0.25)
    gamma  : float — focusing parameter (default 2.0; 0 = standard CE)
    reduction : str — 'mean' | 'sum' | 'none'
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0,
                 reduction: str = "mean"):
        super().__init__()
        self.alpha     = alpha
        self.gamma     = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : (N, C) — raw model output (before softmax)
        targets : (N,)   — integer class labels [0, C-1]
        """
        ce   = F.cross_entropy(logits, targets, reduction="none")
        pt   = torch.exp(-ce)
        loss = self.alpha * (1.0 - pt) ** self.gamma * ce

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


# =============================================================================
# 2. LABEL SMOOTHING LOSS
#    Prevents the model from being 100% confident on ambiguous frames.
#    Important for clinical safety: a model that says "100% sure: Chest Hold"
#    on a blurry / transitional frame could cause false emergency alarms.
# =============================================================================
class LabelSmoothingLoss(nn.Module):
    """
    Cross-entropy with soft targets.
    True class gets (1 - smoothing) probability;
    all other classes share smoothing / (C - 1).

    Parameters
    ----------
    num_classes : int   — 25
    smoothing   : float — epsilon (default 0.1)
    """

    def __init__(self, num_classes: int = 25, smoothing: float = 0.1):
        super().__init__()
        assert 0.0 <= smoothing < 1.0
        self.num_classes = num_classes
        self.smoothing   = smoothing
        self.confidence  = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        log_probs = F.log_softmax(logits, dim=-1)          # (N, C)
        with torch.no_grad():
            soft = torch.full_like(log_probs,
                                   self.smoothing / (self.num_classes - 1))
            soft.scatter_(1, targets.unsqueeze(1), self.confidence)
        return -(soft * log_probs).sum(dim=-1).mean()


# =============================================================================
# 3. LOSS FACTORY
# =============================================================================
def build_loss(cfg: dict) -> nn.Module:
    """
    Build the classification loss from TRAINING_CONFIG.

    Parameters
    ----------
    cfg : dict — TRAINING_CONFIG from config.py

    Returns
    -------
    nn.Module — FocalLoss | LabelSmoothingLoss | CrossEntropyLoss
    """
    name = cfg.get("loss", "focal")

    if name == "focal":
        return FocalLoss(
            alpha=cfg.get("focal_alpha", 0.25),
            gamma=cfg.get("focal_gamma", 2.0),
        )
    if name == "label_smooth":
        return LabelSmoothingLoss(
            num_classes=25,
            smoothing=cfg.get("label_smoothing", 0.1),
        )
    if name == "ce":
        return nn.CrossEntropyLoss(
            label_smoothing=cfg.get("label_smoothing", 0.0)
        )

    raise ValueError(f"Unknown loss '{name}'. Choose: focal | label_smooth | ce")
