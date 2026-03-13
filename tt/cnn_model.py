"""
cnn_model.py — Gesture Recognition CNN Models
==============================================
Two model variants — both wired to config.py (GestureConfig):

  1. GestureCNN         — custom CNN: residual encoder + self-attention + GAP
  2. MobileNetV2Transfer — pretrained ImageNet backbone (best for small datasets)

Your dataset has 10 classes (0–9):
  0=palm  1=l  2=fist  3=fist_moved  4=thumb
  5=index 6=ok 7=palm_moved  8=c  9=down

Architecture (GestureCNN)
--------------------------
  Input (3, 224, 224)
    → Stage 1 : Conv-BN-ReLU          3→32 ch   224×224
    → Stage 2 : ResidualBlock stride2  32→64 ch   112×112
    → Stage 3 : ResidualBlock stride2  64→128 ch   56×56
    → Stage 4 : ResidualBlock stride2 128→256 ch   28×28
    → Self-Attention block             256 ch
    → Stage 5 : ResidualBlock stride2 256→512 ch   14×14
    → Global Average Pool              512-dim
    → Embedding projection  512→256
    → Classifier head       256→cfg.num_classes (10)

Usage
-----
    from cnn_model import build_model

    # MobileNetV2 (recommended — pretrained, fast)
    model = build_model("mobilenetv2")

    # Custom CNN from scratch
    model = build_model("gesturecnn")

    # Forward pass
    x      = torch.randn(4, 3, 224, 224)
    logits = model(x)           # (4, 10)

    # Prediction with class name
    result = model.predict(x)
    print(result["class_name"])  # e.g. ["fist", "palm", ...]

    # Sanity check (run directly):
    python cnn_model.py
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# ── bring in config ──────────────────────────────────────────────────── #
sys.path.insert(0, str(Path(__file__).parent))
from config import GestureConfig

cfg = GestureConfig(mode="train")


# ════════════════════════════════════════════════════════════════════════
# BUILDING BLOCKS
# ════════════════════════════════════════════════════════════════════════

class _ConvBNReLU(nn.Sequential):
    """Conv2d → BatchNorm2d → ReLU  (standard building block)."""
    def __init__(self, in_ch: int, out_ch: int,
                 kernel: int = 3, stride: int = 1, padding: int = 1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel,
                      stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class _ResidualBlock(nn.Module):
    """
    Two-conv residual block with optional channel projection on the skip path.
    Prevents vanishing gradients in deeper encoder stages.
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = _ConvBNReLU(in_ch, out_ch, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        # Skip connection: project channels/stride if they differ
        self.skip = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            if (in_ch != out_ch or stride != 1)
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv2(self.conv1(x)) + self.skip(x))


class _SelfAttention(nn.Module):
    """
    Non-local self-attention (Wang et al. 2018).
    Lets the model relate distant spatial positions — captures whole-hand
    context, e.g. distance between fingertips and palm centre.

    γ (gamma) starts at 0 so the block acts as identity at init and
    activates gradually — safe to insert anywhere without destabilising
    early training.
    """
    def __init__(self, in_ch: int):
        super().__init__()
        mid        = max(in_ch // 8, 1)
        self.query = nn.Conv2d(in_ch, mid, 1)
        self.key   = nn.Conv2d(in_ch, mid, 1)
        self.value = nn.Conv2d(in_ch, in_ch, 1)
        self.gamma = nn.Parameter(torch.zeros(1))   # learnable residual scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.size()
        Q = self.query(x).view(N, -1, H * W).permute(0, 2, 1)  # (N, HW, C//8)
        K = self.key(x).view(N, -1, H * W)                      # (N, C//8, HW)
        A = F.softmax(torch.bmm(Q, K), dim=-1)                  # (N, HW, HW)
        V = self.value(x).view(N, -1, H * W)                    # (N, C, HW)
        out = torch.bmm(V, A.permute(0, 2, 1)).view(N, C, H, W)
        return x + self.gamma * out


# ════════════════════════════════════════════════════════════════════════
# 1. GESTURE CNN  (custom architecture)
# ════════════════════════════════════════════════════════════════════════

class GestureCNN(nn.Module):
    """
    Custom CNN for hand gesture image classification.

    5-stage residual encoder with self-attention, Global Average Pooling,
    a 256-dim embedding projection, and a final classifier head.

    Parameters
    ----------
    num_classes  : int   — number of gesture classes (default: cfg.num_classes = 10)
    base_filters : int   — Stage-1 filter count; doubles each stage (default: 32)
    dropout      : float — dropout rate in classifier head (default: cfg.dropout_rate)
    """

    def __init__(
        self,
        num_classes:  int   = None,
        base_filters: int   = 32,
        dropout:      float = None,
    ):
        super().__init__()
        n_cls = num_classes or cfg.num_classes          # 10
        f     = base_filters                            # 32
        dr    = dropout if dropout is not None else cfg.dropout_rate   # 0.3

        # ── Encoder ──────────────────────────────────────────────── #
        self.stage1    = _ConvBNReLU(3,     f,      stride=1)  # 224×224, 32ch
        self.stage2    = _ResidualBlock(f,  f*2,    stride=2)  # 112×112, 64ch
        self.stage3    = _ResidualBlock(f*2, f*4,   stride=2)  #  56×56, 128ch
        self.stage4    = _ResidualBlock(f*4, f*8,   stride=2)  #  28×28, 256ch
        self.attention = _SelfAttention(f*8)                   #  28×28, 256ch
        self.stage5    = _ResidualBlock(f*8, f*16,  stride=2)  #  14×14, 512ch

        # ── Global Average Pool ───────────────────────────────────── #
        # Collapses spatial dims: (N, 512, 14, 14) → (N, 512)
        self.gap = nn.AdaptiveAvgPool2d(1)

        # ── Embedding projection: 512 → 256 ─────────────────────── #
        # Compact representation used for classification and fusion
        self.embed_proj = nn.Sequential(
            nn.Linear(f * 16, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.embedding_dim = 256

        # ── Classifier head: 256 → num_classes ──────────────────── #
        self.classifier = nn.Sequential(
            nn.Dropout(dr),
            nn.Linear(256, n_cls),
        )

        # Store for predict()
        self.num_classes = n_cls
        self._init_weights()

    # ---------------------------------------------------------------- #
    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ---------------------------------------------------------------- #
    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Shared encoder → 256-dim L2-normalised embedding."""
        x   = self.stage1(x)
        x   = self.stage2(x)
        x   = self.stage3(x)
        x   = self.stage4(x)
        x   = self.attention(x)
        x   = self.stage5(x)
        gap = self.gap(x).flatten(1)    # (N, 512)
        emb = self.embed_proj(gap)      # (N, 256)
        return emb

    # ---------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor (N, 3, 224, 224)  — normalised image batch

        Returns
        -------
        logits : Tensor (N, num_classes)
        """
        return self.classifier(self._encode(x))

    # ---------------------------------------------------------------- #
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised 256-dim embedding (for visualisation / fusion)."""
        with torch.no_grad():
            return F.normalize(self._encode(x), dim=1)

    # ---------------------------------------------------------------- #
    def predict(self, x: torch.Tensor) -> dict:
        """
        Run inference on a single image or batch.

        Parameters
        ----------
        x : Tensor (3, 224, 224) or (N, 3, 224, 224)

        Returns
        -------
        dict with keys:
            logits      — raw scores         (N, num_classes)
            probs       — softmax probs      (N, num_classes)
            class_idx   — predicted index    (N,)
            confidence  — max probability    (N,)
            class_name  — human-readable label list, e.g. ["fist", "palm"]
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs  = F.softmax(logits, dim=1)
        conf, idx = probs.max(dim=1)

        # Map index → human-readable name from config
        names = [
            (cfg.class_names or {}).get(i.item(), f"class_{i.item()}")
            for i in idx
        ]
        return {
            "logits"    : logits,
            "probs"     : probs,
            "class_idx" : idx,
            "confidence": conf,
            "class_name": names,
        }


# ════════════════════════════════════════════════════════════════════════
# 2. MOBILENETV2 TRANSFER LEARNING  (recommended for small datasets)
# ════════════════════════════════════════════════════════════════════════

class MobileNetV2Transfer(nn.Module):
    """
    MobileNetV2 pretrained on ImageNet with a custom gesture head.

    Two-phase fine-tuning:
      Phase 1 — backbone frozen, train head only  (fast, avoids overfitting)
      Phase 2 — unfreeze last N layers, fine-tune  (higher accuracy)

    Call model.unfreeze(last_n_layers=10) to switch from Phase 1 → 2.

    Parameters
    ----------
    num_classes      : int   — default cfg.num_classes (10)
    dropout          : float — default cfg.dropout_rate (0.3)
    freeze_backbone  : bool  — True = Phase 1 (frozen backbone)
    """

    def __init__(
        self,
        num_classes:     int   = None,
        dropout:         float = None,
        freeze_backbone: bool  = True,
    ):
        super().__init__()
        n_cls = num_classes or cfg.num_classes          # 10
        dr    = dropout if dropout is not None else cfg.dropout_rate   # 0.3

        # ── Pretrained backbone ───────────────────────────────────── #
        backbone = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
        )
        self.features = backbone.features   # output: (N, 1280, 7, 7) for 224×224

        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

        # ── Custom head ───────────────────────────────────────────── #
        # AdaptiveAvgPool → Flatten → Linear(1280→256) → BN → ReLU
        # → Dropout → Linear(256→num_classes)
        self.pool    = nn.AdaptiveAvgPool2d(1)   # (N, 1280, 7, 7) → (N, 1280, 1, 1)
        self.flatten = nn.Flatten()              # → (N, 1280)

        self.embed_proj = nn.Sequential(
            nn.Linear(1280, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.embedding_dim = 256

        self.classifier = nn.Sequential(
            nn.Dropout(dr),
            nn.Linear(256, n_cls),
        )

        self.num_classes = n_cls

    # ---------------------------------------------------------------- #
    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Backbone + pool + flatten + embed_proj → (N, 256)."""
        feats = self.features(x)                   # (N, 1280, 7, 7)
        pooled = self.flatten(self.pool(feats))    # (N, 1280)
        return self.embed_proj(pooled)             # (N, 256)

    # ---------------------------------------------------------------- #
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor (N, 3, 224, 224)

        Returns
        -------
        logits : Tensor (N, num_classes)
        """
        return self.classifier(self._encode(x))

    # ---------------------------------------------------------------- #
    def unfreeze(self, last_n_layers: int = 10) -> None:
        """
        Phase 2 fine-tuning: unfreeze the last N backbone layers.
        Call this after Phase 1 converges (typically ~10 epochs).
        Also reduce learning rate by 10× when switching phases.

        Example
        -------
            model.unfreeze(last_n_layers=10)
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=1e-4
            )
        """
        layers = list(self.features.children())
        for layer in layers[-last_n_layers:]:
            for p in layer.parameters():
                p.requires_grad = True
        unfrozen = sum(p.requires_grad for p in self.features.parameters())
        print(f"[MobileNetV2] Unfrozen last {last_n_layers} layers "
              f"({unfrozen} backbone params now trainable)")

    # ---------------------------------------------------------------- #
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised 256-dim embedding."""
        with torch.no_grad():
            return F.normalize(self._encode(x), dim=1)

    # ---------------------------------------------------------------- #
    def predict(self, x: torch.Tensor) -> dict:
        """
        Run inference on a single image or batch.

        Parameters
        ----------
        x : Tensor (3, 224, 224) or (N, 3, 224, 224)

        Returns
        -------
        dict with keys:
            logits, probs, class_idx, confidence, class_name
        """
        if x.dim() == 3:
            x = x.unsqueeze(0)
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs  = F.softmax(logits, dim=1)
        conf, idx = probs.max(dim=1)

        names = [
            (cfg.class_names or {}).get(i.item(), f"class_{i.item()}")
            for i in idx
        ]
        return {
            "logits"    : logits,
            "probs"     : probs,
            "class_idx" : idx,
            "confidence": conf,
            "class_name": names,
        }


# ════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTION  — used by train.py
# ════════════════════════════════════════════════════════════════════════

def build_model(
    backbone:    str   = None,
    num_classes: int   = None,
    dropout:     float = None,
    pretrained:  bool  = None,
) -> nn.Module:
    """
    Build and return a model from config or explicit arguments.

    Parameters
    ----------
    backbone    : "mobilenetv2" | "gesturecnn"  (default: cfg.backbone)
    num_classes : int  (default: cfg.num_classes = 10)
    dropout     : float  (default: cfg.dropout_rate = 0.3)
    pretrained  : bool   (default: cfg.pretrained = True)
                  Only applies to MobileNetV2.

    Returns
    -------
    nn.Module — the model, NOT yet moved to device.
    Call model.to(cfg.device) after build_model().

    Example
    -------
        model = build_model().to(cfg.device)
    """
    arch       = (backbone    or cfg.backbone).lower()
    n_cls      = num_classes  or cfg.num_classes
    dr         = dropout      if dropout is not None else cfg.dropout_rate
    use_pre    = pretrained   if pretrained is not None else cfg.pretrained

    if arch == "mobilenetv2":
        model = MobileNetV2Transfer(
            num_classes    = n_cls,
            dropout        = dr,
            freeze_backbone= use_pre,   # freeze when using pretrained weights
        )
        print(f"[build_model] MobileNetV2Transfer  "
              f"classes={n_cls}  dropout={dr}  frozen={use_pre}")

    elif arch == "gesturecnn":
        model = GestureCNN(
            num_classes  = n_cls,
            base_filters = 32,
            dropout      = dr,
        )
        print(f"[build_model] GestureCNN  "
              f"classes={n_cls}  dropout={dr}")

    else:
        raise ValueError(
            f"Unknown backbone: '{arch}'\n"
            f"Choose 'mobilenetv2' or 'gesturecnn'\n"
            f"Or set cfg.backbone in config.py"
        )

    total_params     = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"           Total params    : {total_params:,}")
    print(f"           Trainable params: {trainable_params:,}")
    return model


# ════════════════════════════════════════════════════════════════════════
# SANITY CHECK  — run as: python cnn_model.py
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 55)
    print("  CNN MODEL SANITY CHECK")
    print("=" * 55)

    dummy = torch.randn(4, 3, 224, 224)

    for arch in ["mobilenetv2", "gesturecnn"]:
        print(f"\n── {arch} ──")
        model  = build_model(arch)
        logits = model(dummy)
        assert logits.shape == (4, cfg.num_classes), \
            f"Expected (4, {cfg.num_classes}), got {logits.shape}"
        print(f"  forward()    → {logits.shape}  ✓")

        emb = model.get_embeddings(dummy)
        assert emb.shape == (4, 256), \
            f"Expected (4, 256), got {emb.shape}"
        print(f"  embeddings   → {emb.shape}  ✓")

        result = model.predict(dummy)
        print(f"  predict()    → class_name={result['class_name']}  ✓")
        print(f"  confidence   → {result['confidence'].tolist()}")

    print("\n✅  All checks passed — ready for train.py")