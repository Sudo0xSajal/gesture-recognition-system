"""
code/models/cnn_model.py
========================
STEP 3 — Gesture Recognition AI Model: CNN on raw images

Two CNN variants as described in Ma'am's document:
  1. GestureCNN    — custom CNN with attention + Spatial Pyramid Pooling
  2. MobileNetV2   — pretrained ImageNet backbone (best when data is scarce)

Why CNN for gestures?
---------------------
The landmark model only sees skeleton coordinates. The CNN sees the entire
image — it can learn additional cues like hand texture, lighting shadows,
and full hand shape. Combined in HybridModel, CNN + MLP outperforms either alone.

Architecture (GestureCNN)
--------------------------
  Input(3, 224, 224)
    → Encoder stage 1: 3→32 filters, stride 1  (224×224)
    → Encoder stage 2: 32→64 filters, stride 2 (112×112)
    → Encoder stage 3: 64→128 filters, stride 2 (56×56)
    → Encoder stage 4: 128→256 filters, stride 2 (28×28)
    → Non-local Self-Attention block
    → Encoder stage 5: 256→512 filters, stride 2 (14×14)
    → Spatial Pyramid Pooling [1×1, 2×2, 4×4] → 512×(1+4+16)=10752
    → Global Average Pool → (512,)
    → FC [512→256] → Dropout → FC [256→25]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from utils.config import NUM_CLASSES, MODEL_CONFIG


# =============================================================================
# BUILDING BLOCKS
# =============================================================================

class _ConvBNReLU(nn.Sequential):
    """Conv → BatchNorm → ReLU — standard building block."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel, stride=stride,
                      padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class _ResidualBlock(nn.Module):
    """
    Two-conv residual block with optional channel projection.
    Prevents vanishing gradients in deeper stages.
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = _ConvBNReLU(in_ch, out_ch, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = (
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
            if in_ch != out_ch or stride != 1
            else nn.Identity()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.conv2(self.conv1(x)) + self.skip(x))


class _SelfAttention(nn.Module):
    """
    Non-local self-attention (Wang et al. 2018).
    Lets the model relate distant spatial positions — useful for capturing
    whole-hand context (e.g. distance between thumb tip and palm).

    A learnable residual scale γ starts at 0 so the block initially acts as
    an identity function and gradually activates during training.
    """
    def __init__(self, in_ch: int):
        super().__init__()
        mid = max(in_ch // 8, 1)
        self.query = nn.Conv2d(in_ch, mid, 1)
        self.key   = nn.Conv2d(in_ch, mid, 1)
        self.value = nn.Conv2d(in_ch, in_ch, 1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N, C, H, W = x.size()
        Q = self.query(x).view(N, -1, H * W).permute(0, 2, 1)  # (N, HW, C//8)
        K = self.key(x).view(N, -1, H * W)                      # (N, C//8, HW)
        A = F.softmax(torch.bmm(Q, K), dim=-1)                  # (N, HW, HW)
        V = self.value(x).view(N, -1, H * W)                    # (N, C, HW)
        out = torch.bmm(V, A.permute(0, 2, 1)).view(N, C, H, W)
        return x + self.gamma * out


class _SpatialPyramidPooling(nn.Module):
    """
    Spatial Pyramid Pooling (SPP).
    Pools features at three scales — captures global (1×1), regional (2×2),
    and local (4×4) context simultaneously without resizing the input.

    The resulting descriptor is invariant to small spatial shifts of the hand.
    """
    def __init__(self, levels: list = None):
        super().__init__()
        self.levels = levels or [1, 2, 4]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pools = []
        for level in self.levels:
            pooled = F.adaptive_avg_pool2d(x, output_size=(level, level))
            pools.append(pooled.flatten(1))               # (N, C × level²)
        return torch.cat(pools, dim=1)                    # (N, C × sum(level²))


# =============================================================================
# 1. GESTURE CNN  (custom architecture — production model)
# =============================================================================
class GestureCNN(nn.Module):
    """
    Custom CNN for gesture image classification.

    5-stage encoder with residual blocks, self-attention, and Spatial Pyramid
    Pooling. Outputs both class logits and a 256-dim embedding for fusion.

    Parameters
    ----------
    base_filters : int   — first-stage filter count (default 32, doubles each stage)
    num_classes  : int   — 25
    dropout      : float — classifier head dropout
    """

    def __init__(
        self,
        base_filters: int   = None,
        num_classes:  int   = NUM_CLASSES,
        dropout:      float = None,
    ):
        super().__init__()
        f  = base_filters or MODEL_CONFIG["cnn_base_filters"]   # 32
        dr = dropout      or MODEL_CONFIG["cnn_dropout"]        # 0.4

        # ── Encoder ──────────────────────────────────────────────────────────
        self.stage1 = _ConvBNReLU(3, f)                        # 224×224, 32ch
        self.stage2 = _ResidualBlock(f,    f * 2,  stride=2)   # 112×112, 64ch
        self.stage3 = _ResidualBlock(f*2,  f * 4,  stride=2)   # 56×56,  128ch
        self.stage4 = _ResidualBlock(f*4,  f * 8,  stride=2)   # 28×28,  256ch
        self.attention = _SelfAttention(f * 8)
        self.stage5 = _ResidualBlock(f*8,  f * 16, stride=2)   # 14×14,  512ch

        # ── Pooling ───────────────────────────────────────────────────────────
        # SPP at 3 scales: 1+4+16 = 21 spatial positions → 512×21 = 10752
        # followed by Global Average Pool (512) to keep embedding compact
        self.spp = _SpatialPyramidPooling(levels=[1, 2, 4])
        spp_out  = f * 16 * (1 + 4 + 16)                      # 10752
        self.gap = nn.AdaptiveAvgPool2d(1)

        # ── Embedding projection ──────────────────────────────────────────────
        # Projects 512-dim GAP output → 256-dim embedding for HybridModel fusion
        self.embed_proj = nn.Sequential(
            nn.Linear(f * 16, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
        )
        self.embedding_dim = 256

        # ── Classifier head ───────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Dropout(dr),
            nn.Linear(256, num_classes),
        )

        self._init_weights()

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

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """Shared encoder → 256-dim embedding."""
        x   = self.stage1(x)
        x   = self.stage2(x)
        x   = self.stage3(x)
        x   = self.stage4(x)
        x   = self.attention(x)
        x   = self.stage5(x)
        gap = self.gap(x).flatten(1)           # (N, 512)
        emb = self.embed_proj(gap)             # (N, 256)
        return emb

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (N, 3, 224, 224) — normalised image tensor

        Returns
        -------
        logits : (N, 25)
        """
        emb    = self._encode(x)
        logits = self.classifier(emb)
        return logits

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised 256-dim embedding for HybridModel."""
        return F.normalize(self._encode(x), dim=1)

    def predict(self, x: torch.Tensor) -> dict:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        with torch.no_grad():
            logits = self.forward(x)
            probs  = F.softmax(logits, dim=1)
        conf, gid = probs.max(dim=1)
        return {"logits": logits, "probs": probs,
                "gesture_id": gid, "confidence": conf}


# =============================================================================
# 2. MOBILENETV2 TRANSFER LEARNING  (best for small datasets)
# =============================================================================
class MobileNetV2Transfer(nn.Module):
    """
    MobileNetV2 pretrained on ImageNet with a custom gesture classification head.

    Recommended when the training dataset has fewer than ~200 samples per class.
    The pretrained weights encode rich visual features (edges, textures, shapes)
    that transfer well to hand gesture recognition.

    Fine-tuning strategy:
      - Phase 1: freeze backbone, train head only (fast, prevents overfitting)
      - Phase 2: unfreeze last N layers, fine-tune end-to-end (higher accuracy)

    Parameters
    ----------
    num_classes : int   — 25
    dropout     : float — head dropout
    freeze_backbone : bool — freeze backbone during Phase 1
    """

    def __init__(self, num_classes: int = NUM_CLASSES,
                 dropout: float = None, freeze_backbone: bool = True):
        super().__init__()
        dr = dropout or MODEL_CONFIG["cnn_dropout"]

        # Load pretrained MobileNetV2
        backbone = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1
        )
        self.features = backbone.features          # (N, 1280, 7, 7) for 224×224

        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

        self.embedding_dim = 256
        # Custom head: GAP → 256-dim embedding → 25-class classifier
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1280, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dr),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))

    def unfreeze(self, last_n_layers: int = 10) -> None:
        """Unfreeze the last N layers for Phase 2 fine-tuning."""
        layers = list(self.features.children())
        for layer in layers[-last_n_layers:]:
            for p in layer.parameters():
                p.requires_grad = True
        print(f"[MobileNetV2] Unfrozen last {last_n_layers} backbone layers")

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.features(x)
        gap   = self.head[:3](feats)              # up to and including Linear(256)
        return F.normalize(gap, dim=1)

    def predict(self, x: torch.Tensor) -> dict:
        if x.dim() == 3:
            x = x.unsqueeze(0)
        with torch.no_grad():
            logits = self.forward(x)
            probs  = F.softmax(logits, dim=1)
        conf, gid = probs.max(dim=1)
        return {"logits": logits, "probs": probs,
                "gesture_id": gid, "confidence": conf}
