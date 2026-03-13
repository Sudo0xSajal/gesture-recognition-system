"""
code/models/lstm_model.py
=========================
STEP 3 — Gesture Recognition AI Model: LSTM for temporal / motion gestures

Ma'am's document specifies LSTM (RNN) for movement-based gestures.
Some gestures cannot be correctly classified from a single frame:
  - Hand Shaking (#19)    — requires seeing the shaking motion over time
  - Both Hands Raise (#18)— transition from resting to raised position
  - Hand Wave (#3)        — waving motion across multiple frames
  - Circle Motion (#14)   — full circle requires temporal context

The LSTM receives a sequence of consecutive landmark vectors and outputs
one gesture prediction for the entire sequence.

Architecture
------------
  Input: (batch, seq_len, 63)    — sequence of normalised landmark vectors
    → LSTM (hidden=128, layers=2, bidirectional=True, dropout=0.3)
  Output: (batch, seq_len, 256)  — hidden states at each time step
    → Attention pooling over time steps
    → (batch, 256)
    → Linear(256 → 25)

Bidirectional LSTM processes the sequence both forward and backward,
capturing both the start and end of a gesture motion.

Temporal Attention weights each time step's contribution to the final
prediction — the most discriminative frames (peak of motion) are weighted more.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import NUM_CLASSES, MODEL_CONFIG


class _TemporalAttention(nn.Module):
    """
    Soft attention over LSTM time steps.

    Computes a scalar weight for each time step t:
      score_t = tanh(W * h_t + b)
      alpha_t = softmax(score_t)
      context = sum_t(alpha_t * h_t)

    The most informative frames (e.g. peak of a shaking motion)
    receive higher weights.
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        h : (N, T, hidden_dim) — LSTM hidden states across time

        Returns
        -------
        context : (N, hidden_dim) — attention-weighted sum
        """
        score  = self.v(torch.tanh(self.W(h)))    # (N, T, 1)
        alpha  = F.softmax(score, dim=1)           # (N, T, 1)
        context = (alpha * h).sum(dim=1)           # (N, hidden_dim)
        return context


class LSTMModel(nn.Module):
    """
    Bidirectional LSTM with temporal attention for sequential gesture recognition.

    Parameters
    ----------
    input_dim   : int  — 63 (21 landmarks × 3)
    hidden_dim  : int  — LSTM hidden size per direction (default 128)
    num_layers  : int  — LSTM depth (default 2)
    num_classes : int  — 25
    dropout     : float
    """

    def __init__(
        self,
        input_dim:   int   = 63,
        hidden_dim:  int   = None,
        num_layers:  int   = None,
        num_classes: int   = NUM_CLASSES,
        dropout:     float = None,
    ):
        super().__init__()
        hidden_dim = hidden_dim or MODEL_CONFIG["lstm_hidden"]   # 128
        num_layers = num_layers or MODEL_CONFIG["lstm_layers"]   # 2
        dropout    = dropout    or MODEL_CONFIG["lstm_dropout"]  # 0.3

        # ── Input projection: 63 → hidden_dim ─────────────────────────────────
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )

        # ── Bidirectional LSTM ────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        lstm_out_dim = hidden_dim * 2   # bidirectional → 2 × hidden

        # ── Temporal attention ─────────────────────────────────────────────────
        self.attention = _TemporalAttention(lstm_out_dim)

        self.embedding_dim = lstm_out_dim   # 256

        # ── Classifier head ────────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.LayerNorm(lstm_out_dim),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_dim, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
                # Set forget gate bias to 1 (helps with long sequences)
                n = param.size(0)
                param.data[n // 4: n // 2].fill_(1.0)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (N, T, 63) — sequence of landmark vectors

        Returns
        -------
        logits : (N, 25)
        """
        x   = self.input_proj(x)               # (N, T, hidden_dim)
        h, _ = self.lstm(x)                    # (N, T, lstm_out_dim)
        ctx  = self.attention(h)               # (N, lstm_out_dim)
        return self.classifier(ctx)            # (N, 25)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Return L2-normalised lstm_out_dim embeddings."""
        x   = self.input_proj(x)
        h, _ = self.lstm(x)
        ctx  = self.attention(h)
        return F.normalize(ctx, dim=1)

    def forward_sequence(self, x: torch.Tensor) -> dict:
        """
        Return per-frame predictions in addition to the sequence-level prediction.
        Useful for visualising which frames triggered the gesture.

        Returns
        -------
        dict:
          "sequence_logits" : (N, 25)          — main prediction
          "frame_logits"    : (N, T, 25)       — per-frame predictions
          "attention_weights": (N, T, 1)       — attention scores
        """
        proj  = self.input_proj(x)
        h, _  = self.lstm(proj)
        score = self.attention.v(torch.tanh(self.attention.W(h)))  # (N, T, 1)
        alpha = F.softmax(score, dim=1)
        ctx   = (alpha * h).sum(dim=1)
        seq_logits   = self.classifier(ctx)
        frame_logits = self.classifier(self.classifier[0](h))      # per frame
        return {
            "sequence_logits":  seq_logits,
            "frame_logits":     frame_logits,
            "attention_weights": alpha,
        }

    def predict(self, x: torch.Tensor) -> dict:
        if x.dim() == 2:
            x = x.unsqueeze(0)
        with torch.no_grad():
            logits = self.forward(x)
            probs  = F.softmax(logits, dim=1)
        conf, gid = probs.max(dim=1)
        return {"logits": logits, "probs": probs,
                "gesture_id": gid, "confidence": conf}
