"""Key choice head.

Autoregressive over sequence of (recent key, BPM at object, gap-from-prev).
Output: P(K1) vs P(K2) for the next note. Trained by teacher forcing.

Small LSTM over the last 8 notes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class KeyConfig:
    ctx: int = 8
    feat_dim: int = 6  # prev_key_onehot(2), bpm/300, prev_gap/1000, is_new_combo, effective_bpm/400 (Phase A)
    hidden: int = 64
    dropout: float = 0.1


class KeyModel(nn.Module):
    def __init__(self, cfg: KeyConfig):
        super().__init__()
        self.cfg = cfg
        self.lstm = nn.LSTM(cfg.feat_dim, cfg.hidden, batch_first=True)
        self.head = nn.Linear(cfg.hidden, 2)  # logits for K1, K2
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        # seq: (B, ctx, feat_dim)
        h, _ = self.lstm(seq)
        last = h[:, -1, :]
        return self.head(self.drop(last))

    def nll(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, target)
