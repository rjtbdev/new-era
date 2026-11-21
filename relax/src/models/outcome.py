"""Outcome head.

Input: (click_time_offset_ms, cursor_dx, cursor_dy, hw300, hw100, hw50, radius,
        bpm, obj_kind, prev_gap, next_gap).
Output: categorical over {300, 100, 50, miss}.

Since outcome is largely deterministic from (offset, distance) given the hit
windows, this is really modeling "when does a human miss a note they clicked near".
The deterministic baseline already gets most of this right; the network's job is
to learn the *stochastic miss rate under difficulty pressure*.

We train with cross-entropy on observed outcomes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class OutcomeConfig:
    feat_dim: int = 12  # +1 for effective_bpm (Phase A)
    hidden: int = 128
    n_classes: int = 4  # 300, 100, 50, miss
    dropout: float = 0.1


CLASS_NAMES = ["300", "100", "50", "miss"]


class OutcomeModel(nn.Module):
    def __init__(self, cfg: OutcomeConfig):
        super().__init__()
        self.cfg = cfg
        self.net = nn.Sequential(
            nn.Linear(cfg.feat_dim, cfg.hidden), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.hidden), nn.GELU(), nn.Dropout(cfg.dropout),
            nn.Linear(cfg.hidden, cfg.n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def nll(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, target)
