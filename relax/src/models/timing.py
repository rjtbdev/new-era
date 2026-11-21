"""Timing head.

Input per hit object: a window of UP TO `ctx` upcoming + past objects described by
small feature vectors (relative-time deltas, kind-one-hot, new-combo, bpm,
position-dx/dy relative to the current object). No cursor info.

Output: parameters of a mixture of Gaussians over click offset (ms, beatmap time),
plus a learnable scaling for mod effects (handled outside of this module).

Architecture: small transformer over the context window. Current object is the
"target" position (index = ctx_past). Output is produced from its output embedding.

Target: match per-player per-pattern cvUR curves. Loss is NLL of the mixture
against observed offset values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TimingConfig:
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    ctx_past: int = 8     # past objects
    ctx_future: int = 8   # future objects (incl current)
    n_mix: int = 4        # mixture components
    dropout: float = 0.1
    max_players: int = 2048
    player_dim: int = 16
    obj_feat_dim: int = 14


class TimingModel(nn.Module):
    def __init__(self, cfg: TimingConfig):
        super().__init__()
        self.cfg = cfg
        self.obj_proj = nn.Linear(cfg.obj_feat_dim, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.ctx_past + 1 + cfg.ctx_future, cfg.d_model)
        self.player_emb = nn.Embedding(cfg.max_players, cfg.player_dim)
        self.player_proj = nn.Linear(cfg.player_dim, cfg.d_model)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4, dropout=cfg.dropout,
            batch_first=True, activation="gelu", norm_first=True,
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)
        # three outputs per mixture component: logit_pi, mu, log_sigma
        self.head = nn.Linear(cfg.d_model, cfg.n_mix * 3)

    def forward(self, obj_feats: torch.Tensor, player_ids: torch.Tensor, pad_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        obj_feats: (B, T, obj_feat_dim)
        player_ids: (B,) long
        pad_mask: (B, T) bool, True at padded positions (ignored in attention)
        """
        B, T, _ = obj_feats.shape
        x = self.obj_proj(obj_feats)
        # add positional embeddings (0..T-1)
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        x = x + self.pos_emb(pos)
        # add player embedding to every position
        pe = self.player_proj(self.player_emb(player_ids))  # (B, d_model)
        x = x + pe.unsqueeze(1)
        y = self.enc(x, src_key_padding_mask=pad_mask)
        # take the target position: cfg.ctx_past (the current object)
        tgt = y[:, self.cfg.ctx_past, :]  # (B, d_model)
        out = self.head(tgt)  # (B, 3*n_mix)
        pi_logits, mu, log_sigma = torch.split(out, self.cfg.n_mix, dim=-1)
        # Constrain log_sigma for numerical safety
        log_sigma = log_sigma.clamp(math.log(1.0), math.log(120.0))
        return {"pi_logits": pi_logits, "mu": mu, "log_sigma": log_sigma}

    def nll(self, pred: Dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        """Mixture-of-Gaussians NLL."""
        pi = torch.log_softmax(pred["pi_logits"], dim=-1)
        mu = pred["mu"]
        sig = torch.exp(pred["log_sigma"])
        # log N(target | mu, sig)
        # target: (B,) → (B, 1)
        t = target.unsqueeze(-1)
        log_comp = -0.5 * ((t - mu) / sig) ** 2 - torch.log(sig) - 0.5 * math.log(2 * math.pi)
        # (B, K)
        log_mix = torch.logsumexp(pi + log_comp, dim=-1)
        return -log_mix.mean()

    @torch.no_grad()
    def sample(self, pred: Dict[str, torch.Tensor], rng: Optional[torch.Generator] = None) -> torch.Tensor:
        """Sample an offset per row."""
        pi = torch.softmax(pred["pi_logits"], dim=-1)
        mu = pred["mu"]
        sig = torch.exp(pred["log_sigma"])
        comp = torch.multinomial(pi, num_samples=1, generator=rng).squeeze(-1)
        sel_mu = mu.gather(-1, comp.unsqueeze(-1)).squeeze(-1)
        sel_sig = sig.gather(-1, comp.unsqueeze(-1)).squeeze(-1)
        eps = torch.randn_like(sel_mu, generator=rng) if rng is None else torch.empty_like(sel_mu).normal_(generator=rng)
        return sel_mu + sel_sig * eps
