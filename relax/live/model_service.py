"""ModelService — loads the three heads and produces press decisions per HO.

Decision:
    for each upcoming hit object within the lookahead window, produce:
      - press_time_ms  (absolute beatmap time to fire)
      - key            ("K1" or "K2")
      - confidence     (optional; from the key head softmax)

The service samples offsets from the timing mixture; keys from the key head
with the K1-dominance rule post-process.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch  # DLL init order on Windows
import numpy as np

from src.models.datasets import OBJ_FEAT_DIM
from src.models.generator import load_key_ckpt, load_timing_ckpt
from src.models.rules import RuleConfig

from .feature_builder import FeatureBuilder, FeatureWindow
from .osu_reader import GameState


@dataclass
class PressDecision:
    center_idx: int
    press_time_ms: float
    key: str                  # "K1" | "K2"


class ModelService:
    def __init__(
        self,
        *,
        timing_ckpt: pathlib.Path | str,
        key_ckpt: Optional[pathlib.Path | str] = None,
        device: str = "cpu",  # inference: CPU fine, latency < 10ms for 1M params
        rules: Optional[RuleConfig] = None,
    ):
        self.device = device
        self.timing, self.tcfg, self.player_idx_map = load_timing_ckpt(timing_ckpt)
        self.timing = self.timing.to(device).eval()
        if key_ckpt is not None:
            self.key_model, self.kcfg = load_key_ckpt(key_ckpt)
            self.key_model = self.key_model.to(device).eval()
        else:
            self.key_model = None
            self.kcfg = None
        self.rules = rules or RuleConfig()
        self.rng = np.random.default_rng(self.rules.seed)

        self.fb = FeatureBuilder(
            ctx_past=self.tcfg.ctx_past,
            ctx_future=self.tcfg.ctx_future,
        )

        # per-decision cache: we only emit one decision per HO, sticky.
        self._decisions: Dict[int, PressDecision] = {}
        # streak state for alternation rule
        self._last_key: Optional[str] = None
        self._streak: int = 1

    def on_cursor(self, t_ms: float, x: float, y: float) -> None:
        self.fb.update_cursor(t_ms, x, y)

    def _player_pid(self, state: GameState) -> int:
        if state.player_id is None:
            return 0
        return self.player_idx_map.get(int(state.player_id), 0)

    def _sample_press_time(self, win: FeatureWindow, pid: int, ref_t: float) -> float:
        feats = torch.from_numpy(win.obj_feats).unsqueeze(0).to(self.device)
        pad = torch.from_numpy(win.pad_mask).unsqueeze(0).to(self.device)
        pids = torch.tensor([pid], dtype=torch.long, device=self.device)
        with torch.no_grad():
            pred = self.timing(feats, pids, pad_mask=pad)
            pi = torch.softmax(pred["pi_logits"], dim=-1)
            comp = torch.multinomial(pi, num_samples=1).squeeze(-1)
            mu = pred["mu"].gather(-1, comp.unsqueeze(-1)).squeeze(-1)
            sig = torch.exp(pred["log_sigma"]).gather(-1, comp.unsqueeze(-1)).squeeze(-1)
            offset = float(mu + sig * torch.randn_like(mu))
        return ref_t + offset

    def _pick_key(self, state: GameState, center_idx: int, ref_t: float) -> str:
        # Apply the same rules we apply offline. Key rule priority:
        #   1. streak cap: if prev_key streak == max_same_key_streak, flip
        #   2. forced alternation: if prev-gap < alt threshold, flip
        #   3. preferred_key dominance: pin to K1 (or configured) for non-stream
        prev_key = self._last_key
        hos = state.hit_objects
        prev_gap = abs(ref_t - hos[center_idx - 1].t) if center_idx > 0 else float("inf")
        key: str
        if prev_key is None:
            key = self.rules.preferred_key or "K1"
        elif prev_gap < self.rules.alt_gap_threshold_ms:
            key = "K2" if prev_key == "K1" else "K1"
        elif self._streak >= self.rules.max_same_key_streak:
            key = "K2" if prev_key == "K1" else "K1"
        elif self.rules.preferred_key is not None:
            key = self.rules.preferred_key
        else:
            # could call the key LSTM here if we wanted its vote
            key = prev_key  # stay on previous as fallback
        if key == prev_key:
            self._streak += 1
        else:
            self._streak = 1
        self._last_key = key
        return key

    def plan(self, state: GameState, lookahead_ms: float = 400.0) -> List[PressDecision]:
        """Return decisions for any upcoming hit objects within lookahead
        that we haven't already planned for.

        The caller should call this every ~30ms and feed the decisions to
        the dispatcher. Decisions are sticky — once planned for a given
        center_idx, we don't re-plan it (prevents oscillation)."""
        decisions: List[PressDecision] = []
        hos = state.hit_objects
        now = state.time_ms
        pid = self._player_pid(state)
        for i, h in enumerate(hos):
            if h.kind == "spinner":
                continue
            if h.t < now - 50.0:
                continue
            if h.t > now + lookahead_ms:
                break
            if i in self._decisions:
                continue
            win = self.fb.build_window(state, i)
            press_t = self._sample_press_time(win, pid, h.t)
            key = self._pick_key(state, i, h.t)
            d = PressDecision(center_idx=i, press_time_ms=press_t, key=key)
            self._decisions[i] = d
            decisions.append(d)
        return decisions
