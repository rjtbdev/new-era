"""Dataset wrappers for the three heads.

Reads parquet shards from data/processed/, aggregates per (beatmap_id,user_id,mods),
and produces training examples:

TimingDataset: per-row context window of object features + player id → target offset.
OutcomeDataset: per hit object features + click_time + cursor_pos → outcome class.
KeyDataset: sequence of recent key_type + BPM + pattern → next key_type.
"""

from __future__ import annotations

import glob
import math
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

PROJECT = pathlib.Path(__file__).resolve().parents[2]
PROC = PROJECT / "data" / "processed"


# ---------------------------------------------------------------- loader
def load_all_shards() -> pd.DataFrame:
    paths = sorted(glob.glob(str(PROC / "shard_*.parquet")))
    if not paths:
        return pd.DataFrame()
    dfs = [pd.read_parquet(p) for p in paths]
    df = pd.concat(dfs, ignore_index=True)
    return df


# ---------------------------------------------------------------- object feature vec
OBJ_FEAT_DIM = 17  # 12 pre-A + 2 phase-A + 3 cursor-at-ho_t


def _make_obj_feats(df: pd.DataFrame, ref_t: np.ndarray) -> np.ndarray:
    """Dense feature per object row relative to a reference time per row.

    Features:
      0  dt                — (ho_t - ref_t) / 1000  (seconds)
      1  log|dt|           — signed log-time gap
      2  x_norm            — x / 512
      3  y_norm            — y / 384
      4  is_circle
      5  is_slider
      6  is_spinner
      7  new_combo
      8  bpm / 300            (base beatmap BPM)
      9  prev_gap / 1000 (signed)
      10 next_gap / 1000
      11 hw300 / 50
      12 effective_bpm / 400   (bpm × speed_mult; handles DT/HT; added Phase A)
      13 skill_percentile      (0..1, from leaderboard rank; added Phase A)
    """
    dt = (df["ho_t"].to_numpy() - ref_t) / 1000.0
    out = np.zeros((len(df), OBJ_FEAT_DIM), dtype=np.float32)
    out[:, 0] = dt
    out[:, 1] = np.sign(dt) * np.log1p(np.abs(dt))
    out[:, 2] = df["ho_x"].fillna(256).to_numpy() / 512.0
    out[:, 3] = df["ho_y"].fillna(192).to_numpy() / 384.0
    kind = df["ho_kind"].fillna("circle").to_numpy()
    out[:, 4] = (kind == "circle").astype(np.float32)
    out[:, 5] = (kind == "slider").astype(np.float32)
    out[:, 6] = (kind == "spinner").astype(np.float32)
    out[:, 7] = df["ho_new_combo"].fillna(False).astype(np.float32)
    out[:, 8] = df["bpm"].fillna(120).to_numpy() / 300.0
    out[:, 9] = df["prev_gap_ms"].fillna(0).to_numpy() / 1000.0
    out[:, 10] = df["next_gap_ms"].fillna(0).to_numpy() / 1000.0
    out[:, 11] = df["hw300"].fillna(40).to_numpy() / 50.0
    # Phase A additions:
    if "effective_bpm" in df.columns:
        eff = df["effective_bpm"].fillna(120).to_numpy()
    else:
        # Fallback: reconstruct from bpm × speed_mult (old shards lacking the col)
        base = df["bpm"].fillna(120).to_numpy()
        spd = df["speed_mult"].fillna(1.0).to_numpy() if "speed_mult" in df.columns else np.ones(len(df))
        eff = base * spd
    out[:, 12] = eff / 400.0  # ~300bpm DT ~ 1.125
    if "skill_percentile" in df.columns:
        out[:, 13] = df["skill_percentile"].fillna(0.95).to_numpy().astype(np.float32)
    else:
        out[:, 13] = 0.95
    return out


# ---------------------------------------------------------------- player indexer
@dataclass
class PlayerIndex:
    """Maps user_id → embedding index. Unknown users → 0 (shared slot)."""
    user_to_idx: Dict[int, int]
    max_players: int

    @classmethod
    def build(cls, df: pd.DataFrame, max_players: int = 2048) -> "PlayerIndex":
        counts = df["user_id"].value_counts()
        top = counts.head(max_players - 1).index.tolist()
        mapping = {int(uid): (i + 1) for i, uid in enumerate(top)}
        return cls(user_to_idx=mapping, max_players=max_players)

    def encode(self, user_ids: np.ndarray) -> np.ndarray:
        return np.array([self.user_to_idx.get(int(u), 0) for u in user_ids], dtype=np.int64)


# ---------------------------------------------------------------- Timing dataset
class TimingDataset(Dataset):
    """Each example = one hit object with its context window + target offset.

    Context: ctx_past past objects + current object + ctx_future future objects.
    Objects outside the replay boundary are zero-padded with a mask.

    Features are PRECOMPUTED once at __init__ into dense numpy arrays, so
    __getitem__ is pure array indexing — no pandas in the hot loop. This is
    what lets the GPU stay saturated during training.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        player_index: PlayerIndex,
        *,
        ctx_past: int = 8,
        ctx_future: int = 8,
        kind: str = "train",  # 'train'|'val' selection done by caller
    ):
        self.ctx_past = ctx_past
        self.ctx_future = ctx_future
        T = ctx_past + 1 + ctx_future

        df = df.dropna(subset=["offset_ms"]).copy()
        df.sort_values(["beatmap_id", "user_id", "mods", "ho_t"], inplace=True)
        df = df.reset_index(drop=True)

        # Per-row absolute features (not dt-relative yet) — we store dt separately.
        # We precompute feature columns that depend only on the hit object itself.
        n = len(df)
        base_feats = np.zeros((n, OBJ_FEAT_DIM), dtype=np.float32)
        ho_t = df["ho_t"].to_numpy(dtype=np.float64)
        # Columns 2..13 are time-independent; column 0 (dt) and 1 (log|dt|) are
        # recomputed per training example relative to the center's ho_t.
        base_feats[:, 2] = df["ho_x"].fillna(256).to_numpy() / 512.0
        base_feats[:, 3] = df["ho_y"].fillna(192).to_numpy() / 384.0
        kind_col = df["ho_kind"].fillna("circle").to_numpy()
        base_feats[:, 4] = (kind_col == "circle").astype(np.float32)
        base_feats[:, 5] = (kind_col == "slider").astype(np.float32)
        base_feats[:, 6] = (kind_col == "spinner").astype(np.float32)
        base_feats[:, 7] = df["ho_new_combo"].fillna(False).astype(np.float32)
        base_feats[:, 8] = df["bpm"].fillna(120).to_numpy() / 300.0
        base_feats[:, 9] = df["prev_gap_ms"].fillna(0).to_numpy() / 1000.0
        base_feats[:, 10] = df["next_gap_ms"].fillna(0).to_numpy() / 1000.0
        base_feats[:, 11] = df["hw300"].fillna(40).to_numpy() / 50.0
        if "effective_bpm" in df.columns:
            eff = df["effective_bpm"].fillna(120).to_numpy()
        else:
            base = df["bpm"].fillna(120).to_numpy()
            spd = df["speed_mult"].fillna(1.0).to_numpy() if "speed_mult" in df.columns else np.ones(n)
            eff = base * spd
        base_feats[:, 12] = eff / 400.0
        if "skill_percentile" in df.columns:
            base_feats[:, 13] = df["skill_percentile"].fillna(0.95).to_numpy().astype(np.float32)
        else:
            base_feats[:, 13] = 0.95
        # Cursor-at-ho_t features (added for nomod robustness). Scale to roughly [-1, 1].
        if "cursor_at_ho_dist" in df.columns:
            base_feats[:, 14] = df["cursor_at_ho_dist"].fillna(100.0).to_numpy().astype(np.float32) / 100.0  # ~radius unit
        if "cursor_at_ho_vel" in df.columns:
            base_feats[:, 15] = df["cursor_at_ho_vel"].fillna(0.0).to_numpy().astype(np.float32) / 2.0  # px/ms; 2 px/ms is fast
        if "cursor_approach_rate" in df.columns:
            base_feats[:, 16] = df["cursor_approach_rate"].fillna(0.0).to_numpy().astype(np.float32) / 2.0

        self.ho_t = ho_t.astype(np.float32)  # absolute times, for dt recompute
        self.base_feats = base_feats
        self.user_ids = df["user_id"].to_numpy(dtype=np.int64)
        self.offset_ms = df["offset_ms"].to_numpy(dtype=np.float32)

        # Pre-encode player ids once
        self.player_idx = player_index.encode(self.user_ids).astype(np.int64)

        # Group boundaries over the sorted df: (start, end) per (beatmap, user, mods)
        sizes = df.groupby(["beatmap_id", "user_id", "mods"], sort=False).size().to_numpy()
        self.group_start = np.empty(len(sizes), dtype=np.int64)
        self.group_end = np.empty(len(sizes), dtype=np.int64)
        s = 0
        for i, sz in enumerate(sizes):
            self.group_start[i] = s
            self.group_end[i] = s + sz
            s += sz

        # For each row globally, store (group_id, local_offset)
        self.row_group = np.empty(n, dtype=np.int64)
        self.row_local = np.empty(n, dtype=np.int64)
        for gi in range(len(sizes)):
            a, b = self.group_start[gi], self.group_end[gi]
            self.row_group[a:b] = gi
            self.row_local[a:b] = np.arange(b - a, dtype=np.int64)

        self._T = T

    def __len__(self) -> int:
        return len(self.ho_t)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        gi = self.row_group[idx]
        local = self.row_local[idx]
        gs = self.group_start[gi]
        ge = self.group_end[gi]
        T = self._T

        # absolute global indices for each context slot; -1 means out-of-bounds pad
        center_global = gs + local
        lp0 = local - self.ctx_past
        feats = np.zeros((T, OBJ_FEAT_DIM), dtype=np.float32)
        pad_mask = np.ones(T, dtype=bool)

        # Determine in-bounds slice within the group
        g_len = ge - gs
        # local positions for each slot
        lo = max(0, -lp0)                               # left slot where group starts
        hi = min(T, g_len - lp0)                        # right slot where group ends
        if hi > lo:
            # global row ids for the valid window
            start_global = gs + (lp0 + lo)
            end_global = gs + (lp0 + hi)
            feats[lo:hi] = self.base_feats[start_global:end_global]
            # Patch dt column relative to center
            ref_t = self.ho_t[center_global]
            dt = (self.ho_t[start_global:end_global] - ref_t) / 1000.0
            feats[lo:hi, 0] = dt
            feats[lo:hi, 1] = np.sign(dt) * np.log1p(np.abs(dt))
            pad_mask[lo:hi] = False

        return {
            "obj_feats": torch.from_numpy(feats),
            "pad_mask": torch.from_numpy(pad_mask),
            "player_id": torch.tensor(self.player_idx[idx], dtype=torch.long),
            "target": torch.tensor(self.offset_ms[idx], dtype=torch.float32),
        }


def collate(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {
        "obj_feats": torch.stack([b["obj_feats"] for b in batch]),
        "pad_mask": torch.stack([b["pad_mask"] for b in batch]),
        "player_id": torch.stack([b["player_id"] for b in batch]),
        "target": torch.stack([b["target"] for b in batch]),
    }
