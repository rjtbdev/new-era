"""FeatureBuilder — produces the 17-dim per-hit-object feature window that
the timing head expects.

The timing head was trained with ctx_past=8, ctx_future=8, obj_feat_dim=17
(12 base + 2 Phase A + 3 cursor-at-ho_t). This module computes the same
features at inference time.

Features (must match src/data/features.py and src/models/datasets.py
OBJ_FEAT_DIM=17 layout):
     0 dt                 (ho_t - ref_t) / 1000 seconds, relative to center
     1 log|dt|            signed log-time gap
     2 x_norm             ho.x / 512
     3 y_norm             ho.y / 384
     4 is_circle
     5 is_slider
     6 is_spinner
     7 new_combo
     8 bpm / 300
     9 prev_gap_ms / 1000 (signed)
    10 next_gap_ms / 1000
    11 hw300 / 50
    12 effective_bpm / 400
    13 skill_percentile   (default 0.95)
    14 cursor_at_ho_dist / 100
    15 cursor_at_ho_vel   / 2
    16 cursor_approach_rate / 2

For live play, cursor features are computed from the user's live cursor
trace. The FeatureBuilder keeps a rolling cursor buffer so it can look up
cursor position at any ho_t (± a few hundred ms into the past).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .osu_reader import GameState, HitObject

OBJ_FEAT_DIM = 17


@dataclass
class FeatureWindow:
    """What model_service feeds to the timing head for ONE center hit object."""
    center_idx: int                     # which HitObject in beatmap this is for
    obj_feats: np.ndarray               # shape (T, 17) where T = ctx_past+1+ctx_future
    pad_mask: np.ndarray                # shape (T,) True where padded
    cursor_at_ho_dist: float            # for optional outcome-head feats
    cursor_at_ho_vel: float
    cursor_approach_rate: float


class FeatureBuilder:
    """Maintains a cursor history ring buffer and builds feature windows."""

    def __init__(
        self,
        *,
        ctx_past: int = 8,
        ctx_future: int = 8,
        cursor_buffer_ms: float = 500.0,
        skill_percentile: float = 0.95,
    ):
        self.ctx_past = ctx_past
        self.ctx_future = ctx_future
        self.T = ctx_past + 1 + ctx_future
        self.skill_percentile = skill_percentile

        # ring buffer of (t_ms, x, y) cursor samples
        self._cursor: deque[Tuple[float, float, float]] = deque(maxlen=1024)
        self._cursor_buf_ms = cursor_buffer_ms

        # Beatmap-static per-HO feature table (recomputed when hit_objects list changes)
        self._ho_base_feats: Optional[np.ndarray] = None  # shape (N, 17)
        self._ho_ts: Optional[np.ndarray] = None
        self._n_ho: int = 0
        self._bm_fingerprint: int = 0

    def update_cursor(self, t_ms: float, x: float, y: float) -> None:
        """Push the current cursor sample. Keeps only recent history."""
        self._cursor.append((t_ms, x, y))
        if self._cursor:
            # drop too-old samples
            cutoff = t_ms - self._cursor_buf_ms
            while self._cursor and self._cursor[0][0] < cutoff:
                self._cursor.popleft()

    def _cursor_at(self, t: float) -> Tuple[float, float]:
        """Interpolate cursor position at time t_ms. Falls back to extrapolate
        from the nearest sample at the edges."""
        if not self._cursor:
            return (256.0, 192.0)
        if len(self._cursor) == 1:
            return (self._cursor[0][1], self._cursor[0][2])
        # Linear scan — small ring; could binary search if buffer grows.
        buf = list(self._cursor)
        if t <= buf[0][0]:
            return (buf[0][1], buf[0][2])
        if t >= buf[-1][0]:
            return (buf[-1][1], buf[-1][2])
        for i in range(len(buf) - 1):
            a = buf[i]; b = buf[i+1]
            if a[0] <= t <= b[0]:
                if b[0] == a[0]:
                    return (a[1], a[2])
                f = (t - a[0]) / (b[0] - a[0])
                return (a[1] + f * (b[1] - a[1]),
                        a[2] + f * (b[2] - a[2]))
        return (buf[-1][1], buf[-1][2])

    def _ensure_base_feats(self, state: GameState) -> None:
        """Precompute the per-HO features that don't depend on cursor state."""
        # Fingerprint changes when the beatmap changes
        fp = len(state.hit_objects)
        if self._ho_base_feats is not None and fp == self._bm_fingerprint:
            return
        hos = state.hit_objects
        n = len(hos)
        feats = np.zeros((n, OBJ_FEAT_DIM), dtype=np.float32)
        ts = np.zeros(n, dtype=np.float64)
        for i, h in enumerate(hos):
            ts[i] = h.t
            feats[i, 2] = h.x / 512.0
            feats[i, 3] = h.y / 384.0
            feats[i, 4] = 1.0 if h.kind == "circle" else 0.0
            feats[i, 5] = 1.0 if h.kind == "slider" else 0.0
            feats[i, 6] = 1.0 if h.kind == "spinner" else 0.0
            feats[i, 7] = 1.0 if h.new_combo else 0.0
            prev_gap = (hos[i-1].t - h.t) if i > 0 else 0.0
            next_gap = (hos[i+1].t - h.t) if i + 1 < n else 0.0
            feats[i, 9] = prev_gap / 1000.0
            feats[i, 10] = next_gap / 1000.0
            feats[i, 11] = state.hw300 / 50.0
            # bpm at hit-object time — we don't have per-HO BPM from GameState;
            # use a sane default derived from gap. 60000/gap gives BPM-equivalent.
            # For live, we could pass per-HO BPM from the reader if needed.
            if prev_gap != 0:
                approx_bpm = max(60.0, min(400.0, 60000.0 / max(1.0, abs(prev_gap) * 4)))
            else:
                approx_bpm = 120.0
            feats[i, 8] = approx_bpm / 300.0
            feats[i, 12] = (approx_bpm * state.speed_mult) / 400.0
            feats[i, 13] = self.skill_percentile
        self._ho_base_feats = feats
        self._ho_ts = ts
        self._n_ho = n
        self._bm_fingerprint = fp

    def build_window(self, state: GameState, center_idx: int) -> FeatureWindow:
        self._ensure_base_feats(state)
        assert self._ho_base_feats is not None and self._ho_ts is not None
        n = self._n_ho
        T = self.T
        feats = np.zeros((T, OBJ_FEAT_DIM), dtype=np.float32)
        pad = np.ones(T, dtype=bool)

        ref_t = float(self._ho_ts[center_idx])
        lo = center_idx - self.ctx_past
        hi = center_idx + self.ctx_future + 1

        # Find the overlap of [lo, hi) with [0, n)
        src_lo = max(0, lo)
        src_hi = min(n, hi)
        if src_hi > src_lo:
            dst_lo = src_lo - lo
            dst_hi = src_hi - lo
            feats[dst_lo:dst_hi] = self._ho_base_feats[src_lo:src_hi]
            # Fill time-relative columns (0 = dt, 1 = log|dt|)
            dts = (self._ho_ts[src_lo:src_hi] - ref_t) / 1000.0
            feats[dst_lo:dst_hi, 0] = dts.astype(np.float32)
            feats[dst_lo:dst_hi, 1] = (np.sign(dts) * np.log1p(np.abs(dts))).astype(np.float32)
            pad[dst_lo:dst_hi] = False

        # Cursor features for the CENTER hit object's absolute time
        cx, cy = self._cursor_at(ref_t)
        cxp, cyp = self._cursor_at(ref_t - 20.0)
        cxp2, cyp2 = self._cursor_at(ref_t - 40.0)
        hx = state.hit_objects[center_idx].x
        hy = state.hit_objects[center_idx].y
        dist_at = math.hypot(cx - hx, cy - hy)
        dist_prev2 = math.hypot(cxp2 - hx, cyp2 - hy)
        vel = math.hypot(cx - cxp, cy - cyp) / 20.0
        approach = (dist_at - dist_prev2) / 40.0

        # Scale and stamp into the CENTER row (ctx_past position)
        center_row = self.ctx_past
        feats[center_row, 14] = dist_at / 100.0
        feats[center_row, 15] = vel / 2.0
        feats[center_row, 16] = approach / 2.0

        return FeatureWindow(
            center_idx=center_idx, obj_feats=feats, pad_mask=pad,
            cursor_at_ho_dist=dist_at,
            cursor_at_ho_vel=vel,
            cursor_approach_rate=approach,
        )
