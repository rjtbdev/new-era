"""Gaussian baseline — one normal distribution for offset, one for hold time.

Fit per-player or globally. Sampling from these produces "statistical relax":
plausible offsets + plausible hold times, but with no coupling to pattern or cursor.

This is the straw-man baseline our models should beat on the detection harness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


@dataclass
class GaussianFit:
    offset_mean: float
    offset_std: float
    hold_mean: float
    hold_std: float
    k1_frac: float   # P(K1) ; else K2 (ignore M1/M2 for sampling simplicity)
    miss_rate: float
    n: int
    # First-order Markov on key choice: P(K1|prev=K1), P(K1|prev=K2)
    k_markov: tuple[float, float] = (0.5, 0.5)


@dataclass
class GaussianBaseline:
    per_user: Dict[int, GaussianFit] = field(default_factory=dict)
    global_fit: Optional[GaussianFit] = None

    def fit(self, df: pd.DataFrame, min_n: int = 100) -> None:
        self.global_fit = _fit_one(df)
        for uid, sub in df.groupby("user_id"):
            if len(sub) < min_n:
                continue
            self.per_user[int(uid)] = _fit_one(sub)

    def get(self, user_id: Optional[int]) -> GaussianFit:
        if user_id is not None and user_id in self.per_user:
            return self.per_user[user_id]
        assert self.global_fit is not None, "call fit() first"
        return self.global_fit

    def sample(self, n: int, user_id: Optional[int] = None, rng: Optional[np.random.Generator] = None) -> pd.DataFrame:
        rng = rng or np.random.default_rng()
        f = self.get(user_id)
        offs = rng.normal(f.offset_mean, f.offset_std, size=n)
        holds = np.clip(rng.normal(f.hold_mean, f.hold_std, size=n), 10.0, 500.0)
        keys = _sample_markov_keys(n, f.k_markov, f.k1_frac, rng)
        outcomes = np.where(rng.random(n) < f.miss_rate, "miss", "300")
        return pd.DataFrame({
            "offset_ms": offs,
            "hold_ms": holds,
            "key_type": keys,
            "outcome": outcomes,
        })


def _sample_markov_keys(n: int, markov: tuple[float, float], k1_frac: float, rng: np.random.Generator) -> np.ndarray:
    out = np.empty(n, dtype=object)
    # start from stationary P(K1) = k1_frac
    prev = 0 if rng.random() < k1_frac else 1
    out[0] = "K1" if prev == 0 else "K2"
    for i in range(1, n):
        p_stay_k1 = markov[0] if prev == 0 else 1.0 - markov[1]
        if prev == 0:
            prev = 0 if rng.random() < p_stay_k1 else 1
        else:
            prev = 0 if rng.random() < markov[1] else 1
        out[i] = "K1" if prev == 0 else "K2"
    return out


def _fit_one(df: pd.DataFrame) -> GaussianFit:
    offs = df["offset_ms"].dropna().to_numpy()
    holds = df["hold_ms"].dropna().to_numpy()
    keys = df["key_type"].dropna().to_list()
    k1 = sum(1 for k in keys if k == "K1")
    km = sum(1 for k in keys if k in ("K1", "K2"))
    miss = (df["outcome"] == "miss").mean()
    # Fit Markov transitions over K1/K2 sequence within each replay group
    k_markov = _fit_markov_keys(df)
    return GaussianFit(
        offset_mean=float(np.mean(offs)) if len(offs) else 0.0,
        offset_std=float(np.std(offs, ddof=1)) if len(offs) > 1 else 20.0,
        hold_mean=float(np.mean(holds)) if len(holds) else 80.0,
        hold_std=float(np.std(holds, ddof=1)) if len(holds) > 1 else 20.0,
        k1_frac=(k1 / km) if km else 0.5,
        miss_rate=float(miss) if not np.isnan(miss) else 0.0,
        n=len(df),
        k_markov=k_markov,
    )


def _fit_markov_keys(df: pd.DataFrame) -> tuple[float, float]:
    """Return (P(K1|K1), P(K1|K2)) averaged over replay groups."""
    counts = {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 0}
    group_cols = ["beatmap_id", "user_id", "mods"]
    if not all(c in df.columns for c in group_cols):
        return (0.5, 0.5)
    for _, g in df.groupby(group_cols):
        seq = g[g["key_type"].isin(["K1", "K2"])]["key_type"].to_list()
        for a, b in zip(seq[:-1], seq[1:]):
            ai = 0 if a == "K1" else 1
            bi = 0 if b == "K1" else 1
            counts[(ai, bi)] += 1
    denom0 = counts[(0, 0)] + counts[(0, 1)]
    denom1 = counts[(1, 0)] + counts[(1, 1)]
    p_k1_given_k1 = counts[(0, 0)] / denom0 if denom0 else 0.5
    p_k1_given_k2 = counts[(1, 0)] / denom1 if denom1 else 0.5
    return (float(p_k1_given_k1), float(p_k1_given_k2))
