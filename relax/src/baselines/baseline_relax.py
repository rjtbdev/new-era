"""`baseline_relax.py` — the straw-man relax generator.

For a given beatmap + cursor trajectory (or synthesized zero-offset cursor):
  - Sample offset from a global/user Gaussian.
  - Sample hold time from a global/user Gaussian.
  - Alternate K1/K2 trivially.
  - Compute hit/miss deterministically from offset + cursor_dist vs hit windows.

Produces the same per-click DataFrame shape as `generator.generate`, so the eval
harness can score it the same way.
"""

from __future__ import annotations

import math
import pathlib
from typing import List, Optional

import numpy as np
import pandas as pd

from src.data.beatmap import Beatmap
from src.data.features import _circle_radius_px, _effective_od, _speed_mult
from src.models.generator import _ho_df_from_beatmap

from src.baselines.gaussian import GaussianBaseline


def generate(
    beatmap: Beatmap,
    *,
    mods: int,
    baseline: GaussianBaseline,
    user_id: Optional[int] = None,
    cursor_frames: Optional[List[dict]] = None,
    seed: int = 0,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    eod = _effective_od(beatmap.od, mods)
    hw300 = 80 - 6 * eod; hw100 = 140 - 8 * eod; hw50 = 200 - 10 * eod
    radius = _circle_radius_px(beatmap.cs, mods)
    speed = _speed_mult(mods)

    ho_df = _ho_df_from_beatmap(beatmap)
    if ho_df.empty:
        return pd.DataFrame()
    if mods & 16:  # HR
        ho_df["ho_y"] = 384.0 - ho_df["ho_y"]
    ho_df["hw300"] = hw300; ho_df["hw100"] = hw100; ho_df["hw50"] = hw50
    ho_df["speed_mult"] = speed
    ho_df["mods"] = mods
    ho_df["circle_radius_px"] = radius
    N = len(ho_df)

    fit = baseline.get(user_id)
    offsets = rng.normal(fit.offset_mean, fit.offset_std, size=N)
    holds = np.clip(rng.normal(fit.hold_mean, fit.hold_std, size=N), 10.0, 500.0)

    ho_df["offset_ms"] = offsets
    ho_df["click_t"] = ho_df["ho_t"] + offsets
    ho_df["hold_ms"] = holds

    if cursor_frames:
        # See generator.py: use cursor_at_ho_t for consistency with the model.
        cf = sorted(cursor_frames, key=lambda f: f["t"])
        ts = np.array([f["t"] for f in cf]); xs = np.array([f["x"] for f in cf]); ys = np.array([f["y"] for f in cf])
        cx = np.interp(ho_df["ho_t"], ts, xs); cy = np.interp(ho_df["ho_t"], ts, ys)
    else:
        cx = ho_df["ho_x"].to_numpy(); cy = ho_df["ho_y"].to_numpy()
    ho_df["click_x"] = cx; ho_df["click_y"] = cy
    ho_df["cursor_dx"] = cx - ho_df["ho_x"]
    ho_df["cursor_dy"] = cy - ho_df["ho_y"]
    ho_df["cursor_dist"] = np.hypot(ho_df["cursor_dx"], ho_df["cursor_dy"])
    ho_df["edge_bias"] = np.clip(ho_df["cursor_dist"] / max(1e-6, radius), 0, 1)

    # outcomes: deterministic from (offset, cursor_dist) vs hit windows
    def _o(off, d):
        if d > radius: return "miss"
        if abs(off) <= hw300: return "300"
        if abs(off) <= hw100: return "100"
        if abs(off) <= hw50: return "50"
        return "miss"
    ho_df["outcome"] = [_o(ho_df["offset_ms"].iloc[i], ho_df["cursor_dist"].iloc[i]) for i in range(N)]

    # keys: 1st-order Markov chain learned from training data
    from src.baselines.gaussian import _sample_markov_keys
    ho_df["key_type"] = _sample_markov_keys(N, fit.k_markov, fit.k1_frac, rng)

    return ho_df
