"""Detection-relevant metrics on per-click feature rows.

All functions accept a pandas DataFrame with the columns produced by
`src.data.features.extract_features_for_replay`:
    offset_ms, hold_ms, edge_bias, outcome, key_type, cursor_dist, speed_mult,
    ho_t, ho_kind, prev_gap_ms, next_gap_ms, bpm, ...

and return simple stats dicts. Comparisons use KS test (no scipy; implemented).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd


# ---------------------------------------------------------------- cvUR
def cv_ur(df: pd.DataFrame) -> float:
    """Converted UR = 10 * stddev(offset) / speed_mult.

    'Converted' adjusts for DT/HT so values are comparable across mods (standard
    circleguard convention).
    """
    sub = df.dropna(subset=["offset_ms"])
    if sub.empty:
        return float("nan")
    # Per-row speed_mult; converted offset = offset / speed_mult
    conv = sub["offset_ms"].to_numpy() / sub["speed_mult"].to_numpy()
    return float(10.0 * np.std(conv, ddof=1)) if len(conv) > 1 else float("nan")


# ---------------------------------------------------------------- hold time histogram
def hold_histogram(df: pd.DataFrame, bins: Sequence[float] | None = None) -> Dict[str, Any]:
    sub = df.dropna(subset=["hold_ms"])
    if sub.empty:
        return {"counts": [], "bins": []}
    if bins is None:
        bins = list(np.arange(0, 500, 5))
    counts, edges = np.histogram(sub["hold_ms"].to_numpy(), bins=bins)
    return {
        "counts": counts.tolist(),
        "bins": [float(e) for e in edges],
        "mean": float(sub["hold_ms"].mean()),
        "median": float(sub["hold_ms"].median()),
        "std": float(sub["hold_ms"].std()),
    }


# ---------------------------------------------------------------- edge hit
def edge_hit_rate(df: pd.DataFrame, cutoff: float = 0.85) -> float:
    """Fraction of hits where cursor was at >= cutoff * radius from center.

    cutoff=0.85 is the circleguard default (ur_edge_hit).
    """
    sub = df.dropna(subset=["edge_bias"])
    if sub.empty:
        return float("nan")
    return float((sub["edge_bias"] >= cutoff).mean())


# ---------------------------------------------------------------- miss distribution
def miss_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    n = len(df)
    outs = df["outcome"].value_counts().to_dict()
    for k in ["300", "100", "50", "miss"]:
        outs.setdefault(k, 0)
    return {
        "n": int(n),
        "n_300": int(outs["300"]),
        "n_100": int(outs["100"]),
        "n_50":  int(outs["50"]),
        "n_miss": int(outs["miss"]),
        "accuracy": (6 * outs["300"] + 2 * outs["100"] + outs["50"]) / (6 * n) if n else float("nan"),
        "miss_rate": outs["miss"] / n if n else float("nan"),
    }


# ---------------------------------------------------------------- same-key streaks
def same_key_streaks(df: pd.DataFrame) -> Dict[str, Any]:
    """Distribution of run-lengths where the same key is used consecutively.

    Only counts hits (ignores misses that have no key).
    """
    keys = [k for k in df["key_type"].tolist() if k is not None]
    if not keys:
        return {"max": 0, "mean": 0.0, "hist": {}}
    runs: List[int] = []
    cur = 1
    for i in range(1, len(keys)):
        if keys[i] == keys[i - 1]:
            cur += 1
        else:
            runs.append(cur)
            cur = 1
    runs.append(cur)
    hist: Dict[int, int] = {}
    for r in runs:
        hist[r] = hist.get(r, 0) + 1
    return {
        "max": max(runs),
        "mean": float(np.mean(runs)),
        "runs": runs,
        "hist": {int(k): v for k, v in hist.items()},
    }


# ---------------------------------------------------------------- KS test (no scipy)
def ks_2samp(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    """Two-sample Kolmogorov-Smirnov statistic. Returns stat and p-value (approx).

    Formula from NIST handbook.
    """
    a = np.asarray(a, dtype=float); a = a[~np.isnan(a)]
    b = np.asarray(b, dtype=float); b = b[~np.isnan(b)]
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return {"stat": float("nan"), "p": float("nan"), "n_a": na, "n_b": nb}
    a_sorted = np.sort(a); b_sorted = np.sort(b)
    all_vals = np.concatenate([a_sorted, b_sorted])
    cdf_a = np.searchsorted(a_sorted, all_vals, side="right") / na
    cdf_b = np.searchsorted(b_sorted, all_vals, side="right") / nb
    d = float(np.max(np.abs(cdf_a - cdf_b)))
    # p-value via Smirnov series approximation
    en = np.sqrt(na * nb / (na + nb))
    # Avoid overflow
    lam = (en + 0.12 + 0.11 / en) * d
    p = 2.0 * np.sum((-1) ** (np.arange(1, 101) - 1) * np.exp(-2.0 * (np.arange(1, 101) * lam) ** 2))
    p = max(0.0, min(1.0, float(p)))
    return {"stat": d, "p": p, "n_a": na, "n_b": nb}


# ---------------------------------------------------------------- bundled report
def full_report(df: pd.DataFrame) -> Dict[str, Any]:
    out = {
        "n_clicks": int(len(df)),
        "cvUR": cv_ur(df),
        "hold": hold_histogram(df),
        "edge_hit_rate@0.85": edge_hit_rate(df, 0.85),
        "edge_bias_mean": float(df["edge_bias"].dropna().mean()) if "edge_bias" in df else float("nan"),
        "outcome": miss_distribution(df),
    }
    # If user_id is present, compute same-key streaks per (beatmap, user, mods)
    # group and summarize. Otherwise fall back to the global sequence.
    if {"beatmap_id", "user_id", "mods"}.issubset(df.columns):
        maxes = []
        means = []
        for _, sub in df.groupby(["beatmap_id", "user_id", "mods"]):
            st = same_key_streaks(sub)
            maxes.append(st["max"]); means.append(st["mean"])
        out["same_key_streaks"] = {
            "max_mean": float(np.mean(maxes)) if maxes else 0.0,
            "max_p95": float(np.percentile(maxes, 95)) if maxes else 0.0,
            "max_overall": int(max(maxes)) if maxes else 0,
            "mean_mean": float(np.mean(means)) if means else 0.0,
            "max": int(max(maxes)) if maxes else 0,  # legacy key
        }
    else:
        out["same_key_streaks"] = same_key_streaks(df)
    return out


def compare(human_df: pd.DataFrame, gen_df: pd.DataFrame) -> Dict[str, Any]:
    """Core comparison the eval harness builds reports on."""
    out: Dict[str, Any] = {
        "human": full_report(human_df),
        "generated": full_report(gen_df),
    }
    # KS tests on key distributions
    h_off = human_df["offset_ms"].to_numpy()
    g_off = gen_df["offset_ms"].to_numpy()
    out["ks_offset"] = ks_2samp(h_off, g_off)
    h_hold = human_df["hold_ms"].to_numpy()
    g_hold = gen_df["hold_ms"].to_numpy()
    out["ks_hold"] = ks_2samp(h_hold, g_hold)
    h_edge = human_df["edge_bias"].to_numpy()
    g_edge = gen_df["edge_bias"].to_numpy()
    out["ks_edge_bias"] = ks_2samp(h_edge, g_edge)
    return out
