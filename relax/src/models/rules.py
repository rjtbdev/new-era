"""Tunable post-process rules applied to the raw model output.

The trained heads give us (click_time, outcome, key) per hit object. These
rules run after generation and before the dataframe is returned, so the caller
can tune behavior without retraining.

Defaults are chosen to be neutral — i.e. `apply_rules` with a default
`RuleConfig` should reproduce (near-exactly) the pre-rules generator output,
except for the streak limiter which caps at 25 (the model observed streaks of
49-52, so the cap is a real intervention).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np
import pandas as pd


PreferredKey = Literal["K1", "K2"]
MissMode = Literal["random", "pressure"]


@dataclass
class RuleConfig:
    # Rule 1: force alternation when physical gap is below threshold.
    # 80ms ≈ 1/4 at 187 BPM — the human single-tap / alternate crossover point.
    alt_gap_threshold_ms: float = 80.0

    # Rule 2: pin the dominant hand for all single-tap (non-forced-alternate)
    # notes. Humans pick a dominant hand and stay with it across a whole map;
    # letting the model choose per-note causes within-replay drift that is
    # obvious to any observer. Default "K1" (most common dominant). Set to
    # "K2" if you tap K2-dominant, or None to let the model drift (legacy).
    preferred_key: Optional[PreferredKey] = "K1"
    bias_logit: float = 1.0  # retained for API compat; unused when preferred_key set

    # Rule 3: miss injection. Default off — model plays cleanly unless caller opts in.
    miss_rate_target: float = 0.0        # 0.0 .. 0.1
    miss_mode: MissMode = "random"
    # Pressure-mode knobs (used only when miss_mode="pressure"):
    pressure_stream_gap_ms: float = 100.0   # gap below this counts as "stream"
    pressure_stream_min_len: int = 5         # need this many consecutive stream notes
    pressure_endgame_frac: float = 0.85      # object_index / total above this = endgame
    pressure_streak_threshold: int = 20      # same-key streak longer than this
    miss_shift_ms: float = 100.0             # how late to shift the click for a miss

    # Rule 4: cap consecutive same-key presses.
    # LSTM over-commits (eval showed 49-52). 25 keeps us comfortably under a
    # typical Circleguard same-key-streak flag threshold.
    max_same_key_streak: int = 25

    # Rule 5: optional gaussian jitter on click time (default off — model's
    # 9.7ms residual std is already human-tier).
    jitter_std_ms: float = 0.0

    # Random seed used for miss rolls and jitter. The generator already has a
    # seed; passing it through keeps generation deterministic for eval.
    seed: int = 0

    def as_dict(self) -> dict:
        return {
            "alt_gap_threshold_ms": self.alt_gap_threshold_ms,
            "preferred_key": self.preferred_key,
            "bias_logit": self.bias_logit,
            "miss_rate_target": self.miss_rate_target,
            "miss_mode": self.miss_mode,
            "pressure_stream_gap_ms": self.pressure_stream_gap_ms,
            "pressure_stream_min_len": self.pressure_stream_min_len,
            "pressure_endgame_frac": self.pressure_endgame_frac,
            "pressure_streak_threshold": self.pressure_streak_threshold,
            "miss_shift_ms": self.miss_shift_ms,
            "max_same_key_streak": self.max_same_key_streak,
            "jitter_std_ms": self.jitter_std_ms,
        }


def apply_rules(ho_df: pd.DataFrame, cfg: RuleConfig) -> pd.DataFrame:
    """Modify generator output in-place style (returns a new df for safety).

    Expects columns: ho_t, click_t, offset_ms, key_type, outcome, prev_gap_ms,
    cursor_dist, circle_radius_px, hw50.
    """
    if ho_df.empty:
        return ho_df
    df = ho_df.copy()
    rng = np.random.default_rng(cfg.seed)
    n = len(df)

    # --- Rule 5: timing jitter (applied first so downstream outcome recompute
    # sees the jittered offset).
    if cfg.jitter_std_ms > 0.0:
        noise = rng.normal(0.0, cfg.jitter_std_ms, size=n)
        df["offset_ms"] = df["offset_ms"].to_numpy() + noise
        df["click_t"] = df["ho_t"].to_numpy() + df["offset_ms"].to_numpy()

    # --- Rules 1 + 4 + 2: key choice post-process (streak + alternation + bias).
    # We rewrite key_type sequentially because each decision depends on the
    # chosen previous key.
    keys = df["key_type"].tolist()
    # prev_gap_ms in the generator is stored as (prev_ho_t - current_ho_t), i.e.
    # negative. We want |gap|.
    gaps = np.abs(df["prev_gap_ms"].to_numpy()).astype(float)
    streak = 1  # current same-key run length
    for i in range(n):
        gap = gaps[i] if i > 0 else float("inf")  # first click is never "stream"
        prev_key = keys[i - 1] if i > 0 else None
        # Rule 4: streak cap (flip if we'd hit max_same_key_streak with this key)
        if prev_key is not None and keys[i] == prev_key and streak >= cfg.max_same_key_streak:
            keys[i] = "K2" if prev_key == "K1" else "K1"
        # Rule 1: forced alternation at high BPM (overrides whatever we have)
        if prev_key is not None and gap < cfg.alt_gap_threshold_ms:
            keys[i] = "K2" if prev_key == "K1" else "K1"
        # Rule 2: preferred-key dominance in the single-tap (non-forced-alt) zone.
        # Humans pick a dominant hand and stick with it; the model's key head
        # drifts its "default" across the map, which is a tell. When
        # preferred_key is set, every single-tap note is pinned to that key.
        # The alternation rule above still fires on fast consecutive notes,
        # so streams get alternated normally.
        elif cfg.preferred_key is not None:
            keys[i] = cfg.preferred_key
        # update streak counter
        if prev_key is not None and keys[i] == prev_key:
            streak += 1
        else:
            streak = 1
    df["key_type"] = keys

    # --- Rule 3: miss injection.
    if cfg.miss_rate_target > 0.0:
        # Build a per-click "should miss" mask.
        miss_mask = np.zeros(n, dtype=bool)
        if cfg.miss_mode == "random":
            miss_mask = rng.random(n) < cfg.miss_rate_target
        else:  # pressure
            # Mark objects that are "under pressure":
            #  - in a stream of >= pressure_stream_min_len consecutive small gaps
            #  - in the endgame (last 15% of the map)
            #  - following a long same-key streak
            stream_run = 0
            same_key_run = 1
            pressure = np.zeros(n, dtype=bool)
            for i in range(n):
                gap = gaps[i] if i > 0 else float("inf")
                if gap < cfg.pressure_stream_gap_ms:
                    stream_run += 1
                else:
                    stream_run = 0
                if i > 0 and keys[i] == keys[i - 1]:
                    same_key_run += 1
                else:
                    same_key_run = 1
                endgame = (i / max(1, n - 1)) >= cfg.pressure_endgame_frac
                if (
                    stream_run >= cfg.pressure_stream_min_len
                    or endgame
                    or same_key_run >= cfg.pressure_streak_threshold
                ):
                    pressure[i] = True
            # Among pressured objects, roll miss at an elevated rate so the
            # overall miss rate converges on miss_rate_target.
            n_pressure = int(pressure.sum())
            if n_pressure > 0:
                target_hits = int(round(cfg.miss_rate_target * n))
                # Clamp target to what pressure region can absorb.
                target_hits = min(target_hits, n_pressure)
                # Roll per-pressured-click so hits land ~uniformly in the region.
                p = min(1.0, target_hits / n_pressure)
                roll = rng.random(n) < p
                miss_mask = pressure & roll

        if miss_mask.any():
            # Shift the click time late — click lands outside hit window.
            shifts = np.where(miss_mask, cfg.miss_shift_ms, 0.0)
            df["offset_ms"] = df["offset_ms"].to_numpy() + shifts
            df["click_t"] = df["ho_t"].to_numpy() + df["offset_ms"].to_numpy()
            # Mark outcome as miss on the injected ones.
            outcomes = df["outcome"].tolist()
            for i in np.where(miss_mask)[0]:
                outcomes[i] = "miss"
            df["outcome"] = outcomes

    return df


def parse_rules_arg(spec: Optional[str]) -> RuleConfig:
    """Parse a comma-separated k=v string into a RuleConfig.

    Example: "alt_gap_threshold_ms=80,preferred_key=K1,max_same_key_streak=25"
    Unknown keys raise ValueError so typos surface instead of being silently ignored.
    """
    cfg = RuleConfig()
    if not spec:
        return cfg
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"rules arg missing '=': {part!r}")
        k, v = part.split("=", 1)
        k = k.strip()
        v = v.strip()
        if not hasattr(cfg, k):
            raise ValueError(f"unknown rule key: {k!r}")
        current = getattr(cfg, k)
        # Type-coerce by looking at the default
        if current is None:
            # Could be preferred_key (Optional[str]) — try string
            setattr(cfg, k, v if v.lower() not in ("none", "null", "") else None)
        elif isinstance(current, bool):
            setattr(cfg, k, v.lower() in ("1", "true", "yes", "on"))
        elif isinstance(current, int) and not isinstance(current, bool):
            setattr(cfg, k, int(v))
        elif isinstance(current, float):
            setattr(cfg, k, float(v))
        else:
            # string field (e.g. miss_mode, preferred_key already set)
            setattr(cfg, k, v if v.lower() not in ("none", "null", "") else None)
    return cfg
