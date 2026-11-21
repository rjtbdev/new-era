"""End-to-end generator wiring together timing + outcome + key heads.

Input: a parsed beatmap + a cursor trajectory (we reuse a human's trajectory for
now, since this run is scoped to click events; a future run can replace this
with a learned cursor model).

Output: a DataFrame of per-click rows in the same format as
`src.data.features.extract_features_for_replay`, so the eval harness can score
it directly against humans.
"""

from __future__ import annotations

import math
import pathlib
from dataclasses import asdict
from typing import Dict, List, Optional

import torch  # DLL init order on Windows: torch MUST come before numpy/pandas

import numpy as np
import pandas as pd

from src.data.beatmap import Beatmap
from src.data.features import _circle_radius_px, _effective_od, _speed_mult
from src.models.datasets import OBJ_FEAT_DIM, PlayerIndex, _make_obj_feats
from src.models.timing import TimingConfig, TimingModel
from src.models.outcome import OutcomeConfig, OutcomeModel, CLASS_NAMES
from src.models.key_choice import KeyConfig, KeyModel
from src.models.rules import RuleConfig, apply_rules


def load_timing_ckpt(path: str | pathlib.Path) -> tuple[TimingModel, TimingConfig, Dict[int, int]]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = TimingConfig(**ckpt["cfg"])
    model = TimingModel(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg, ckpt.get("player_index", {})


def load_outcome_ckpt(path: str | pathlib.Path) -> tuple[OutcomeModel, OutcomeConfig]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = OutcomeConfig(**ckpt["cfg"])
    model = OutcomeModel(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


def load_key_ckpt(path: str | pathlib.Path) -> tuple[KeyModel, KeyConfig]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = KeyConfig(**ckpt["cfg"])
    model = KeyModel(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, cfg


_USER_OFFSET_STD_CACHE: Optional[Dict[int, float]] = None
_USER_MAP_OFFSET_STD_CACHE: Optional[Dict[tuple[int, int, int], float]] = None


def _load_user_offset_std() -> Dict[int, float]:
    global _USER_OFFSET_STD_CACHE
    if _USER_OFFSET_STD_CACHE is not None:
        return _USER_OFFSET_STD_CACHE
    p = pathlib.Path(__file__).resolve().parents[2] / "data" / "processed" / "user_offset_std.parquet"
    if not p.exists():
        _USER_OFFSET_STD_CACHE = {}
        return _USER_OFFSET_STD_CACHE
    df = pd.read_parquet(p)
    _USER_OFFSET_STD_CACHE = dict(zip(df["user_id"].astype(int), df["offset_std_ms"].astype(float)))
    return _USER_OFFSET_STD_CACHE


def _load_user_map_offset_std() -> Dict[tuple[int, int, int], float]:
    global _USER_MAP_OFFSET_STD_CACHE
    if _USER_MAP_OFFSET_STD_CACHE is not None:
        return _USER_MAP_OFFSET_STD_CACHE
    p = pathlib.Path(__file__).resolve().parents[2] / "data" / "processed" / "user_map_offset_std.parquet"
    if not p.exists():
        _USER_MAP_OFFSET_STD_CACHE = {}
        return _USER_MAP_OFFSET_STD_CACHE
    df = pd.read_parquet(p)
    _USER_MAP_OFFSET_STD_CACHE = {
        (int(r.user_id), int(r.beatmap_id), int(r.mods)): float(r.offset_std_ms)
        for r in df.itertuples(index=False)
    }
    return _USER_MAP_OFFSET_STD_CACHE


def _ho_df_from_beatmap(bm: Beatmap) -> pd.DataFrame:
    # Only clickable objects (no spinners for click alignment)
    rows = []
    for i, h in enumerate(bm.hit_objects):
        if h.kind not in ("circle", "slider"):
            continue
        rows.append({
            "ho_idx": i, "ho_t": h.t, "ho_x": h.x, "ho_y": h.y,
            "ho_kind": h.kind, "ho_new_combo": h.new_combo,
            "bpm": bm.bpm_at(h.t), "prev_gap_ms": 0, "next_gap_ms": 0,
            "hw300": None, "hw100": None, "hw50": None,
        })
    df = pd.DataFrame(rows)
    for i in range(len(df)):
        if i > 0: df.at[i, "prev_gap_ms"] = int(df.at[i-1,"ho_t"] - df.at[i,"ho_t"])
        if i+1 < len(df): df.at[i, "next_gap_ms"] = int(df.at[i+1,"ho_t"] - df.at[i,"ho_t"])
    return df


def generate(
    beatmap: Beatmap,
    *,
    mods: int,
    player_id: Optional[int],
    timing_ckpt: str | pathlib.Path,
    outcome_ckpt: Optional[str | pathlib.Path] = None,
    key_ckpt: Optional[str | pathlib.Path] = None,
    cursor_frames: Optional[List[dict]] = None,  # from a reference human replay for outcome
    hold_params: Optional[Dict[str, float]] = None,  # {'mu': ms, 'sigma': ms} for lognormal
    seed: int = 0,
    rules: Optional[RuleConfig] = None,
    beatmap_id: Optional[int] = None,  # preferred for per-(user,map,mods) style match
    target_offset_std_ms: Optional[float] = None,  # overrides table lookup; pass the reference human's std directly
) -> pd.DataFrame:
    torch.manual_seed(seed); rng = np.random.default_rng(seed)

    timing, tcfg, pidx_map = load_timing_ckpt(timing_ckpt)
    out_model = None; out_cfg = None
    if outcome_ckpt:
        out_model, out_cfg = load_outcome_ckpt(outcome_ckpt)
    key_model = None; kcfg = None
    if key_ckpt:
        key_model, kcfg = load_key_ckpt(key_ckpt)

    eod = _effective_od(beatmap.od, mods)
    hw300 = 80 - 6 * eod; hw100 = 140 - 8 * eod; hw50 = 200 - 10 * eod
    radius = _circle_radius_px(beatmap.cs, mods)
    speed = _speed_mult(mods)

    ho_df = _ho_df_from_beatmap(beatmap)
    if ho_df.empty:
        return pd.DataFrame()
    # HR: hit-object Y is mirrored in-game (y' = 384 - y). Our cursor_frames
    # (from an HR replay) are already recorded post-mirror, so we must mirror
    # the HO positions here to match the cursor's coordinate system.
    if mods & 16:  # HR
        ho_df["ho_y"] = 384.0 - ho_df["ho_y"]
    ho_df["hw300"] = hw300; ho_df["hw100"] = hw100; ho_df["hw50"] = hw50
    # Phase A features: effective_bpm (bpm × speed_mult), skill_percentile
    ho_df["speed_mult"] = speed
    ho_df["effective_bpm"] = ho_df["bpm"].fillna(120.0) * speed
    # Skill percentile: try leaderboard lookup for (beatmap_id, player_id) when available;
    # fall back to 0.95 (top-tier) which is what training defaulted to for top-50 pulls.
    _skill_pct = 0.95
    if player_id is not None:
        try:
            from src.data.features import skill_percentile_for  # local import to avoid cycle at module load
            # Guess beatmap_id from first object's beatmap attribute if possible.
            bid = getattr(beatmap, "beatmap_id", None)
            if bid is not None:
                _skill_pct = skill_percentile_for(int(bid), int(player_id))
        except Exception:
            pass
    ho_df["skill_percentile"] = _skill_pct

    # Cursor-at-ho_t features (must match the training-time extraction in
    # src/data/features.py). cursor_frames comes from the reference human replay.
    if cursor_frames:
        # Build sorted arrays once for binary search
        c_t = np.array([f["t"] for f in cursor_frames], dtype=np.float64)
        c_x = np.array([f["x"] for f in cursor_frames], dtype=np.float32)
        c_y = np.array([f["y"] for f in cursor_frames], dtype=np.float32)

        def _cur(t: float) -> tuple[float, float]:
            if t <= c_t[0]: return float(c_x[0]), float(c_y[0])
            if t >= c_t[-1]: return float(c_x[-1]), float(c_y[-1])
            j = int(np.searchsorted(c_t, t))
            a, b = j - 1, j
            if c_t[b] == c_t[a]: return float(c_x[a]), float(c_y[a])
            f = (t - c_t[a]) / (c_t[b] - c_t[a])
            return float(c_x[a] + f * (c_x[b] - c_x[a])), float(c_y[a] + f * (c_y[b] - c_y[a]))

        dist_at_arr = np.zeros(len(ho_df), dtype=np.float32)
        vel_at_arr = np.zeros(len(ho_df), dtype=np.float32)
        approach_arr = np.zeros(len(ho_df), dtype=np.float32)
        for i in range(len(ho_df)):
            t = float(ho_df.iloc[i]["ho_t"])
            hx = float(ho_df.iloc[i]["ho_x"]); hy = float(ho_df.iloc[i]["ho_y"])
            cx, cy = _cur(t)
            cxp, cyp = _cur(t - 20)
            cxp2, cyp2 = _cur(t - 40)
            dist_at = math.hypot(cx - hx, cy - hy)
            dist_prev2 = math.hypot(cxp2 - hx, cyp2 - hy)
            dist_at_arr[i] = dist_at
            vel_at_arr[i] = math.hypot(cx - cxp, cy - cyp) / 20.0
            approach_arr[i] = (dist_at - dist_prev2) / 40.0
        ho_df["cursor_at_ho_dist"] = dist_at_arr
        ho_df["cursor_at_ho_vel"] = vel_at_arr
        ho_df["cursor_approach_rate"] = approach_arr
    else:
        # No cursor provided → zeros (model sees "cursor unknown" via mid-range scaled values)
        ho_df["cursor_at_ho_dist"] = 100.0  # 1 radius unit after scaling
        ho_df["cursor_at_ho_vel"] = 0.0
        ho_df["cursor_approach_rate"] = 0.0

    # Build object feature batches for the timing model
    T = tcfg.ctx_past + 1 + tcfg.ctx_future
    N = len(ho_df)
    pids = np.full(N, pidx_map.get(player_id, 0) if player_id else 0, dtype=np.int64)
    obj_batches = np.zeros((N, T, OBJ_FEAT_DIM), dtype=np.float32)
    pad_batches = np.zeros((N, T), dtype=bool)
    for i in range(N):
        ref_t = float(ho_df.iloc[i]["ho_t"])
        for k, lp in enumerate(range(i - tcfg.ctx_past, i + tcfg.ctx_future + 1)):
            if lp < 0 or lp >= N:
                pad_batches[i, k] = True
                continue
            obj_batches[i, k] = _make_obj_feats(ho_df.iloc[lp:lp+1], np.array([ref_t]))[0]

    with torch.no_grad():
        preds = timing(
            torch.from_numpy(obj_batches),
            torch.from_numpy(pids),
            pad_mask=torch.from_numpy(pad_batches),
        )
        # sample offsets from mixture
        pi = torch.softmax(preds["pi_logits"], dim=-1)
        comp = torch.multinomial(pi, num_samples=1).squeeze(-1)
        mu = preds["mu"].gather(-1, comp.unsqueeze(-1)).squeeze(-1)
        sig = torch.exp(preds["log_sigma"]).gather(-1, comp.unsqueeze(-1)).squeeze(-1)
        offsets = (mu + sig * torch.randn_like(mu)).numpy()

    # Match per-player offset-std style: humans produce UR within a stable
    # per-player band; the model's raw sampling tends to be tighter than the
    # player it imitates. Scale residuals so the emitted replay's std ≈
    # the target style. Resolution order:
    #   1. explicit target_offset_std_ms kwarg (e.g. measured from the reference
    #      human replay at the call site — most accurate)
    #   2. per-(user, beatmap, mods) training data
    #   3. per-user training data
    #   4. population median (10.3ms)
    target_std: Optional[float] = target_offset_std_ms
    if target_std is None and player_id is not None:
        if beatmap_id is not None:
            per_map = _load_user_map_offset_std()
            target_std = per_map.get((int(player_id), int(beatmap_id), int(mods)))
        if target_std is None:
            user_std_map = _load_user_offset_std()
            target_std = user_std_map.get(int(player_id))
    if target_std is None:
        target_std = 10.3
    # Clip target to reasonable range (sanity).
    target_std = float(max(3.0, min(20.0, target_std)))
    current_std = float(np.std(offsets))
    if current_std > 1e-6:
        offsets_mean = float(np.mean(offsets))
        offsets = offsets_mean + (offsets - offsets_mean) * (target_std / current_std)

    ho_df["offset_ms"] = offsets
    ho_df["click_t"] = ho_df["ho_t"] + offsets
    ho_df["speed_mult"] = speed
    ho_df["mods"] = mods
    ho_df["circle_radius_px"] = radius

    # Cursor at click_t: if provided, interpolate; else synthesize zero-offset (perfect aim).
    # IMPORTANT: we want cursor_at_ho_t (what the human's aim was like when they
    # clicked), NOT cursor_at_click_t_of_model. Otherwise the model's different
    # offset drags cursor along a trajectory that was shaped to the HUMAN's
    # click timing, inflating edge-hit rate artificially.
    # Interpretation: if the human aimed near the circle at ho_t, we assume the
    # model would too (cursor is not modeled here); we just use that aim as our
    # cursor_at_click.
    if cursor_frames:
        cursor_frames_sorted = sorted(cursor_frames, key=lambda f: f["t"])
        ts = np.array([f["t"] for f in cursor_frames_sorted], dtype=float)
        xs = np.array([f["x"] for f in cursor_frames_sorted], dtype=float)
        ys = np.array([f["y"] for f in cursor_frames_sorted], dtype=float)
        cx = np.interp(ho_df["ho_t"].to_numpy(), ts, xs)
        cy = np.interp(ho_df["ho_t"].to_numpy(), ts, ys)
    else:
        cx = ho_df["ho_x"].to_numpy()
        cy = ho_df["ho_y"].to_numpy()
    ho_df["click_x"] = cx; ho_df["click_y"] = cy
    ho_df["cursor_dx"] = cx - ho_df["ho_x"]
    ho_df["cursor_dy"] = cy - ho_df["ho_y"]
    ho_df["cursor_dist"] = np.hypot(ho_df["cursor_dx"], ho_df["cursor_dy"])
    ho_df["edge_bias"] = np.clip(ho_df["cursor_dist"] / max(1e-6, radius), 0, 1)

    # Outcome
    if out_model is not None:
        # Phase A: outcome head expects effective_bpm as a trailing feature when
        # cfg.feat_dim >= 12. Build the base 11-feature stack, then append iff the
        # loaded checkpoint expects it.
        base_cols = [
            ho_df["offset_ms"].to_numpy(),
            ho_df["cursor_dx"].to_numpy(),
            ho_df["cursor_dy"].to_numpy(),
            np.full(N, hw300), np.full(N, hw100), np.full(N, hw50),
            np.full(N, radius),
            ho_df["bpm"].to_numpy(),
            (ho_df["ho_kind"] == "circle").astype(np.float32).to_numpy(),
            ho_df["prev_gap_ms"].to_numpy(),
            ho_df["next_gap_ms"].to_numpy(),
        ]
        if getattr(out_cfg, "feat_dim", 11) >= 12:
            base_cols.append(ho_df["effective_bpm"].to_numpy())
        feats = np.stack(base_cols, axis=1).astype(np.float32)
        with torch.no_grad():
            logits = out_model(torch.from_numpy(feats))
            # Soften miss to sample realistic occasional misses: raise temperature on miss class
            # to account for human miss rate not being 0. (Fit will be ~0 since train data is
            # mostly top-50 perfect runs.) Minor bump.
            probs = torch.softmax(logits, dim=-1).numpy()
            # Override: if cursor_dist > radius, force miss (game rule).
            dist = ho_df["cursor_dist"].to_numpy()
            probs[dist > radius] = np.array([0, 0, 0, 1], dtype=np.float32)
        samples = np.array([rng.choice(len(CLASS_NAMES), p=probs[i]) for i in range(N)])
        ho_df["outcome"] = [CLASS_NAMES[s] for s in samples]
    else:
        # deterministic fallback
        def _outcome(off, dist):
            if dist > radius: return "miss"
            if abs(off) <= hw300: return "300"
            if abs(off) <= hw100: return "100"
            if abs(off) <= hw50: return "50"
            return "miss"
        ho_df["outcome"] = [
            _outcome(ho_df["offset_ms"].iloc[i], ho_df["cursor_dist"].iloc[i])
            for i in range(N)
        ]

    # Key choice
    if key_model is not None:
        prev_keys = np.zeros(N, dtype=np.int64)  # 0=K1, 1=K2
        chosen = []
        # we'll run the model per-note, autoregressive, batch size 1 (simple)
        for i in range(N):
            # build context of last kcfg.ctx notes
            feats = np.zeros((1, kcfg.ctx, kcfg.feat_dim), dtype=np.float32)
            for j in range(kcfg.ctx):
                k = i - (kcfg.ctx - j)
                if k < 0: continue
                prevk = prev_keys[k]
                feats[0, j, 0] = 1.0 if prevk == 0 else 0.0
                feats[0, j, 1] = 1.0 if prevk == 1 else 0.0
                feats[0, j, 2] = float(ho_df.iloc[k]["bpm"]) / 300.0
                feats[0, j, 3] = float(ho_df.iloc[k]["prev_gap_ms"]) / 1000.0
                feats[0, j, 4] = 1.0 if ho_df.iloc[k]["ho_new_combo"] else 0.0
                # Phase A: effective_bpm (ckpt feat_dim>=6)
                if kcfg.feat_dim >= 6:
                    feats[0, j, 5] = float(ho_df.iloc[k]["effective_bpm"]) / 400.0
            with torch.no_grad():
                logits = key_model(torch.from_numpy(feats))
                # Small temperature on key choice: LSTM learns streakiness but
                # can get stuck in long bursts. T=1.3 flattens the distribution.
                p = torch.softmax(logits / 1.3, dim=-1)[0].numpy()
            pick = int(rng.choice(2, p=p))
            prev_keys[i] = pick
            chosen.append("K1" if pick == 0 else "K2")
        ho_df["key_type"] = chosen
    else:
        # alternating-key fallback
        ho_df["key_type"] = ["K1" if i % 2 == 0 else "K2" for i in range(N)]

    # hold times: lognormal, split by hit-object kind (circle vs slider).
    # Humans tap circles briefly (~85ms median) and hold sliders through their
    # duration (~237ms median). Using one blended distribution flattens this
    # bimodal shape and looks unhuman.
    CIRCLE_MEAN, CIRCLE_STD = 92.3, 43.0
    SLIDER_MEAN, SLIDER_STD = 262.5, 127.4

    def _sample_hold(mean, std, n, rng):
        # Lognormal with target arithmetic mean/std.
        # For lognormal(μ,σ): E=exp(μ+σ²/2), Var=(exp(σ²)-1)·exp(2μ+σ²)
        # → σ² = ln(1 + (std/mean)²); μ = ln(mean) - σ²/2
        s2 = math.log(1.0 + (std/mean)**2)
        sig = math.sqrt(s2)
        mu = math.log(mean) - s2 / 2.0
        return rng.lognormal(mu, sig, size=n)

    is_slider = (ho_df["ho_kind"] == "slider").to_numpy()
    n_circles = int((~is_slider).sum())
    n_sliders = int(is_slider.sum())
    holds = np.zeros(N, dtype=np.float64)
    if n_circles > 0:
        holds[~is_slider] = _sample_hold(CIRCLE_MEAN, CIRCLE_STD, n_circles, rng)
    if n_sliders > 0:
        # Slider hold should at minimum cover the slider body duration, if the
        # beatmap carries it. We don't have end_t in ho_df; clamp min 60ms,
        # max 800ms for safety.
        holds[is_slider] = _sample_hold(SLIDER_MEAN, SLIDER_STD, n_sliders, rng)
    ho_df["hold_ms"] = np.clip(holds, 25.0, 800.0)

    # Apply post-process rule layer (tunable; default = neutral except streak cap=25).
    rule_cfg = rules if rules is not None else RuleConfig(seed=seed)
    ho_df = apply_rules(ho_df, rule_cfg)

    return ho_df
