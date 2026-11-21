"""Systematic diagnostic for the .osr emitter bug.

Follows the 7-step debug plan from the parent agent:
  1. Round-trip check:  _wrap_human.osr vs source .bin frames.
  2. Direct model-vs-osr click diff on 954692_model_default.osr.
  3. Cursor sync check: fraction of clicks where cursor is in/out of circle.
  4. Lead-in/time-origin check.
  5. Key encoding check.
  6. Hold duration check.
  7. Identify ONE root cause.

Outputs:
  data/generated/debug/model_clicks_954692.jsonl
  data/generated/debug/osr_clicks_954692.jsonl
  data/generated/debug/debug_report.json
  stdout summary.
"""
from __future__ import annotations

import json
import lzma
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import hashlib
from typing import List, Dict, Any, Tuple

import numpy as np
import pandas as pd
from osrparse import Replay

from src.data.parse import parse_replay_bytes, parse_replay_text, extract_key_events
from src.data.beatmap import parse_osu
from src.data.pull import REPLAY_DIR
from src.data.emit import ClickEvent, clicks_from_generator_df, write_osr
from src.data.features import _circle_radius_px, _effective_od, _speed_mult
from src.models.datasets import load_all_shards
from src.models.generator import generate as model_generate
from src.models.rules import RuleConfig


PROJECT = pathlib.Path(__file__).resolve().parents[1]
MODELS = PROJECT / "models"
DEBUG = PROJECT / "data" / "generated" / "debug"
DEBUG.mkdir(parents=True, exist_ok=True)
OUT_DIR = PROJECT / "data" / "generated" / "replays"


def _dump_jsonl(path: pathlib.Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=float) + "\n")


def _osr_cum_frames(osr_path: pathlib.Path) -> List[Dict[str, Any]]:
    """Read a .osr and return cumulative-time frames [{t, x, y, keys}, ...].

    Excludes the RNG seed trailer (time_delta == -12345).
    """
    r = Replay.from_path(str(osr_path))
    frames = []
    cum = 0
    for ev in r.replay_data:
        if ev.time_delta == -12345:
            continue
        cum += ev.time_delta
        frames.append({"t": cum, "x": float(ev.x), "y": float(ev.y), "keys": int(ev.keys)})
    return frames


def _extract_clicks_from_frames(frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Scan frames for press events: K1 bit (4) or K2 bit (8) transitioning 0->1.

    Returns [{t_down, t_up, key, x_down, y_down, hold_ms}, ...].
    """
    clicks = []
    prev_k = 0
    actives = {}  # name -> (t_down, x, y)
    for f in frames:
        cur = f["keys"]
        for name, bit in (("K1", 4), ("K2", 8)):
            was = bool(prev_k & bit)
            now = bool(cur & bit)
            if now and not was:
                actives[name] = (f["t"], f["x"], f["y"])
            elif was and not now and name in actives:
                td, xd, yd = actives.pop(name)
                clicks.append({
                    "t_down": td, "t_up": f["t"],
                    "hold_ms": f["t"] - td,
                    "key": name,
                    "x_down": xd, "y_down": yd,
                })
        prev_k = cur
    # unclosed
    if frames and actives:
        last_t = frames[-1]["t"]
        for name, (td, xd, yd) in actives.items():
            clicks.append({
                "t_down": td, "t_up": last_t,
                "hold_ms": last_t - td,
                "key": name,
                "x_down": xd, "y_down": yd,
            })
    clicks.sort(key=lambda c: c["t_down"])
    return clicks


def step1_roundtrip() -> dict:
    """Compare _wrap_human.osr output frames to the source .bin frames."""
    wrapped = OUT_DIR / "954692_human_wrapped.osr"
    if not wrapped.exists():
        return {"ok": False, "err": "no wrapped file"}
    # Find source bin
    srcs = list(REPLAY_DIR.glob("954692_*_72.bin"))
    if not srcs:
        return {"ok": False, "err": "no 954692 _72 bin"}
    src = srcs[0]

    # Parse both with the SAME parser logic we parse real replays with.
    src_text = lzma.decompress(src.read_bytes()).decode("ascii", errors="replace")
    src_parsed = parse_replay_text(src_text)
    src_frames = [(f.t, f.x, f.y, f.keys) for f in src_parsed.frames]

    # For the wrapped .osr, pull the raw replay_data (lzma blob is inside the
    # .osr). osrparse exposes the decompressed event list directly.
    wrapped_frames = _osr_cum_frames(wrapped)

    # Compare counts
    out = {
        "source_bin": str(src),
        "n_src_frames": len(src_frames),
        "n_wrapped_frames_total": len(wrapped_frames),
    }

    # Align: source parser strips parked-cursor prelude; wrapped frames should
    # include everything. For diff, re-parse the source like a FRESH replay
    # (same code path) and scan the wrapped frames for the same cursor stream.
    #
    # Compare first 100 non-trivial frames
    n = min(100, len(src_frames), len(wrapped_frames))

    # What the wrapped stream has as its first N (post-parking) frames
    # Skip parking frames in wrapped
    i = 0
    while i < len(wrapped_frames) and wrapped_frames[i]["x"] == 256.0 and wrapped_frames[i]["y"] == -500.0:
        i += 1
    wrapped_post = wrapped_frames[i:]

    # Check first-N sample alignment by t
    sample = []
    for k in range(min(20, len(src_frames), len(wrapped_post))):
        s_t, s_x, s_y, s_k = src_frames[k]
        w = wrapped_post[k]
        sample.append({
            "idx": k,
            "src_t": s_t, "src_xy": (s_x, s_y), "src_k": s_k,
            "wrapped_t": w["t"], "wrapped_xy": (w["x"], w["y"]), "wrapped_k": w["keys"],
            "t_diff": w["t"] - s_t,
        })
    out["first20_sample"] = sample
    # Count how many of the first 500 match exactly on (t, x, y, keys)
    N = min(500, len(src_frames), len(wrapped_post))
    exact = 0
    for k in range(N):
        s = src_frames[k]; w = wrapped_post[k]
        if s[0] == w["t"] and s[1] == w["x"] and s[2] == w["y"] and s[3] == w["keys"]:
            exact += 1
    out["exact_matches_in_first_500"] = exact
    out["pct_exact"] = exact / N if N else 0
    return out


def step2_6_main_debug() -> dict:
    """Instrument generator, dump model clicks, dump osr clicks, diff them."""
    bid = 954692

    # ---- Regenerate the model output (same seed/params as generate_replays.py)
    df = load_all_shards()
    sub = df[df["beatmap_id"] == bid]
    if sub.empty:
        return {"err": f"no training clicks for {bid}"}
    top_user = int(sub.groupby("user_id").size().idxmax())
    osu_file = PROJECT / "data" / "raw" / "beatmaps" / f"{bid}.osu"
    bm = parse_osu(osu_file)

    # Cursor source (same as generate_replays.py)
    cf = None
    mods_int = 0
    for fn in REPLAY_DIR.glob(f"{bid}_{top_user}_*.bin"):
        parsed_src = parse_replay_bytes(fn.read_bytes())
        cf = [{"t": f.t, "x": f.x, "y": f.y} for f in parsed_src.frames]
        try:
            mods_int = int(fn.stem.split("_")[-1])
        except Exception:
            mods_int = 0
        break
    if cf is None:
        return {"err": "no cursor source"}

    uhold = df[df["user_id"] == top_user]["hold_ms"].dropna()
    if len(uhold) < 20:
        uhold = df["hold_ms"].dropna()
    hold_params = {"mu": float(uhold.mean()), "sigma": float(uhold.std())}

    gen_df = model_generate(
        bm,
        mods=mods_int,
        player_id=top_user,
        timing_ckpt=MODELS / "timing_best.pt",
        outcome_ckpt=MODELS / "outcome_best.pt",
        key_ckpt=MODELS / "key_best.pt",
        cursor_frames=cf,
        hold_params=hold_params,
        seed=42,
        rules=RuleConfig(seed=42),
    )

    # Dump model-internal clicks
    model_clicks_rows = []
    for i, row in gen_df.iterrows():
        model_clicks_rows.append({
            "object_index": int(i),
            "ho_t": float(row["ho_t"]),
            "ho_x": float(row["ho_x"]),
            "ho_y": float(row["ho_y"]),
            "model_click_time_ms": float(row["click_t"]),
            "model_key": str(row["key_type"]),
            "hold_ms": float(row["hold_ms"]),
            "cursor_at_ho_x": float(row["click_x"]),
            "cursor_at_ho_y": float(row["click_y"]),
            "outcome": str(row["outcome"]),
        })
    _dump_jsonl(DEBUG / "model_clicks_954692.jsonl", model_clicks_rows)

    # ---- Parse the EXISTING .osr file and extract clicks
    osr_path = OUT_DIR / "954692_model_default.osr"
    r = Replay.from_path(str(osr_path))
    frames = _osr_cum_frames(osr_path)
    osr_clicks_raw = _extract_clicks_from_frames(frames)
    _dump_jsonl(DEBUG / "osr_clicks_954692.jsonl",
                [{"object_index": i, **c} for i, c in enumerate(osr_clicks_raw)])

    # ---- Diff: match each model click to the nearest osr click within +/- 20ms
    model_ts = np.array([r["model_click_time_ms"] for r in model_clicks_rows])
    osr_ts = np.array([c["t_down"] for c in osr_clicks_raw])
    matched_within_1ms = 0
    matched_within_5ms = 0
    matched_within_20ms = 0
    unmatched_model = 0
    shifted = []
    used_osr = set()
    for m_t in model_ts:
        if len(osr_ts) == 0:
            unmatched_model += 1
            continue
        d = np.abs(osr_ts - m_t)
        j = int(np.argmin(d))
        if d[j] <= 1: matched_within_1ms += 1
        if d[j] <= 5: matched_within_5ms += 1
        if d[j] <= 20:
            matched_within_20ms += 1
            used_osr.add(j)
            shifted.append(osr_ts[j] - m_t)
        else:
            unmatched_model += 1
    extra_osr = len(osr_ts) - len(used_osr)

    # ---- Key encoding check
    key_values = sorted({c["key"] for c in osr_clicks_raw})
    raw_keys_int = sorted({f["keys"] for f in frames if f["keys"] != 0})
    unexpected_keys = [k for k in raw_keys_int if k not in (5, 10, 15, 16, 21, 26, 31)]

    # ---- Hold duration check
    holds = np.array([c["hold_ms"] for c in osr_clicks_raw], dtype=float)
    zero_holds = int(np.sum(holds <= 0))
    neg_holds = int(np.sum(holds < 0))

    # ---- Cursor sync check: for each osr click, find the hit object nearest in
    # time, interpolate cursor at click time, measure distance.
    eod = _effective_od(bm.od, mods_int)
    radius = _circle_radius_px(bm.cs, mods_int)
    hw50 = 200 - 10 * eod

    clickable = [h for h in bm.hit_objects if h.kind in ("circle", "slider")]
    ho_ts = np.array([h.t for h in clickable])
    ho_xs = np.array([h.x for h in clickable])
    ho_ys = np.array([h.y for h in clickable])
    if mods_int & 16:  # HR
        ho_ys = 384.0 - ho_ys

    # Interpolate cursor pos
    frame_ts = np.array([f["t"] for f in frames], dtype=float)
    frame_xs = np.array([f["x"] for f in frames], dtype=float)
    frame_ys = np.array([f["y"] for f in frames], dtype=float)
    # Sort by t (should already be)
    order = np.argsort(frame_ts, kind="stable")
    frame_ts = frame_ts[order]; frame_xs = frame_xs[order]; frame_ys = frame_ys[order]

    cursor_dists = []
    in_circle = 0
    in_hit50 = 0
    off_screen = 0
    for c in osr_clicks_raw:
        t = c["t_down"]
        if t < frame_ts[0] or t > frame_ts[-1]:
            cursor_dists.append(None)
            off_screen += 1
            continue
        cx = float(np.interp(t, frame_ts, frame_xs))
        cy = float(np.interp(t, frame_ts, frame_ys))
        # nearest ho
        j = int(np.argmin(np.abs(ho_ts - t)))
        d = float(np.hypot(cx - ho_xs[j], cy - ho_ys[j]))
        cursor_dists.append({"dist": d, "ho_dt": float(t - ho_ts[j]), "ho_idx": j})
        if d <= radius: in_circle += 1
        if d <= radius * 1.5 and abs(t - ho_ts[j]) <= hw50: in_hit50 += 1

    # ---- Lead-in check
    cum = 0
    first_10_cum = []
    for ev in r.replay_data[:10]:
        cum += ev.time_delta
        first_10_cum.append({"td": int(ev.time_delta), "cum_t": int(cum),
                             "xy": (float(ev.x), float(ev.y)), "k": int(ev.keys)})

    first_ho_t = int(clickable[0].t) if clickable else -1
    first_click_t = int(osr_clicks_raw[0]["t_down"]) if osr_clicks_raw else -1
    first_model_t = int(model_clicks_rows[0]["model_click_time_ms"]) if model_clicks_rows else -1

    return {
        "bid": bid,
        "n_hit_objects": int(len(clickable)),
        "n_model_clicks": int(len(model_clicks_rows)),
        "n_osr_clicks": int(len(osr_clicks_raw)),
        "diff_model_vs_osr": {
            "matched_within_1ms": int(matched_within_1ms),
            "matched_within_5ms": int(matched_within_5ms),
            "matched_within_20ms": int(matched_within_20ms),
            "unmatched_model": int(unmatched_model),
            "extra_osr_clicks": int(extra_osr),
            "mean_shift_ms": float(np.mean(shifted)) if shifted else None,
            "median_shift_ms": float(np.median(shifted)) if shifted else None,
            "shift_ms_std": float(np.std(shifted)) if shifted else None,
        },
        "key_encoding": {
            "key_values_seen": key_values,
            "raw_keys_int_seen": raw_keys_int,
            "unexpected_keys": unexpected_keys,
        },
        "hold_durations": {
            "n_total": int(len(holds)),
            "mean_ms": float(np.mean(holds)) if len(holds) else None,
            "min_ms": float(np.min(holds)) if len(holds) else None,
            "max_ms": float(np.max(holds)) if len(holds) else None,
            "n_zero": zero_holds,
            "n_negative": neg_holds,
        },
        "cursor_sync": {
            "n_clicks": int(len(osr_clicks_raw)),
            "n_in_circle": int(in_circle),
            "pct_in_circle": float(in_circle / max(1, len(osr_clicks_raw))),
            "n_in_hit50_window": int(in_hit50),
            "pct_hittable": float(in_hit50 / max(1, len(osr_clicks_raw))),
            "n_offscreen_time": int(off_screen),
            "radius": float(radius),
            "hw50_ms": float(hw50),
        },
        "lead_in": {
            "first_10_frames": first_10_cum,
            "first_hit_object_t_ms": first_ho_t,
            "first_osr_click_t_ms": first_click_t,
            "first_model_click_t_ms": first_model_t,
            "ho_vs_osr_click_delta": first_click_t - first_ho_t,
            "ho_vs_model_click_delta": first_model_t - first_ho_t,
        },
    }


def main() -> None:
    print("=== STEP 1: human-replay round-trip ===")
    s1 = step1_roundtrip()
    print(json.dumps(s1, indent=2, default=str)[:2000])
    print()

    print("=== STEPS 2-6: generator vs osr diff + cursor + keys + holds + lead-in ===")
    s2 = step2_6_main_debug()
    print(json.dumps(s2, indent=2, default=str))

    (DEBUG / "debug_report.json").write_text(
        json.dumps({"step1": s1, "step2to6": s2}, indent=2, default=str)
    )
    print()
    print(f"\n>>> Wrote debug report to {DEBUG / 'debug_report.json'}")


if __name__ == "__main__":
    main()
