"""Focused: find the 66 missing clicks and understand why they are missing."""
from __future__ import annotations

import json
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
from osrparse import Replay
from src.data.pull import REPLAY_DIR
from src.data.parse import parse_replay_bytes
from src.models.datasets import load_all_shards
from src.models.generator import generate as model_generate
from src.data.beatmap import parse_osu
from src.models.rules import RuleConfig

PROJECT = pathlib.Path(__file__).resolve().parents[1]
MODELS = PROJECT / "models"
OUT_DIR = PROJECT / "data" / "generated" / "replays"


def regen():
    bid = 954692
    df = load_all_shards()
    sub = df[df["beatmap_id"] == bid]
    top_user = int(sub.groupby("user_id").size().idxmax())
    bm = parse_osu(PROJECT / "data" / "raw" / "beatmaps" / f"{bid}.osu")
    cf = None; mods = 0
    for fn in REPLAY_DIR.glob(f"{bid}_{top_user}_*.bin"):
        p = parse_replay_bytes(fn.read_bytes())
        cf = [{"t": f.t, "x": f.x, "y": f.y} for f in p.frames]
        mods = int(fn.stem.split("_")[-1])
        break
    uhold = df[df["user_id"] == top_user]["hold_ms"].dropna()
    hold_params = {"mu": float(uhold.mean()), "sigma": float(uhold.std())}
    g = model_generate(bm, mods=mods, player_id=top_user,
                       timing_ckpt=MODELS/"timing_best.pt",
                       outcome_ckpt=MODELS/"outcome_best.pt",
                       key_ckpt=MODELS/"key_best.pt",
                       cursor_frames=cf, hold_params=hold_params,
                       seed=42, rules=RuleConfig(seed=42))
    return g, cf, mods


def main():
    gen_df, cursor_frames, mods = regen()

    r = Replay.from_path(str(OUT_DIR / "954692_model_default.osr"))
    frames = []
    cum = 0
    for ev in r.replay_data:
        if ev.time_delta == -12345: continue
        cum += ev.time_delta
        frames.append((cum, float(ev.x), float(ev.y), int(ev.keys)))

    # Extract press events
    osr_presses = []
    prev_k = 0
    for (t, x, y, k) in frames:
        for name, bit in (("K1", 4), ("K2", 8)):
            if (k & bit) and not (prev_k & bit):
                osr_presses.append((t, name))
        prev_k = k
    osr_ts = {name: [t for (t, n) in osr_presses if n == name] for name in ("K1", "K2")}

    # Model clicks
    model_clicks = [(float(row["click_t"]), str(row["key_type"]), int(i),
                     float(row["hold_ms"]))
                    for i, row in gen_df.iterrows()]
    model_clicks.sort(key=lambda c: c[0])

    print(f"model n_clicks={len(model_clicks)}  osr n_presses={len(osr_presses)}")
    print(f"missing={len(model_clicks) - len(osr_presses)}")

    # Walk model clicks in order. For each, see if there's an osr press within +/-5ms
    # of same key.
    missing = []
    k_pool = {"K1": list(osr_ts["K1"]), "K2": list(osr_ts["K2"])}
    for (ct, key, i, hold) in model_clicks:
        pool = k_pool[key]
        if not pool:
            missing.append({"i": i, "t": ct, "key": key, "hold": hold, "why": "pool empty"})
            continue
        arr = np.array(pool)
        d = np.abs(arr - ct)
        j = int(np.argmin(d))
        if d[j] <= 10:
            pool.pop(j)
        else:
            missing.append({"i": i, "t": ct, "key": key, "hold": hold,
                           "nearest_osr_dt": float(arr[j] - ct)})

    print(f"\nmissing clicks: {len(missing)}")
    print("first 30:")
    for m in missing[:30]:
        # find nearest model click of either key just before
        prev = None
        for (ct, key, i, hold) in model_clicks:
            if ct >= m["t"]: break
            prev = (ct, key, hold)
        nxt = None
        for (ct, key, i, hold) in model_clicks:
            if ct > m["t"]:
                nxt = (ct, key, hold); break
        gap_prev = (m["t"] - prev[0]) if prev else None
        gap_next = (nxt[0] - m["t"]) if nxt else None
        print(f"  i={m['i']:4d} key={m['key']} t={m['t']:.2f} hold={m['hold']:.1f} "
              f"gap_prev={gap_prev} gap_next={gap_next}")

    # Specifically analyze: are missing clicks clustered when gap to previous
    # model click is small (< 20ms)?
    small_gap_missing = 0
    close_pair_both_missing = 0
    for m in missing:
        prev = None
        for (ct, key, i, hold) in model_clicks:
            if ct >= m["t"]: break
            prev = (ct, key, hold)
        if prev and (m["t"] - prev[0]) < 20:
            small_gap_missing += 1
    print(f"\nof missing, # with prev-gap < 20ms: {small_gap_missing}/{len(missing)}")

    # Also: analyze hold distribution for missing vs not-missing
    all_holds = [h for (_, _, _, h) in model_clicks]
    missing_holds = [m["hold"] for m in missing]
    print(f"\nholds:  all mean={np.mean(all_holds):.1f}  missing mean={np.mean(missing_holds):.1f}")

    # Dump full missing list
    out = {"n_missing": len(missing), "missing": missing}
    (PROJECT / "data/generated/debug/missing_clicks.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
