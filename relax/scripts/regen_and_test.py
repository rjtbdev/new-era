"""Regenerate 954692_model_default.osr at three jitter values; run Circleguard on each."""
from __future__ import annotations

import json
import os
import pathlib
import sys
from typing import Any, Dict

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=pathlib.Path(__file__).resolve().parents[1] / ".env")
except Exception:
    pass

import numpy as np
import pandas as pd

from src.data.beatmap import parse_osu
from src.data.emit import ClickEvent, clicks_from_generator_df, write_osr
from src.data.parse import parse_replay_bytes
from src.data.pull import REPLAY_DIR
from src.models.datasets import load_all_shards
from src.models.generator import generate as model_generate
from src.models.rules import RuleConfig

from circleguard import Circleguard, ReplayPath

PROJECT = pathlib.Path(__file__).resolve().parents[1]
MODELS = PROJECT / "models"
OUT_DIR = PROJECT / "data" / "generated" / "replays"
BID = 954692


def pick_cursor_source(bid: int, user_id: int):
    for fn in REPLAY_DIR.glob(f"{bid}_{user_id}_*.bin"):
        parsed = parse_replay_bytes(fn.read_bytes())
        cf = [{"t": f.t, "x": f.x, "y": f.y} for f in parsed.frames]
        try:
            mods = int(fn.stem.split("_")[-1])
        except Exception:
            mods = 0
        return cf, mods
    raise FileNotFoundError(f"no cursor source found for bid={bid} user={user_id}")


def gen_one(df: pd.DataFrame, jitter_ms: float) -> pathlib.Path:
    sub = df[df["beatmap_id"] == BID]
    top_user = int(sub.groupby("user_id").size().idxmax())
    bm = parse_osu(PROJECT / "data" / "raw" / "beatmaps" / f"{BID}.osu")
    cursor_frames, mods_int = pick_cursor_source(BID, top_user)
    uhold = df[df["user_id"] == top_user]["hold_ms"].dropna()
    if len(uhold) < 20:
        uhold = df["hold_ms"].dropna()
    hold_params = {"mu": float(uhold.mean()), "sigma": float(uhold.std())}

    rules = RuleConfig(seed=42, jitter_std_ms=jitter_ms)
    gen_df = model_generate(
        bm,
        mods=mods_int,
        player_id=top_user,
        timing_ckpt=MODELS / "timing_best.pt",
        outcome_ckpt=MODELS / "outcome_best.pt",
        key_ckpt=MODELS / "key_best.pt",
        cursor_frames=cursor_frames,
        hold_params=hold_params,
        seed=42,
        rules=rules,
    )
    clicks, judgments = clicks_from_generator_df(gen_df)

    meta_path = PROJECT / "data" / "raw" / "beatmaps" / f"{BID}.json"
    beatmap_md5 = str(json.loads(meta_path.read_text()).get("checksum", "0" * 32))

    tag = f"default_jitter{jitter_ms}".replace(".", "p")
    out_path = OUT_DIR / f"{BID}_model_{tag}.osr"
    write_osr(
        out_path,
        beatmap_md5=beatmap_md5,
        player_name=f"neweraR_{tag}",
        mods_int=mods_int,
        click_events=clicks,
        cursor_frames=cursor_frames,
        judgments=judgments,
    )
    return out_path


def frametime_stats(osr_path: pathlib.Path) -> Dict[str, Any]:
    from osrparse import Replay
    r = Replay.from_path(str(osr_path))
    deltas = []
    for e in r.replay_data:
        if e.time_delta == -12345:
            continue
        if e.time_delta >= 0:
            deltas.append(e.time_delta)
    a = np.array(deltas)
    if len(a) == 0:
        return {}
    return {
        "n_frames": int(len(a)),
        "median": float(np.median(a)),
        "p25": float(np.percentile(a, 25)),
        "p75": float(np.percentile(a, 75)),
        "frac_lt6": float((a < 6).mean()),
        "frac_lt13": float((a < 13).mean()),
    }


def cg_on(osr_path: pathlib.Path, cg: Circleguard) -> Dict[str, Any]:
    r = ReplayPath(str(osr_path))
    out: Dict[str, Any] = {}
    cg.load(r)
    out["count_300"] = r.count_300
    out["count_100"] = r.count_100
    out["count_50"] = r.count_50
    out["count_miss"] = r.count_miss
    try:
        out["ur"] = cg.ur(r)
    except Exception as e:
        out["ur_err"] = repr(e)
    try:
        out["frametime_median"] = cg.frametime(r)
    except Exception as e:
        out["frametime_err"] = repr(e)
    try:
        snaps = cg.snaps(r)
        out["snaps"] = len(snaps)
    except Exception as e:
        out["snaps_err"] = repr(e)
    return out


def main():
    api_key = os.environ.get("OSU_API_V1_KEY")
    if not api_key:
        print("no API key", file=sys.stderr)
        sys.exit(1)

    df = load_all_shards()
    cg = Circleguard(api_key)

    results = {}
    for j in (0.0, 1.0, 2.0):
        print(f"[regen] jitter={j} ...", flush=True)
        p = gen_one(df, j)
        ft = frametime_stats(p)
        print(f"  file: {p}", flush=True)
        print(f"  frametime: {ft}", flush=True)
        cg_stats = cg_on(p, cg)
        print(f"  cg: {cg_stats}", flush=True)
        results[j] = {
            "path": str(p),
            "frametime": ft,
            "cg": cg_stats,
        }

    out_json = PROJECT / "data" / "regen_results.json"
    out_json.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nwrote {out_json}", flush=True)


if __name__ == "__main__":
    main()
