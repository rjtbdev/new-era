"""Generate .osr replay files from the trained model for a hand-picked set of
held-out beatmaps.

Produces, per beatmap:
  <bid>_model_default.osr — default rules (streak cap=25, no miss, no jitter)
  <bid>_model_human.osr   — "human-flavor" rules (jitter 1.5ms, miss 1.5%, pressure)

Both reuse one reference human's cursor trajectory (we don't have a cursor model).

Also runs a round-trip osrparse read of each written file and prints a sanity
report.
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.data.beatmap import parse_osu
from src.data.emit import ClickEvent, clicks_from_generator_df, write_osr
from src.data.parse import parse_replay_bytes
from src.data.pull import REPLAY_DIR
from src.models.datasets import load_all_shards
from src.models.generator import generate as model_generate
from src.models.rules import RuleConfig


PROJECT = pathlib.Path(__file__).resolve().parents[1]
MODELS = PROJECT / "models"
OUT_DIR = PROJECT / "data" / "generated" / "replays"


HELD_OUT = [954692, 1059894, 2449781]


def _load_bmap_meta(bid: int) -> Dict:
    meta_path = PROJECT / "data" / "raw" / "beatmaps" / f"{bid}.json"
    return json.loads(meta_path.read_text()) if meta_path.exists() else {}


def _pick_cursor_source(bid: int, user_id: int) -> tuple[list[dict], int]:
    """Load a real human replay's cursor trajectory for this map.

    Returns (cursor_frames, mods_int). mods_int is read from the filename
    convention: {bid}_{user}_{mods}.bin.
    """
    for fn in REPLAY_DIR.glob(f"{bid}_{user_id}_*.bin"):
        parsed = parse_replay_bytes(fn.read_bytes())
        cf = [{"t": f.t, "x": f.x, "y": f.y} for f in parsed.frames]
        try:
            mods = int(fn.stem.split("_")[-1])
        except Exception:
            mods = 0
        return cf, mods
    raise FileNotFoundError(f"no cursor source found for bid={bid} user={user_id}")


def _run_one(
    bid: int,
    *,
    df: pd.DataFrame,
    tag: str,
    rules: RuleConfig,
) -> dict:
    sub = df[df["beatmap_id"] == bid]
    if sub.empty:
        return {"bid": bid, "skipped": "no clicks in dataset"}
    counts = sub.groupby("user_id").size()
    top_user = int(counts.idxmax())

    osu_file = PROJECT / "data" / "raw" / "beatmaps" / f"{bid}.osu"
    bm = parse_osu(osu_file)

    cursor_frames, mods_int = _pick_cursor_source(bid, top_user)

    # hold_params: use user-specific hold distribution (same as run_eval does)
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
        cursor_frames=cursor_frames,
        hold_params=hold_params,
        seed=42,
        rules=rules,
    )
    if gen_df.empty:
        return {"bid": bid, "skipped": "generator produced 0 rows"}

    clicks, judgments = clicks_from_generator_df(gen_df)

    meta = _load_bmap_meta(bid)
    beatmap_md5 = str(meta.get("checksum", "0" * 32))

    # Output path
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{bid}_model_{tag}.osr"

    player_name = f"neweraR_{tag}"

    write_osr(
        out_path,
        beatmap_md5=beatmap_md5,
        player_name=player_name,
        mods_int=mods_int,
        click_events=clicks,
        cursor_frames=cursor_frames,
        judgments=judgments,
    )

    # Round-trip verify
    from osrparse import Replay
    r = Replay.from_path(str(out_path))
    verify = {
        "md5_ok": r.beatmap_hash == beatmap_md5,
        "username_ok": r.username == player_name,
        "n_events": len(r.replay_data),
        "counts_ok": (
            r.count_300 == judgments["300"]
            and r.count_100 == judgments["100"]
            and r.count_50 == judgments["50"]
            and r.count_miss == judgments["miss"]
        ),
        "mods_bits": int(r.mods),
        "mods_match": int(r.mods) == mods_int,
    }
    # Compute cum-t range of the replay and verify it covers the map
    cum = 0
    t_min = None
    t_max = None
    for e in r.replay_data:
        cum += e.time_delta
        if e.time_delta == -12345:
            continue
        if t_min is None or cum < t_min: t_min = cum
        if t_max is None or cum > t_max: t_max = cum
    verify["t_min_ms"] = t_min
    verify["t_max_ms"] = t_max
    verify["map_last_t_ms"] = int(max(h.t for h in bm.hit_objects))
    verify["covers_map"] = t_max is not None and t_max >= verify["map_last_t_ms"]

    return {
        "bid": bid,
        "tag": tag,
        "out": str(out_path),
        "mods_int": mods_int,
        "cursor_source_user": top_user,
        "n_clicks": len(clicks),
        "judgments": judgments,
        "verify": verify,
    }


def main() -> None:
    df = load_all_shards()
    if df.empty:
        print("no data")
        return

    default_rules = RuleConfig(seed=42)
    human_rules = RuleConfig(
        seed=42,
        jitter_std_ms=1.5,
        miss_rate_target=0.015,
        miss_mode="pressure",
        max_same_key_streak=25,
    )

    reports: List[dict] = []
    for bid in HELD_OUT:
        print(f"[gen] {bid} default ...")
        reports.append(_run_one(bid, df=df, tag="default", rules=default_rules))
        print(f"[gen] {bid} human  ...")
        reports.append(_run_one(bid, df=df, tag="human", rules=human_rules))

    (PROJECT / "data" / "generated" / "generation_report.json").write_text(
        json.dumps(reports, indent=2, default=str)
    )
    for r in reports:
        if "skipped" in r:
            print(f"  [skip] {r['bid']}: {r['skipped']}")
            continue
        v = r["verify"]
        print(
            f"  [ok] {r['bid']} ({r['tag']}): {r['out']}\n"
            f"       mods={r['mods_int']} clicks={r['n_clicks']} "
            f"judg={r['judgments']}\n"
            f"       verify: md5={v['md5_ok']} counts={v['counts_ok']} "
            f"mods={v['mods_match']} n_events={v['n_events']} "
            f"t=[{v['t_min_ms']}, {v['t_max_ms']}]ms (map ends at {v['map_last_t_ms']}ms, "
            f"covers={v['covers_map']})"
        )


if __name__ == "__main__":
    main()
