"""Generate model replay for bid=252236 (xi - Blue Zenith [FOUR DIMENSIONS]) HDDT,
using the top leaderboard player's cursor trajectory. Then run Circleguard.

Second DT confirmation test — the 954692 clean result shouldn't be a fluke.
"""
from __future__ import annotations
import pathlib, sys, json, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.api.v1 import OsuApiV1
from src.api.v2 import OsuApiV2
from src.data.beatmap import parse_osu
from src.data.emit import clicks_from_generator_df, write_osr
from src.data.parse import parse_replay_bytes
from src.models.generator import generate as model_generate
from src.models.rules import RuleConfig

PROJECT = pathlib.Path(__file__).resolve().parents[1]
BM_DIR = PROJECT / "data" / "raw" / "beatmaps"
REPLAY_DIR = PROJECT / "data" / "raw" / "replays"
OUT_DIR = PROJECT / "data" / "generated" / "replays"
MODELS = PROJECT / "models"

import json as _json
_PICK = _json.loads((pathlib.Path(__file__).resolve().parents[1] / "data" / "raw" / "dt_pick.json").read_text())
BID = _PICK["bid"]
PRESET_USER = _PICK["user_id"]
PRESET_MODS = _PICK["mods_bits"]

def download_beatmap_osu(bid: int, dst: pathlib.Path):
    import requests
    url = f"https://osu.ppy.sh/osu/{bid}"
    r = requests.get(url, headers={"User-Agent": "new-era-relax/0.1"}, timeout=30)
    r.raise_for_status()
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(r.content)
    return dst


def main():
    print(f"[gen] target bid={BID} — need HDDT human replay for cursor source")

    api2 = OsuApiV2()
    api1 = OsuApiV1()

    full = api2.get_beatmap(BID)
    beatmap_md5 = (full or {}).get("checksum", "0"*32)
    print(f"[gen] md5={beatmap_md5}")

    osu_path = BM_DIR / f"{BID}.osu"
    if not osu_path.exists():
        print(f"[gen] downloading {osu_path}")
        download_beatmap_osu(BID, osu_path)
    bm_parsed = parse_osu(osu_path)
    print(f"[gen] beatmap: {len(bm_parsed.hit_objects)} hit objects")

    # Use preset pick
    chosen_user = PRESET_USER
    chosen_mods = PRESET_MODS
    rp_path = REPLAY_DIR / f"{BID}_{chosen_user}_{chosen_mods}.bin"
    if not rp_path.exists() or rp_path.stat().st_size == 0:
        print(f"[gen] pulling preset replay user={chosen_user} mods={chosen_mods}")
        rp_bytes = api1.get_replay_bytes(beatmap_id=BID, user_id=chosen_user, mods=chosen_mods)
        if not rp_bytes:
            print("[gen] FATAL: preset replay unavailable from API")
            sys.exit(1)
        rp_path.parent.mkdir(parents=True, exist_ok=True)
        rp_path.write_bytes(rp_bytes)
        print(f"[gen] wrote {len(rp_bytes)} bytes → {rp_path}")

    parsed = parse_replay_bytes(rp_path.read_bytes())
    cursor_frames = [{"t": f.t, "x": f.x, "y": f.y} for f in parsed.frames]
    print(f"[gen] cursor: {len(cursor_frames)} frames, t=[{cursor_frames[0]['t']},{cursor_frames[-1]['t']}]ms")

    hold_params = {"mu": 155.0, "sigma": 50.0}
    rules = RuleConfig(seed=42, jitter_std_ms=1.0, max_same_key_streak=25, alt_gap_threshold_ms=80)
    print(f"[gen] rules: {rules}")

    gen_df = model_generate(
        bm_parsed,
        mods=chosen_mods,
        player_id=chosen_user,
        timing_ckpt=MODELS / "timing_best.pt",
        outcome_ckpt=MODELS / "outcome_best.pt",
        key_ckpt=MODELS / "key_best.pt",
        cursor_frames=cursor_frames,
        hold_params=hold_params,
        seed=42,
        rules=rules,
    )
    print(f"[gen] generator produced {len(gen_df)} rows")

    clicks, judgments = clicks_from_generator_df(gen_df)
    print(f"[gen] clicks={len(clicks)} judg={judgments}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{BID}_model_default.osr"
    write_osr(
        out_path,
        beatmap_md5=beatmap_md5,
        player_name="neweraR_dt2",
        mods_int=chosen_mods,
        click_events=clicks,
        cursor_frames=cursor_frames,
        judgments=judgments,
    )
    print(f"[gen] wrote {out_path} ({out_path.stat().st_size} bytes)")

    # Round-trip verify
    import osrparse
    rp2 = osrparse.Replay.from_path(out_path)
    print(f"[verify] parsed back: md5={rp2.beatmap_hash}  event_count={len(rp2.replay_data)}  mods={rp2.mods}")
    assert rp2.beatmap_hash == beatmap_md5, "md5 mismatch"
    print(f"[verify] round-trip OK")

    # Write pick metadata for the follow-up CG step
    meta = {
        "bid": BID,
        "chosen_user": chosen_user,
        "chosen_mods": chosen_mods,
        "human_replay_bin": str(rp_path),
        "model_osr": str(out_path),
        "beatmap_md5": beatmap_md5,
    }
    (OUT_DIR / f"{BID}_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[gen] meta → {OUT_DIR / f'{BID}_meta.json'}")


if __name__ == "__main__":
    main()
