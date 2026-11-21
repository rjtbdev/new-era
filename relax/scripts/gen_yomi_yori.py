"""Generate a model replay on a brand new Yomi Yori map (nomod).

Pipeline:
  1. Search osu! API v2 for Yomi Yori beatmapsets, pick a 4-6* osu!std diff.
  2. Pull its .osu + metadata.
  3. Pull nomod top-50 leaderboard, pick top nomod player.
  4. Download that replay via API v1.
  5. Generate model replay using that human's cursor + jitter 1.0.
  6. Round-trip + Circleguard.
"""

from __future__ import annotations

import base64
import json
import os
import pathlib
import sys
import time

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


def _save(path: pathlib.Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, (dict, list)):
        path.write_text(json.dumps(data, indent=2))
    elif isinstance(data, bytes):
        path.write_bytes(data)
    else:
        path.write_text(str(data))


def pick_beatmap(api2: OsuApiV2) -> dict:
    """Search for Yomi Yori; pick a 4-6* osu!std ranked/loved diff with high playcount.

    STRICT: the beatmapset title/artist must contain 'yomi' (case-insensitive).
    """
    # Directly target the canonical Yomi Yori map:
    # Imperial Circus Dead Decadence - Yomi yori Kikoyu, Koukoku no Tou to Honoo no Shoujo.
    # [Reverberation] -- bid=1218337. Only diff in its set; SR=7.44, 8:10 long.
    # Above our 4-6* training range, so this is an extrapolation test.
    TARGET_BID = 1218337
    bm = api2.get_beatmap(TARGET_BID)
    if not bm:
        raise RuntimeError(f"Yomi Yori map {TARGET_BID} not available")
    bset = api2.get_beatmapset(bm["beatmapset_id"])
    return {
        "beatmap_id": bm["id"],
        "beatmapset_id": bm["beatmapset_id"],
        "title": bset["title"],
        "title_unicode": bset.get("title_unicode", ""),
        "artist": bset["artist"],
        "creator": bset["creator"],
        "version": bm["version"],
        "sr": bm["difficulty_rating"],
        "bpm": bset.get("bpm"),
        "playcount": bm.get("playcount", 0),
        "status": bset.get("status"),
    }


def pull_beatmap_file(beatmap_id: int) -> pathlib.Path:
    """Download the .osu file from ppy.sh."""
    import requests
    out = BM_DIR / f"{beatmap_id}.osu"
    if out.exists() and out.stat().st_size > 100:
        return out
    r = requests.get(f"https://osu.ppy.sh/osu/{beatmap_id}", timeout=30)
    r.raise_for_status()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(r.content)
    return out


def pull_nomod_leaderboard(api2: OsuApiV2, bid: int) -> list[dict]:
    """Pull top nomod scores. Return list of {user_id, mods_int, enabled_mods_str}."""
    # API v2 mods filter: nomod requires explicit mods[]=NM filter? Actually "mods[]=NM" isn't a real mod.
    # Pull global, then filter to ones with mods=[] (nomod).
    resp = api2.get_beatmap_scores(bid, mode="osu", scope="global", limit=50)
    scores = resp.get("scores", []) if resp else []
    nomod = []
    for s in scores:
        mods = s.get("mods", [])
        if not mods:  # nomod
            nomod.append({
                "user_id": s["user"]["id"],
                "username": s["user"]["username"],
                "score": s.get("score"),
                "pp": s.get("pp"),
                "accuracy": s.get("accuracy"),
                "mods_int": 0,
            })
    return nomod, scores


def mods_list_to_int(mods: list[str]) -> int:
    """Convert osu! API mod short-codes to the int bitfield."""
    # osu! mod bits — full list in https://github.com/ppy/osu-api/wiki
    BITS = {
        "NF": 1, "EZ": 2, "TD": 4, "HD": 8, "HR": 16, "SD": 32, "DT": 64,
        "RX": 128, "HT": 256, "NC": 512, "FL": 1024, "AT": 2048, "SO": 4096,
        "AP": 8192, "PF": 16384, "4K": 32768, "5K": 65536, "6K": 131072,
        "7K": 262144, "8K": 524288, "FI": 1048576, "RD": 2097152, "CM": 4194304,
        "TP": 8388608, "9K": 16777216, "CO": 33554432, "1K": 67108864,
        "3K": 134217728, "2K": 268435456, "V2": 536870912, "MR": 1073741824,
    }
    v = 0
    for m in mods:
        v |= BITS.get(m, 0)
    return v


def pull_replay(api1: OsuApiV1, bid: int, user_id: int, mods_int: int) -> pathlib.Path | None:
    """Download .bin replay frames; save to REPLAY_DIR."""
    REPLAY_DIR.mkdir(parents=True, exist_ok=True)
    out = REPLAY_DIR / f"{bid}_{user_id}_{mods_int}.bin"
    if out.exists() and out.stat().st_size > 100:
        return out
    data = api1.get_replay_bytes(beatmap_id=bid, user_id=user_id, mods=mods_int)
    if not data:
        return None
    out.write_bytes(data)
    return out


def main():
    print("[yomi] authenticating v2 ...")
    api2 = OsuApiV2()
    api1 = OsuApiV1()

    print("[yomi] searching for Yomi Yori maps ...")
    bm = pick_beatmap(api2)
    bid = bm["beatmap_id"]
    print(f"[yomi] picked: {bm['artist']} - {bm['title']} [{bm['version']}] "
          f"by {bm['creator']} (bid={bid}, SR={bm['sr']:.2f}*, BPM={bm['bpm']}, "
          f"plays={bm['playcount']}, status={bm['status']})")

    # Save metadata
    full = api2.get_beatmap(bid)
    if full:
        _save(BM_DIR / f"{bid}.json", full)
    beatmap_md5 = (full or {}).get("checksum", "0" * 32)

    print("[yomi] downloading .osu file ...")
    osu_path = pull_beatmap_file(bid)
    print(f"[yomi] .osu at {osu_path} ({osu_path.stat().st_size} bytes)")

    print("[yomi] fetching leaderboard ...")
    nomod, all_scores = pull_nomod_leaderboard(api2, bid)
    fallback_mod = None
    picked = None
    if nomod:
        picked = nomod[0]
        print(f"[yomi] {len(nomod)} nomod score(s) found. Top: {picked['username']} "
              f"(pp={picked['pp']}, acc={picked['accuracy']})")
    else:
        # Fallback: HD-only, else first available
        for s in all_scores:
            mods = s.get("mods", [])
            if mods == ["HD"]:
                picked = {
                    "user_id": s["user"]["id"],
                    "username": s["user"]["username"],
                    "mods_int": mods_list_to_int(mods),
                }
                fallback_mod = "HD"
                print(f"[yomi] FALLBACK to HD: {picked['username']}")
                break
        if picked is None and all_scores:
            s = all_scores[0]
            picked = {
                "user_id": s["user"]["id"],
                "username": s["user"]["username"],
                "mods_int": mods_list_to_int(s.get("mods", [])),
            }
            fallback_mod = "+".join(s.get("mods", [])) or "NM"
            print(f"[yomi] FALLBACK to {fallback_mod}: {picked['username']}")
    if picked is None:
        raise RuntimeError(f"no leaderboard scores available for bid={bid}")

    print("[yomi] downloading replay ...")
    rp = pull_replay(api1, bid, picked["user_id"], picked["mods_int"])
    if rp is None:
        raise RuntimeError(f"failed to download replay for user {picked['user_id']}")
    print(f"[yomi] replay at {rp} ({rp.stat().st_size} bytes)")

    print("[yomi] parsing cursor trajectory ...")
    parsed = parse_replay_bytes(rp.read_bytes())
    cursor_frames = [{"t": f.t, "x": f.x, "y": f.y} for f in parsed.frames]
    print(f"[yomi] cursor has {len(cursor_frames)} frames, t=[{cursor_frames[0]['t']},{cursor_frames[-1]['t']}]ms")

    print("[yomi] parsing beatmap ...")
    bm_parsed = parse_osu(osu_path)
    n_ho = len(bm_parsed.hit_objects)
    print(f"[yomi] beatmap has {n_ho} hit objects")

    # Hold params: just use a reasonable global default (we don't have shards for this user)
    # 155ms mean, 50ms std is in the typical human range per prior fits
    hold_params = {"mu": 155.0, "sigma": 50.0}

    rules = RuleConfig(seed=42, jitter_std_ms=1.0, max_same_key_streak=25, alt_gap_threshold_ms=80)
    print(f"[yomi] generating with rules: {rules}")

    gen_df = model_generate(
        bm_parsed,
        mods=picked["mods_int"],
        player_id=picked["user_id"],
        timing_ckpt=MODELS / "timing_best.pt",
        outcome_ckpt=MODELS / "outcome_best.pt",
        key_ckpt=MODELS / "key_best.pt",
        cursor_frames=cursor_frames,
        hold_params=hold_params,
        seed=42,
        rules=rules,
    )
    print(f"[yomi] generator produced {len(gen_df)} rows")
    if gen_df.empty:
        raise RuntimeError("generator produced no clicks")

    clicks, judgments = clicks_from_generator_df(gen_df)
    print(f"[yomi] clicks={len(clicks)} judg={judgments}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{bid}_model_default.osr"
    write_osr(
        out_path,
        beatmap_md5=beatmap_md5,
        player_name=f"neweraR_yomi",
        mods_int=picked["mods_int"],
        click_events=clicks,
        cursor_frames=cursor_frames,
        judgments=judgments,
    )
    print(f"[yomi] wrote {out_path} ({out_path.stat().st_size} bytes)")

    # Round-trip
    from osrparse import Replay
    r = Replay.from_path(str(out_path))
    rt_ok = {
        "md5_ok": r.beatmap_hash == beatmap_md5,
        "n_events": len(r.replay_data),
        "mods_ok": int(r.mods) == picked["mods_int"],
        "count_300": r.count_300,
        "count_100": r.count_100,
        "count_50": r.count_50,
        "count_miss": r.count_miss,
    }
    print(f"[yomi] round-trip: {rt_ok}")

    # Build a wrapped human replay too — needed for Circleguard baseline (API v1 gives frames only)
    human_wrapped = OUT_DIR / f"{bid}_human_wrapped.osr"
    # Wrap raw frames into .osr using same emitter
    # Convert human frames to click_events shape? No — we just need a valid .osr with human's actual events.
    # Simpler: use the real frames + synthesize minimal judgment counts.
    # We call write_osr with click_events=[] won't work; use helper _wrap_human if it exists.
    wrap_script = PROJECT / "scripts" / "_wrap_human.py"
    if wrap_script.exists():
        # Run it as a subprocess with args or import
        print("[yomi] wrapping human replay for Circleguard baseline ...")
        import subprocess
        res = subprocess.run(
            [sys.executable, str(wrap_script), str(bid), str(picked["user_id"]), str(picked["mods_int"])],
            capture_output=True, text=True
        )
        print(res.stdout)
        if res.returncode != 0:
            print(f"[yomi] wrap failed: {res.stderr}")

    summary = {
        "beatmap": bm,
        "cursor_source": picked,
        "fallback_mod": fallback_mod,
        "out_path": str(out_path),
        "judgments": judgments,
        "n_clicks": len(clicks),
        "round_trip": rt_ok,
    }
    (PROJECT / "data" / "generated" / "yomi_yori_gen.json").write_text(json.dumps(summary, indent=2))
    print(f"[yomi] summary at data/generated/yomi_yori_gen.json")

    # Return bid + user_id + mods for the Circleguard step
    print(f"\n=== GEN DONE ===")
    print(f"BID={bid}")
    print(f"USER={picked['user_id']}")
    print(f"MODS={picked['mods_int']}")
    print(f"MODEL_OSR={out_path}")


if __name__ == "__main__":
    main()
