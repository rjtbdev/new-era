"""Generate model replay for bid=1851299 (Harumachi Clover [Woey's Extra]) nomod,
using Roro Rosset's cursor trajectory. Then run Circleguard.
"""
from __future__ import annotations
import pathlib, sys, json, os
import torch  # DLL init order on Windows: torch MUST come before numpy/pandas
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

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

BID = 1851299
USER_ID = 673214  # Roro Rosset
MODS = 0  # nomod

def main():
    print(f"[gen] target bid={BID} user={USER_ID} mods={MODS}")
    api2 = OsuApiV2()
    full = api2.get_beatmap(BID)
    beatmap_md5 = (full or {}).get("checksum", "0"*32)
    print(f"[gen] md5={beatmap_md5}")

    osu_path = BM_DIR / f"{BID}.osu"
    bm_parsed = parse_osu(osu_path)
    print(f"[gen] beatmap: {len(bm_parsed.hit_objects)} hit objects")

    rp = REPLAY_DIR / f"{BID}_{USER_ID}_{MODS}.bin"
    parsed = parse_replay_bytes(rp.read_bytes())
    cursor_frames = [{"t": f.t, "x": f.x, "y": f.y} for f in parsed.frames]
    print(f"[gen] cursor: {len(cursor_frames)} frames, t=[{cursor_frames[0]['t']},{cursor_frames[-1]['t']}]ms")

    # Measure the reference human's offset std on this map → use as the style
    # target so our emitted replay matches this human's tightness on this map.
    from src.data.features import extract_features_for_replay
    _rows = extract_features_for_replay(bm_parsed, parsed, beatmap_id=BID, user_id=USER_ID, mods=MODS)
    import statistics as _st
    _offs = [r["offset_ms"] for r in _rows if r.get("offset_ms") is not None]
    ref_std = _st.pstdev(_offs) if len(_offs) >= 2 else None
    print(f"[gen] reference human offset std: {ref_std:.2f} ms ({len(_offs)} clicks)")

    hold_params = {"mu": 155.0, "sigma": 50.0}
    rules = RuleConfig(seed=42, jitter_std_ms=1.0, max_same_key_streak=25, alt_gap_threshold_ms=80)
    print(f"[gen] rules: {rules}")

    gen_df = model_generate(
        bm_parsed,
        mods=MODS,
        player_id=USER_ID,
        beatmap_id=BID,
        target_offset_std_ms=ref_std,
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
        player_name="neweraR_nomod",
        mods_int=MODS,
        click_events=clicks,
        cursor_frames=cursor_frames,
        judgments=judgments,
    )
    print(f"[gen] wrote {out_path} ({out_path.stat().st_size} bytes)")

    # Round-trip verify
    with open(out_path, "rb") as f:
        osr_bytes = f.read()
    import osrparse
    rp2 = osrparse.Replay.from_path(out_path)
    print(f"[verify] parsed back: md5={rp2.beatmap_hash}  event_count={len(rp2.replay_data)}  mods={rp2.mods}")
    assert rp2.beatmap_hash == beatmap_md5, "md5 mismatch"
    print(f"[verify] round-trip OK")

if __name__ == "__main__":
    main()
