"""Phase A eval: generate model replays for the three known hold-out maps and
run Circleguard on each, comparing against the human reference replay used for
cursor.

Maps:
  954692  (HDDT jump, 5.6*, Shiawase!!)         -> known-good pre-A
  713442  (HDDT, 5.53*, Reol - No title)        -> post-pre-A confirmation
  1851299 (nomod, 4.83*, Harumachi Clover)      -> known-bad pre-A  <-- the test

Each map already has a source human replay cached in data/raw/replays/.
Uses the NEW Phase A checkpoints at models/{timing,outcome,key}_best.pt.
"""
from __future__ import annotations
import pathlib, sys, json, os, time
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

# Hold-out map configs: (bid, user_id, mods, label)
CASES = [
    (954692,  None, None, "jump_hddt_phase_a"),
    (713442,  None, None, "dt_phase_a"),
    (1851299, 673214, 0,  "nomod_phase_a"),  # Roro Rosset, known cached
]


def find_replay_for(bid: int):
    """Return (user_id, mods, file_path) picking any cached replay for this bid."""
    cands = sorted(REPLAY_DIR.glob(f"{bid}_*.bin"))
    if not cands:
        return None
    # Pattern: {bid}_{user_id}_{mods}.bin
    import re
    m = re.match(rf"{bid}_(\d+)_(\d+)\.bin", cands[0].name)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), cands[0]


def run_one(bid, user_id, mods, label, api2, bm_md5s):
    print(f"\n=== {label} bid={bid} ===")
    if user_id is None or mods is None:
        got = find_replay_for(bid)
        if not got:
            print(f"  SKIP: no cached replay in {REPLAY_DIR} for bid={bid}")
            return None
        user_id, mods, rp_file = got
    else:
        rp_file = REPLAY_DIR / f"{bid}_{user_id}_{mods}.bin"
        if not rp_file.exists():
            print(f"  SKIP: {rp_file} not found")
            return None
    print(f"  user_id={user_id} mods={mods} cursor_src={rp_file.name}")

    beatmap_md5 = bm_md5s.get(bid) or "0" * 32
    osu_path = BM_DIR / f"{bid}.osu"
    bm = parse_osu(osu_path)
    try:
        setattr(bm, "beatmap_id", bid)
    except Exception:
        pass

    parsed = parse_replay_bytes(rp_file.read_bytes())
    cursor_frames = [{"t": f.t, "x": f.x, "y": f.y} for f in parsed.frames]
    print(f"  cursor: {len(cursor_frames)} frames")

    hold_params = {"mu": 155.0, "sigma": 50.0}
    rules = RuleConfig(seed=42, jitter_std_ms=1.0, max_same_key_streak=25, alt_gap_threshold_ms=80)
    gen_df = model_generate(
        bm, mods=mods, player_id=user_id,
        timing_ckpt=MODELS / "timing_best.pt",
        outcome_ckpt=MODELS / "outcome_best.pt",
        key_ckpt=MODELS / "key_best.pt",
        cursor_frames=cursor_frames,
        hold_params=hold_params,
        seed=42, rules=rules,
    )
    clicks, judgments = clicks_from_generator_df(gen_df)
    print(f"  model clicks={len(clicks)} judg={judgments}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{bid}_model_phase_a.osr"
    write_osr(
        out_path,
        beatmap_md5=beatmap_md5,
        player_name="neweraR_phaseA",
        mods_int=mods,
        click_events=clicks,
        cursor_frames=cursor_frames,
        judgments=judgments,
    )
    print(f"  wrote {out_path}")
    return {"bid": bid, "label": label, "mods": mods, "user_id": user_id,
            "out": str(out_path), "judg": judgments, "clicks": len(clicks)}


def main():
    api2 = OsuApiV2()
    bm_md5s = {}
    for bid, _, _, _ in CASES:
        try:
            bm = api2.get_beatmap(bid)
            bm_md5s[bid] = (bm or {}).get("checksum")
        except Exception as e:
            print(f"[md5] bid={bid} lookup failed: {e}")
    results = []
    for bid, uid, mods, label in CASES:
        r = run_one(bid, uid, mods, label, api2, bm_md5s)
        if r:
            results.append(r)
    out = PROJECT / "data" / "phase_a_eval.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\n[summary] wrote {out}")
    for r in results:
        print(f"  bid={r['bid']} {r['label']}  clicks={r['clicks']}  judg={r['judg']}")


if __name__ == "__main__":
    main()
