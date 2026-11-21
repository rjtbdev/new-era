"""Generate + Circleguard a nomod replay for any (bid, user_id).

Usage: python -u scripts/gen_cg_nomod_any.py <bid> <user_id>
"""
from __future__ import annotations
import pathlib, sys, lzma, os, json, datetime as _dt
import torch  # DLL init order
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from osrparse import GameMode, Key, Mod, Replay, ReplayEventOsu
from dotenv import load_dotenv

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
MODS = 0

BID = int(sys.argv[1])
USER_ID = int(sys.argv[2])

print(f"[gen] target bid={BID} user={USER_ID} mods={MODS}")
api2 = OsuApiV2()
full = api2.get_beatmap(BID)
beatmap_md5 = (full or {}).get("checksum", "0"*32)
print(f"[gen] md5={beatmap_md5}")

osu_path = BM_DIR / f"{BID}.osu"
bm = parse_osu(osu_path)
print(f"[gen] beatmap: {len(bm.hit_objects)} hit objects")

rp_path = REPLAY_DIR / f"{BID}_{USER_ID}_{MODS}.bin"
parsed = parse_replay_bytes(rp_path.read_bytes())
cursor_frames = [{"t": f.t, "x": f.x, "y": f.y} for f in parsed.frames]
print(f"[gen] cursor: {len(cursor_frames)} frames")

# Measure reference human's offset std → style target
from src.data.features import extract_features_for_replay
import statistics as _st
_rows = extract_features_for_replay(bm, parsed, beatmap_id=BID, user_id=USER_ID, mods=MODS)
_offs = [r["offset_ms"] for r in _rows if r.get("offset_ms") is not None]
ref_std = _st.pstdev(_offs) if len(_offs) >= 2 else None
print(f"[gen] reference human offset std: {ref_std:.2f} ms ({len(_offs)} clicks)")

rules = RuleConfig(seed=42, jitter_std_ms=1.0, max_same_key_streak=25, alt_gap_threshold_ms=80)
gen_df = model_generate(
    bm, mods=MODS, player_id=USER_ID, beatmap_id=BID,
    target_offset_std_ms=ref_std,
    timing_ckpt=MODELS / "timing_best.pt",
    outcome_ckpt=MODELS / "outcome_best.pt",
    key_ckpt=MODELS / "key_best.pt",
    cursor_frames=cursor_frames,
    hold_params={"mu": 155.0, "sigma": 50.0},
    seed=42,
    rules=rules,
)
print(f"[gen] {len(gen_df)} rows")

clicks, judgments = clicks_from_generator_df(gen_df)
print(f"[gen] clicks={len(clicks)} judg={judgments}")

OUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUT_DIR / f"{BID}_model_cursor_nomod.osr"
write_osr(out_path, beatmap_md5=beatmap_md5, player_name="neweraR_nomod_v2",
          mods_int=MODS, click_events=clicks, cursor_frames=cursor_frames, judgments=judgments)
print(f"[gen] wrote {out_path}")

# Also wrap the human as .osr for comparison
out_wrap = OUT_DIR / f"{BID}_human_wrapped_nomod_v2.osr"
raw = rp_path.read_bytes()
text = lzma.decompress(raw).decode("ascii", errors="replace")
events = []
for part in text.rstrip(",").split(","):
    if not part: continue
    pcs = part.split("|")
    if len(pcs) != 4: continue
    try:
        w = int(pcs[0]); x = float(pcs[1]); y = float(pcs[2]); k = int(pcs[3])
    except ValueError:
        continue
    events.append(ReplayEventOsu(time_delta=w, x=x, y=y, keys=Key(k)))
n_obj = len([h for h in bm.hit_objects if h.kind in ("circle","slider")])
rp = Replay(mode=GameMode.STD, game_version=20210520, beatmap_hash=beatmap_md5,
            username="neweraR_human", replay_hash="",
            count_300=n_obj, count_100=0, count_50=0, count_geki=0, count_katu=0, count_miss=0,
            score=1_000_000, max_combo=n_obj, perfect=True, mods=Mod(MODS),
            life_bar_graph=None, timestamp=_dt.datetime.utcnow(),
            replay_data=events, replay_id=0, rng_seed=0)
rp.write_path(str(out_wrap))

load_dotenv(PROJECT / ".env")
API_KEY = os.environ["OSU_API_V1_KEY"]
from circleguard import Circleguard, ReplayPath
cg = Circleguard(API_KEY)

def measure(name, path):
    import statistics
    r = ReplayPath(str(path))
    cg.load(r)
    try: ur = cg.ur(r)
    except Exception as e: ur = f"ERR:{e}"
    try: snaps = len(cg.snaps(r))
    except Exception as e: snaps = f"ERR:{e}"
    try: ft = cg.frametime(r)
    except Exception as e: ft = f"ERR:{e}"
    try:
        fts = cg.frametimes(r)
        med = statistics.median(fts) if len(fts) else -1
        sub6 = sum(1 for d in fts if d < 6) / max(len(fts), 1)
    except Exception:
        med, sub6 = -1, -1
    try: j = f"{r.count_300}/{r.count_100}/{r.count_50}/{r.count_miss}"
    except Exception: j = "?"
    ur_s = f"{ur:.2f}" if isinstance(ur,(float,int)) else ur
    ft_s = f"{ft:.2f}" if isinstance(ft,(float,int)) else ft
    med_s = f"{med:.2f}" if isinstance(med,(float,int)) else med
    print(f"[{name:<8}] UR={ur_s}  snaps={snaps}  ft={ft_s}  ft_med={med_s}  sub6={sub6:.3f}  300/100/50/miss={j}")

print("\n=== Circleguard ===")
measure("model", out_path)
measure("human", out_wrap)
