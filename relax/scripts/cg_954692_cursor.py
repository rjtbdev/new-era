"""Circleguard on the cursor-model 954692 HDDT replay. Compare to human."""
from __future__ import annotations
import pathlib, sys, lzma, os, json, datetime as _dt
import torch  # DLL init order
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from osrparse import GameMode, Key, Mod, Replay, ReplayEventOsu
from dotenv import load_dotenv

PROJECT = pathlib.Path(__file__).resolve().parents[1]
BID = 954692
USER_ID = 10302933
USERNAME = "neweraR_user_1"
MODS = 72  # HDDT

meta = json.loads((PROJECT / "data/raw/beatmaps" / f"{BID}.json").read_text())
md5 = meta["checksum"]
print(f"md5={md5}")

# Wrap human
bin_path = PROJECT / "data/raw/replays" / f"{BID}_{USER_ID}_{MODS}.bin"
out_wrap = PROJECT / "data/generated/replays" / f"{BID}_human_wrapped_dt.osr"
raw = bin_path.read_bytes()
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
print(f"[human] parsed {len(events)} frames")

rp = Replay(
    mode=GameMode.STD, game_version=20210520, beatmap_hash=md5,
    username=USERNAME, replay_hash="",
    count_300=407, count_100=0, count_50=0, count_geki=0, count_katu=0, count_miss=0,
    score=1_000_000, max_combo=407, perfect=True, mods=Mod(MODS),
    life_bar_graph=None, timestamp=_dt.datetime.utcnow(),
    replay_data=events, replay_id=0, rng_seed=0,
)
rp.write_path(str(out_wrap))
print(f"[human] wrote {out_wrap} ({out_wrap.stat().st_size} bytes)")

model_osr = PROJECT / "data/generated/replays" / f"{BID}_model_cursor_dt.osr"
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
        sub13 = sum(1 for d in fts if d < 13) / max(len(fts), 1)
    except Exception as e:
        med, sub6, sub13 = -1, -1, -1
    try:
        j = f"{r.count_300}/{r.count_100}/{r.count_50}/{r.count_miss}"
    except Exception:
        j = "?"
    ur_s = f"{ur:.2f}" if isinstance(ur,(float,int)) else ur
    ft_s = f"{ft:.2f}" if isinstance(ft,(float,int)) else ft
    med_s = f"{med:.2f}" if isinstance(med,(float,int)) else med
    print(f"[{name:<8}] UR={ur_s}  snaps={snaps}  ft_cg={ft_s}  ft_median_raw={med_s}  sub6={sub6:.3f}  sub13={sub13:.3f}  300/100/50/miss={j}")

print("\n=== Circleguard results ===")
measure("model", model_osr)
measure("human", out_wrap)
