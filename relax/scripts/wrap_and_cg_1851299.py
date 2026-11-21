"""Wrap the Roro Rosset .bin into .osr, then Circleguard both model + human."""
from __future__ import annotations
import pathlib, sys, lzma, datetime as _dt, json, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from typing import List
from osrparse import GameMode, Key, Mod, Replay, ReplayEventOsu

PROJECT = pathlib.Path(__file__).resolve().parents[1]
BID = 1851299
USER_ID = 673214
USERNAME = "Roro Rosset"
MODS = 0

# Load beatmap md5 from v2 metadata
meta = json.loads((PROJECT / "data/raw/beatmaps" / f"{BID}.json").read_text())
md5 = meta["checksum"]
print(f"md5={md5}")

# Wrap the .bin
bin_path = PROJECT / "data/raw/replays" / f"{BID}_{USER_ID}_{MODS}.bin"
out_wrap = PROJECT / "data/generated/replays" / f"{BID}_human_wrapped.osr"

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
    count_300=113, count_100=0, count_50=0, count_geki=0, count_katu=0, count_miss=0,
    score=1_000_000, max_combo=113, perfect=True, mods=Mod(MODS),
    life_bar_graph=None, timestamp=_dt.datetime.utcnow(),
    replay_data=events, replay_id=0, rng_seed=0,
)
rp.write_path(str(out_wrap))
print(f"[human] wrote {out_wrap} ({out_wrap.stat().st_size} bytes)")

# Circleguard both
model_osr = PROJECT / "data/generated/replays" / f"{BID}_model_default.osr"
from dotenv import load_dotenv
load_dotenv(PROJECT / ".env")
API_KEY = os.environ["OSU_API_V1_KEY"]

from circleguard import Circleguard, ReplayPath

cg = Circleguard(API_KEY)

def measure(name, path):
    r = ReplayPath(str(path))
    cg.load(r)
    try:
        ur = cg.ur(r)
    except Exception as e:
        ur = f"ERR:{e}"
    try:
        snaps = len(cg.snaps(r))
    except Exception as e:
        snaps = f"ERR:{e}"
    try:
        ft = cg.frametime(r)
    except Exception as e:
        ft = f"ERR:{e}"
    try:
        fts = cg.frametimes(r)
        import statistics
        med = statistics.median(fts) if len(fts) else -1
        sub6 = sum(1 for d in fts if d < 6) / max(len(fts), 1)
        sub13 = sum(1 for d in fts if d < 13) / max(len(fts), 1)
    except Exception as e:
        med = sub6 = sub13 = -1
    try:
        judg = cg.judgments(r)
        c300 = sum(1 for j in judg if str(j.type) == "JudgmentType.Hit300" or getattr(j, 'type', None) and 'Hit300' in str(j.type))
        c100 = sum(1 for j in judg if 'Hit100' in str(j.type))
        c50  = sum(1 for j in judg if 'Hit50' in str(j.type))
        cmiss= sum(1 for j in judg if 'Miss' in str(j.type))
        judgs = f"{c300}/{c100}/{c50}/{cmiss}"
    except Exception as e:
        judgs = f"ERR:{e}"
    print(f"[{name:<8}] UR={ur if isinstance(ur, str) else f'{ur:.2f}'}  snaps={snaps}  ft_cg={ft if isinstance(ft, str) else f'{ft:.2f}'}  ft_median_raw={med:.2f}  sub6={sub6:.3f}  sub13={sub13:.3f}  300/100/50/miss={judgs}")

print("\n=== Circleguard results ===")
measure("model", model_osr)
measure("human", out_wrap)
