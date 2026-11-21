"""Check whether the 16 phantom misses are:
 (a) timing misses — generator clicks fall outside hw50 of ho_t
 (b) aim misses — cursor isn't on the circle at press time
 (c) emitter losses — press event missing from .osr
"""
from __future__ import annotations
import pathlib, sys, math
import torch  # DLL init order
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.api.v2 import OsuApiV2
from src.data.beatmap import parse_osu
from src.data.parse import parse_replay_bytes
from src.models.generator import generate as model_generate
from src.models.rules import RuleConfig
from src.data.features import _circle_radius_px, _effective_od

PROJECT = pathlib.Path(__file__).resolve().parents[1]
BID = 1851299
USER_ID = 673214
MODS = 0

bm = parse_osu(PROJECT / "data/raw/beatmaps" / f"{BID}.osu")
rp = parse_replay_bytes((PROJECT / "data/raw/replays" / f"{BID}_{USER_ID}_{MODS}.bin").read_bytes())
cursor_frames = [{"t": f.t, "x": f.x, "y": f.y} for f in rp.frames]

eod = _effective_od(bm.od, MODS)
hw50 = 200 - 10 * eod
radius = _circle_radius_px(bm.cs, MODS)
print(f"[cfg] hw50={hw50:.1f}ms radius={radius:.1f}px")

gen_df = model_generate(
    bm, mods=MODS, player_id=USER_ID,
    timing_ckpt=PROJECT / "models/timing_best.pt",
    outcome_ckpt=PROJECT / "models/outcome_best.pt",
    key_ckpt=PROJECT / "models/key_best.pt",
    cursor_frames=cursor_frames,
    hold_params={"mu": 155.0, "sigma": 50.0},
    seed=42,
    rules=RuleConfig(seed=42, jitter_std_ms=1.0, max_same_key_streak=25, alt_gap_threshold_ms=80),
)
print(f"[gen] {len(gen_df)} rows produced")

def cursor_at(t):
    frames = rp.frames
    if t <= frames[0].t: return frames[0].x, frames[0].y
    if t >= frames[-1].t: return frames[-1].x, frames[-1].y
    lo, hi = 0, len(frames)-1
    while lo < hi-1:
        mid = (lo+hi)//2
        if frames[mid].t <= t: lo = mid
        else: hi = mid
    a, b = frames[lo], frames[hi]
    if b.t == a.t: return a.x, a.y
    f = (t-a.t)/(b.t-a.t)
    return a.x + f*(b.x-a.x), a.y + f*(b.y-a.y)

timing_miss = 0; aim_miss = 0; both_miss = 0; ok = 0
for _, row in gen_df.iterrows():
    ct = float(row["click_t"])
    if math.isnan(ct): continue
    ho_t = float(row["ho_t"])
    ho_x = float(row["ho_x"]); ho_y = float(row["ho_y"])
    off = ct - ho_t
    cx, cy = cursor_at(ct)
    dist = math.hypot(cx - ho_x, cy - ho_y)
    bad_time = abs(off) > hw50
    bad_aim = dist > radius
    if bad_time and bad_aim: both_miss += 1
    elif bad_time: timing_miss += 1
    elif bad_aim: aim_miss += 1
    else: ok += 1
    if bad_time or bad_aim:
        print(f"  row ho_t={ho_t:.0f} off={off:.1f} dist={dist:.1f}  bad_time={bad_time} bad_aim={bad_aim}")

print(f"\n[summary] ok={ok}  timing_miss={timing_miss}  aim_miss={aim_miss}  both_miss={both_miss}")
print(f"[summary] total_miss = {timing_miss + aim_miss + both_miss}")
