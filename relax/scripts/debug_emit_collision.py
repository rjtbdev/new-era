"""Check the actual emitted .osr for missing or swallowed key events."""
from __future__ import annotations
import pathlib, sys, lzma, re
import torch  # DLL init order
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.api.v2 import OsuApiV2
from src.data.beatmap import parse_osu
from src.data.emit import clicks_from_generator_df, _build_frames
from src.data.parse import parse_replay_bytes
from src.models.generator import generate as model_generate
from src.models.rules import RuleConfig

PROJECT = pathlib.Path(__file__).resolve().parents[1]
BID = 1851299
USER_ID = 673214
MODS = 0

bm = parse_osu(PROJECT / "data/raw/beatmaps" / f"{BID}.osu")
rp = parse_replay_bytes((PROJECT / "data/raw/replays" / f"{BID}_{USER_ID}_{MODS}.bin").read_bytes())
cursor_frames = [{"t": f.t, "x": f.x, "y": f.y} for f in rp.frames]

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
clicks, judg = clicks_from_generator_df(gen_df)
print(f"[gen] produced {len(clicks)} clicks")

# Look at consecutive same-key collisions where second press is within prev hold window
print("\n=== Same-key overlaps (press inside prev hold) ===")
last_by_key = {}
overlaps = 0
for c in clicks:
    prev = last_by_key.get(c.key)
    if prev is not None:
        prev_release = prev.click_t_ms + prev.hold_ms
        if prev_release >= c.click_t_ms:
            overlaps += 1
            gap = c.click_t_ms - prev.click_t_ms
            overlap_ms = prev_release - c.click_t_ms
            print(f"  key={c.key}  prev t={prev.click_t_ms:.0f} hold={prev.hold_ms:.1f}  → next t={c.click_t_ms:.0f}  gap={gap:.1f}ms  overlap={overlap_ms:.1f}ms")
    last_by_key[c.key] = c
print(f"[total overlaps] {overlaps}")

# Now actually build frames and inspect the final .osr key events
events = _build_frames(cursor_frames, clicks, lead_in_ms=-2000, rng_seed=0)
print(f"\n[emit] {len(events)} ReplayEvents")

# Walk through to count K1/K2 press/release transitions in the output
cum_t = 0
presses_k1 = 0; presses_k2 = 0
last_k1_down = False; last_k2_down = False
for e in events:
    cum_t += e.time_delta
    k1_now = bool(e.keys.value & 4) or bool(e.keys.value & 1)
    k2_now = bool(e.keys.value & 8) or bool(e.keys.value & 2)
    if k1_now and not last_k1_down:
        presses_k1 += 1
    if k2_now and not last_k2_down:
        presses_k2 += 1
    last_k1_down = k1_now
    last_k2_down = k2_now

print(f"[emit output presses] K1={presses_k1}  K2={presses_k2}  TOTAL={presses_k1+presses_k2}")
print(f"[expected] {len(clicks)} total presses from gen_df")
k1_in, k2_in = 0, 0
for c in clicks:
    if c.key == "K1": k1_in += 1
    else: k2_in += 1
print(f"[expected] K1={k1_in} K2={k2_in}")
print(f"[DELTA] missing K1 presses: {k1_in - presses_k1}, missing K2 presses: {k2_in - presses_k2}")
