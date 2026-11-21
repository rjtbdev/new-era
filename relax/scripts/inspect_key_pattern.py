"""Analyze key-choice behavior in the generated model replays.

Questions:
 1. Preference: is the model biased toward K1 or K2 overall?
 2. Combo starts: when a new combo starts, is it K1-first? (That is the human
    convention — start a burst/combo with the dominant hand.)
 3. Alternation vs single-tap: where does it switch between modes?
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

PROJECT = pathlib.Path(__file__).resolve().parents[1]

CASES = [
    (1851299, 673214,       0,  "Harumachi 4.83* nomod"),
    (1113561, 1251071,      0,  "1113561 6.01* nomod slow"),
    (1921069, 10434787,     0,  "1921069 6.86* nomod tech"),
    (954692,  10302933,    72,  "954692 HDDT"),
]

for bid, uid, mods, label in CASES:
    bm = parse_osu(PROJECT / "data/raw/beatmaps" / f"{bid}.osu")
    rp = parse_replay_bytes((PROJECT / "data/raw/replays" / f"{bid}_{uid}_{mods}.bin").read_bytes())
    cursor_frames = [{"t": f.t, "x": f.x, "y": f.y} for f in rp.frames]
    gen_df = model_generate(
        bm, mods=mods, player_id=uid,
        timing_ckpt=PROJECT / "models/timing_best.pt",
        outcome_ckpt=PROJECT / "models/outcome_best.pt",
        key_ckpt=PROJECT / "models/key_best.pt",
        cursor_frames=cursor_frames,
        hold_params={"mu": 155.0, "sigma": 50.0},
        seed=42,
        rules=RuleConfig(seed=42, jitter_std_ms=1.0, max_same_key_streak=25, alt_gap_threshold_ms=80),
    )

    k1 = sum(1 for _, r in gen_df.iterrows() if r["key_type"] == "K1")
    k2 = sum(1 for _, r in gen_df.iterrows() if r["key_type"] == "K2")
    total = k1 + k2

    # Combo starts — rows where ho_new_combo is True (first note of a combo)
    combo_starts_k1 = 0
    combo_starts_k2 = 0
    combo_starts_total = 0
    for _, r in gen_df.iterrows():
        if r.get("ho_new_combo"):
            combo_starts_total += 1
            if r["key_type"] == "K1":
                combo_starts_k1 += 1
            else:
                combo_starts_k2 += 1

    # Streak lengths — runs of same key
    streaks = []
    cur_key = None; cur_len = 0
    for _, r in gen_df.iterrows():
        k = r["key_type"]
        if k == cur_key:
            cur_len += 1
        else:
            if cur_len > 0:
                streaks.append((cur_key, cur_len))
            cur_key = k; cur_len = 1
    if cur_len > 0: streaks.append((cur_key, cur_len))
    max_streak = max((n for _,n in streaks), default=0)
    avg_streak = sum(n for _,n in streaks)/max(1,len(streaks))
    # Count streaks >= 3 (bursts/short runs — typically humans alternate past 2)
    bursts_k1 = sum(1 for k,n in streaks if k=="K1" and n>=3)
    bursts_k2 = sum(1 for k,n in streaks if k=="K2" and n>=3)

    # Same-key chains that look like "stuck on one key" vs alternation
    runs_of_1 = sum(1 for _,n in streaks if n==1)  # i.e. alternation (each key only held once)
    runs_of_2 = sum(1 for _,n in streaks if n==2)
    runs_ge3 = sum(1 for _,n in streaks if n>=3)

    print(f"\n=== {label}  (bid={bid}) ===")
    print(f"  total clicks: {total}   K1={k1} ({100*k1/total:.1f}%)   K2={k2} ({100*k2/total:.1f}%)")
    print(f"  combo starts: {combo_starts_total}   K1-start={combo_starts_k1} ({100*combo_starts_k1/max(1,combo_starts_total):.1f}%)   K2-start={combo_starts_k2} ({100*combo_starts_k2/max(1,combo_starts_total):.1f}%)")
    print(f"  streaks: max={max_streak}  avg={avg_streak:.1f}  #runs={len(streaks)}")
    print(f"           singles (len=1, pure alt)={runs_of_1}   doubles (len=2)={runs_of_2}   len>=3 (held)={runs_ge3}")
    print(f"           long-K1 bursts (>=3): {bursts_k1}   long-K2 bursts (>=3): {bursts_k2}")
