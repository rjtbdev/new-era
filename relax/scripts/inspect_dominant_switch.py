"""Check whether the model's 'default' key for single-tap sections shifts
around within one replay.

Approach: split the replay into chunks of ~50 clicks, and for each chunk
compute the dominant key during SINGLE-TAP sections (gap > alt threshold).
A consistent player uses the same key as their default across the whole map;
the model shifting would be a tell.
"""
from __future__ import annotations
import pathlib, sys
import torch  # DLL init order
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

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

ALT_GAP_MS = 80.0  # above this = single-tap zone

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

    # Walk the clicks — only single-tap zone (|prev_gap_ms| > ALT_GAP_MS)
    import math
    rows = list(gen_df.itertuples(index=False))

    chunk_size = max(50, len(rows) // 8)
    print(f"\n=== {label}  n={len(rows)}  chunk={chunk_size} ===")
    chunks = []
    for i in range(0, len(rows), chunk_size):
        sub = rows[i:i+chunk_size]
        k1 = 0; k2 = 0
        for r in sub:
            gap = r.prev_gap_ms
            if gap is None or (isinstance(gap, float) and math.isnan(gap)):
                continue
            if abs(gap) < ALT_GAP_MS:
                continue  # skip alternation zone
            if r.key_type == "K1": k1 += 1
            elif r.key_type == "K2": k2 += 1
        tot = k1 + k2
        if tot == 0:
            chunks.append(("--", 0))
            continue
        dom = "K1" if k1 > k2 else ("K2" if k2 > k1 else "==")
        pct = 100 * max(k1, k2) / tot
        chunks.append((dom, pct))
        print(f"  chunk {i:>4}-{i+len(sub)-1:<4}  singletap:{tot:3d}  K1={k1:3d}  K2={k2:3d}  dominant={dom} ({pct:.0f}%)")

    # Summary: does the dominant key stay consistent?
    doms = [d for d,_ in chunks if d not in ("--","==")]
    if len(set(doms)) <= 1:
        print(f"  CONSISTENT: dominant = {doms[0] if doms else '?'}")
    else:
        flips = sum(1 for a,b in zip(doms, doms[1:]) if a != b)
        print(f"  INCONSISTENT: {flips} dominance flips across {len(chunks)} chunks  ({doms})")
