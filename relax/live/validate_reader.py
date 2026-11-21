"""Attach AoReader to live osu! and sanity-check reads.

Usage:
    py -3 relax/live/validate_reader.py                    # 3 second sample
    py -3 relax/live/validate_reader.py --duration 10      # longer sample
    py -3 relax/live/validate_reader.py --interval 0.05    # faster polling

Prints resolved field VAs after attach(), then polls time_ms + is_paused
at the given interval. Reports time delta vs wall clock.

Requires: ao HV loaded, osu! running, hook_tool.exe reachable (either on
PATH env AO_HOOK_TOOL or at the default path in ao_reader.py).
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

# Path shuffle so `from live.*` and `from src.*` both resolve
HERE = pathlib.Path(__file__).resolve()
RELAX = HERE.parents[1]
sys.path.insert(0, str(RELAX))

from live.ao_reader import AoReader, _HypercallError


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--process", default="osu!.exe")
    ap.add_argument("--duration", type=float, default=3.0, help="seconds to sample")
    ap.add_argument("--interval", type=float, default=0.1, help="seconds between reads")
    args = ap.parse_args()

    reader = AoReader(process_name=args.process)

    print(f"[validate] attaching to {args.process!r}...")
    try:
        reader.attach()
    except _HypercallError as e:
        print(f"[validate] ATTACH FAILED: {e}")
        return 1

    print(f"[validate] cr3=0x{reader._cr3:X}")
    print("[validate] resolved field VAs:")
    for name, va in sorted(reader.resolved_vas().items()):
        print(f"    {name:20s} 0x{va:08X}")

    print(f"[validate] polling time + is_paused for {args.duration}s "
          f"(every {args.interval*1000:.0f}ms)...")
    samples = []
    t_start = time.perf_counter()
    while time.perf_counter() - t_start < args.duration:
        wall_ms = (time.perf_counter() - t_start) * 1000.0
        try:
            # Read primitives directly — avoids the .osu dependency
            t = reader._read_u32(reader._va["time"])
            paused = bool(reader._read_u8(reader._va["is_paused"]))
            mode = reader._read_u32(reader._va["play_mode"])
            samples.append((wall_ms, t, paused, mode))
            print(f"  wall={wall_ms:7.1f}ms  osu_time={t:10d}  "
                  f"paused={int(paused)}  play_mode={mode}")
        except _HypercallError as e:
            print(f"  wall={wall_ms:7.1f}ms  READ FAILED: {e}")
        time.sleep(args.interval)

    if len(samples) < 2:
        print("[validate] not enough samples")
        return 1

    # Sanity checks
    t0 = samples[0][1]
    t1 = samples[-1][1]
    wall_delta_ms = samples[-1][0] - samples[0][0]
    game_delta_ms = t1 - t0

    print()
    print(f"[validate] first  sample: time={t0} paused={samples[0][2]}")
    print(f"[validate] last   sample: time={t1} paused={samples[-1][2]}")
    print(f"[validate] wall delta: {wall_delta_ms:.1f}ms   "
          f"game delta: {game_delta_ms}ms")

    # Verdict
    any_paused = any(s[2] for s in samples)
    if any_paused:
        print("[validate] osu reported paused during sample — can't verify "
              "time advance. Un-pause and re-run.")
    elif game_delta_ms <= 0:
        print("[validate] WARN: time did not advance. osu may be in menu, "
              "spectating, or the time VA resolved wrong.")
    else:
        ratio = game_delta_ms / wall_delta_ms if wall_delta_ms > 0 else 0.0
        print(f"[validate] time advanced {game_delta_ms}ms over {wall_delta_ms:.1f}ms wall "
              f"(ratio={ratio:.2f} — should be ~1.0 at normal speed, 1.5 DT, 0.75 HT)")
        print("[validate] PASS — reader is live.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
