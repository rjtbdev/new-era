"""Live service entry point. Run with mock reader+sink to validate end-to-end.

Usage (mock):
    python -u live/main.py --mock --bid 1851299 --user 673214 --mods 0

The mock path simulates a live session against a known map + known cursor
trace. It prints every press/release event with its scheduled beatmap time
vs the wall-clock time it actually fired, so you can see whether the
inference loop keeps up.
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import torch  # DLL init order on Windows

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from live.dispatcher import Dispatcher
from live.model_service import ModelService
from live.mock_reader import MockReader
from live.sink import MockSink
from src.models.rules import RuleConfig

PROJECT = pathlib.Path(__file__).resolve().parents[1]


def run_mock(args):
    osu_path = PROJECT / "data" / "raw" / "beatmaps" / f"{args.bid}.osu"
    cursor_bin = PROJECT / "data" / "raw" / "replays" / f"{args.bid}_{args.user}_{args.mods}.bin"
    if not osu_path.exists() or not cursor_bin.exists():
        print(f"[live] missing {osu_path} or {cursor_bin}")
        return

    reader = MockReader(
        osu_path=osu_path,
        cursor_bin_path=cursor_bin,
        mods=args.mods,
        beatmap_id=args.bid,
        player_id=args.user,
        start_offset_ms=args.start_offset_ms,
        speed=args.speed,
    )
    reader.attach()

    sink = MockSink(verbose=args.verbose)
    model = ModelService(
        timing_ckpt=PROJECT / "models" / "timing_best.pt",
        key_ckpt=PROJECT / "models" / "key_best.pt",
        device="cpu",
        rules=RuleConfig(seed=42),
    )
    dispatcher = Dispatcher(sink)

    print(f"[live] mock start bid={args.bid} user={args.user} mods={args.mods} "
          f"speed={args.speed} start_offset={args.start_offset_ms}ms")

    # Tick loop. We tick fast for responsive timing of presses.
    poll_hz = args.poll_hz
    plan_hz = args.plan_hz
    tick_dt = 1.0 / poll_hz
    plan_period = 1.0 / plan_hz
    last_plan = 0.0
    last_wall = time.perf_counter()

    end_bt = max((h.t for h in reader._hit_objects), default=0.0) + 1000.0

    while True:
        wall = time.perf_counter()
        if wall - last_wall < tick_dt:
            time.sleep(max(0, tick_dt - (wall - last_wall)))
        last_wall = time.perf_counter()

        state = reader.read()
        model.on_cursor(state.time_ms, state.cursor_x, state.cursor_y)

        if (time.perf_counter() - last_plan) >= plan_period:
            last_plan = time.perf_counter()
            decisions = model.plan(state, lookahead_ms=args.lookahead_ms)
            dispatcher.submit(decisions)

        dispatcher.tick(state.time_ms)

        if state.time_ms >= end_bt:
            break

    print(f"[live] done. {len(sink.events)} key events emitted.")
    # Simple summary + accuracy vs hit object times
    presses = [e for e in sink.events if e[1] == "press"]
    k1 = sum(1 for e in presses if e[2] == "K1")
    k2 = sum(1 for e in presses if e[2] == "K2")
    print(f"[live] presses: K1={k1} K2={k2} total={len(presses)}")

    # Align each press to the nearest clickable HO and report offset distribution
    clickable = [h for h in reader._hit_objects if h.kind in ("circle", "slider")]
    if presses and clickable:
        import numpy as np
        ho_ts = np.array([h.t for h in clickable], dtype=np.float64)
        # Press wall→beatmap mapping: wall_ms in sink is wall-time; we want to
        # compare to the beatmap time at that instant. For speed=1 the sink's
        # t and state.time_ms are aligned (minus negligible loop lag).
        offs = []
        for (t_wall, _, _) in presses:
            # Convert wall-ms-since-start to beatmap time using mock speed
            bt = reader.start_offset_ms + t_wall * reader.speed
            j = int(np.argmin(np.abs(ho_ts - bt)))
            offs.append(bt - ho_ts[j])
        offs = np.array(offs)
        print(f"[live] press_offset: mean={offs.mean():.2f}ms  std={offs.std():.2f}ms  "
              f"max|off|={np.max(np.abs(offs)):.1f}ms")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--bid", type=int, required=True)
    ap.add_argument("--user", type=int, required=True)
    ap.add_argument("--mods", type=int, default=0)
    ap.add_argument("--speed", type=float, default=1.0,
                    help="run faster than realtime (default 1.0). Test only.")
    ap.add_argument("--start-offset-ms", type=float, default=-1000.0,
                    help="where to start playback (negative = before first HO)")
    ap.add_argument("--poll-hz", type=float, default=200.0,
                    help="cursor sampling + dispatcher tick rate")
    ap.add_argument("--plan-hz", type=float, default=30.0,
                    help="how often to re-plan upcoming presses")
    ap.add_argument("--lookahead-ms", type=float, default=400.0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.mock:
        run_mock(args)
    else:
        print("[live] non-mock mode not wired yet (HemlockReader stub).")


if __name__ == "__main__":
    main()
