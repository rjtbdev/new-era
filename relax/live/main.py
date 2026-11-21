"""Live service entry point.

Three modes:

  --mock         offline: MockReader (replays a known cursor trace from disk)
                 + MockSink (logs press/release events, doesn't touch osu!).
                 Validates the ML side end-to-end without any HV.

  --live         online: AoReader (reads osu! time via ao HV, static .osu HOs)
                 + AoSink (forces K1/K2 via the key_inject hook pair).
                 Pre-requisites:
                   1. ao HV loaded (`load ao.efi`, reboot).
                   2. osu! running with the target map selected / loaded.
                   3. `key_inject install` has been run so hooks are armed.
                 This mode PARSES THE MAP AHEAD OF TIME (not from live
                 memory); we track osu's live clock + is_paused and fire
                 pre-planned presses when the beatmap time reaches each HO.

  --dry-live     same as --live but AoSink is in dry_run mode: logs what it
                 WOULD send but doesn't touch any CFG. Safe wet-feet test.

Usage:
    py -3 -u live/main.py --mock --bid 1851299 --user 673214 --mods 0
    py -3 -u live/main.py --live --bid 1851299 --user 673214 --mods 0 --verbose
    py -3 -u live/main.py --dry-live --bid 1851299 --user 673214 --mods 0
"""
from __future__ import annotations

import argparse
import pathlib
import sys
import time

import torch  # DLL init order on Windows

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from live.ao_reader import AoReader, _HypercallError
from live.dispatcher import Dispatcher
from live.model_service import ModelService
from live.mock_reader import MockReader
from live.sink import AoSink, MockSink
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


def run_live(args, *, dry_run: bool):
    """AoReader + AoSink. Map parsed from disk, live osu! clock drives timing.

    Does NOT re-plan every tick (unlike mock); the model plans once from the
    static HO list, dispatcher fires presses when osu!'s clock reaches each
    one. Simpler + lower latency than the mock path. Re-planning per tick
    is unnecessary when HOs are known up front.
    """
    osu_path = PROJECT / "data" / "raw" / "beatmaps" / f"{args.bid}.osu"
    if not osu_path.exists():
        print(f"[live] missing {osu_path}")
        return 1

    reader = AoReader(
        process_name="osu!.exe",
        osu_path=osu_path,
        mods=args.mods,
        beatmap_id=args.bid,
        player_id=args.user,
    )
    print(f"[live] attaching AoReader to osu!.exe...")
    try:
        reader.attach()
    except _HypercallError as e:
        print(f"[live] attach failed: {e}")
        return 1
    print(f"[live] attached. VAs: {reader.resolved_vas()}")

    sink = AoSink(verbose=args.verbose, dry_run=dry_run)
    print(f"[live] binding AoSink (dry_run={dry_run})...")
    try:
        sink.attach()
    except Exception as e:
        print(f"[live] sink attach failed: {e}")
        return 1

    model = ModelService(
        timing_ckpt=PROJECT / "models" / "timing_best.pt",
        key_ckpt=PROJECT / "models" / "key_best.pt",
        device="cpu",
        rules=RuleConfig(seed=42),
    )
    dispatcher = Dispatcher(sink)

    print(f"[live] waiting for play_mode=0 (std) and time to start advancing...")
    # Wait until osu! is actually in a std map and clock is moving.
    last_time = None
    stable_advances = 0
    while True:
        state = reader.read()
        if state.is_paused:
            time.sleep(0.1)
            continue
        if state.time_ms != (last_time if last_time is not None else -999999):
            stable_advances += 1 if last_time is not None else 0
            last_time = state.time_ms
            if stable_advances >= 3:
                break
        time.sleep(0.05)

    print(f"[live] clock live (time={state.time_ms:.0f}ms, pause={state.is_paused}, "
          f"mode={ 'std' if state.is_playing else 'non-std'})")

    # Plan ALL presses once from the static HO list. ModelService.plan with a
    # giant lookahead returns every clickable press.
    decisions = model.plan(state, lookahead_ms=float("inf"))
    print(f"[live] planned {len(decisions)} presses across the map")
    dispatcher.submit(decisions)

    # Tick loop: poll time, fire presses whose press_time_ms has been reached.
    poll_hz = args.poll_hz
    tick_dt = 1.0 / poll_hz
    last_wall = time.perf_counter()
    start_bt = state.time_ms

    while True:
        wall = time.perf_counter()
        if wall - last_wall < tick_dt:
            time.sleep(max(0, tick_dt - (wall - last_wall)))
        last_wall = time.perf_counter()

        state = reader.read()

        # Stop if osu! ends the map (play_mode changes) or user quits.
        if state.is_paused:
            if args.verbose:
                print(f"[live] paused at {state.time_ms:.0f}ms; holding")
            time.sleep(0.1)
            continue

        dispatcher.tick(state.time_ms)

        # Simple exit: when we've been past all HOs for 2 seconds
        if (not dispatcher._pending and not dispatcher._to_release
                and state.time_ms > start_bt + 2000):
            # All planned presses fired; keep going for 2s in case of late ticks
            break

    # Cleanup: release any lingering CFG overrides.
    sink.detach()
    print(f"[live] done.")
    return 0


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mock", action="store_true", help="offline mock run")
    mode.add_argument("--live", action="store_true", help="AoReader + AoSink (real osu!)")
    mode.add_argument("--dry-live", action="store_true",
                      help="AoReader + AoSink in dry-run mode (logs CFG writes, doesn't apply)")
    ap.add_argument("--bid", type=int, required=True)
    ap.add_argument("--user", type=int, required=True)
    ap.add_argument("--mods", type=int, default=0)
    ap.add_argument("--speed", type=float, default=1.0,
                    help="(mock only) run faster than realtime")
    ap.add_argument("--start-offset-ms", type=float, default=-1000.0,
                    help="(mock only) where to start playback")
    ap.add_argument("--poll-hz", type=float, default=200.0,
                    help="cursor sampling + dispatcher tick rate")
    ap.add_argument("--plan-hz", type=float, default=30.0,
                    help="(mock only) how often to re-plan upcoming presses")
    ap.add_argument("--lookahead-ms", type=float, default=400.0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.mock:
        run_mock(args)
    elif args.dry_live:
        sys.exit(run_live(args, dry_run=True))
    elif args.live:
        sys.exit(run_live(args, dry_run=False))


if __name__ == "__main__":
    main()
