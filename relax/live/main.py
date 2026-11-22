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
from typing import Optional

import torch  # DLL init order on Windows

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from live.ao_reader import AoReader, _HypercallError
from live.dispatcher import Dispatcher
from live.hook_data_poller import HookDataPoller
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
    """AoReader + AoSink. HOs come from live HOM (if --use-hom) or .osu file.

    Does NOT re-plan every tick; the model plans once from the HO list,
    dispatcher fires presses when osu!'s clock reaches each one.
    """
    osu_path = None
    if args.bid is not None:
        osu_path = PROJECT / "data" / "raw" / "beatmaps" / f"{args.bid}.osu"
        if not osu_path.exists():
            if not args.use_hom:
                print(f"[live] missing {osu_path} and --use-hom not set")
                return 1
            osu_path = None  # fall back to HOM-only

    if osu_path is None and not args.use_hom:
        print("[live] either --bid (with map file present) or --use-hom required")
        return 1

    reader = AoReader(
        process_name="osu!.exe",
        osu_path=osu_path,
        mods=args.mods,
        beatmap_id=args.bid,
        player_id=args.user,
        use_hom=args.use_hom,
    )
    print(f"[live] attaching AoReader to osu!.exe "
          f"(use_hom={args.use_hom}, osu_path={'.osu' if osu_path else 'none'})...")
    try:
        reader.attach()
    except _HypercallError as e:
        print(f"[live] attach failed: {e}")
        if args.use_hom:
            print("[live] HINT: --use-hom requires osu! to already be IN a map.")
        return 1
    print(f"[live] attached. VAs: {reader.resolved_vas()}")
    print(f"[live] HOs loaded: {len(reader._hit_objects)} "
          f"(source: {'HOM' if args.use_hom else '.osu file'})")

    sink = AoSink(verbose=args.verbose, dry_run=dry_run)
    print(f"[live] binding AoSink (dry_run={dry_run})...")
    try:
        sink.attach()
    except Exception as e:
        err_str = str(e)
        if ("no multi-insn hook found" in err_str or "no active hook with 8 insns" in err_str) and not dry_run:
            # Auto-install the key_inject hook pair and retry.
            import subprocess
            key_inject = r"C:\Users\riley\projects\new-era\tools\key-inject\key_inject.exe"
            print(f"[live] key_inject hooks not installed; running `{key_inject} install`...")
            r = subprocess.run([key_inject, "install"], capture_output=True, text=True, timeout=60)
            print(r.stdout)
            if r.returncode != 0:
                print(f"[live] auto-install failed: {r.stderr}")
                return 1
            try:
                sink.attach()
            except Exception as e2:
                print(f"[live] sink attach still failing after install: {e2}")
                return 1
        else:
            print(f"[live] sink attach failed: {e}")
            return 1

    model = ModelService(
        timing_ckpt=PROJECT / "models" / "timing_best.pt",
        key_ckpt=PROJECT / "models" / "key_best.pt",
        device="cpu",
        rules=RuleConfig(seed=42),
    )

    # -----------------------------------------------------------------
    # Install/discover cursor_capture hook if cursor gating is enabled
    # -----------------------------------------------------------------
    cursor_va: Optional[int] = None
    if args.cursor_gate:
        import re as _re
        import subprocess as _sp
        cursor_tool = r"C:\Users\riley\projects\new-era\tools\cursor-capture\cursor_capture.exe"
        hook_tool = r"C:\Users\riley\projects\ao\cli\hook_tool.exe"

        # Try to find existing cursor_capture hook in status; if none, install.
        def _find_cursor_hook():
            r = _sp.run([hook_tool, "status"], capture_output=True, text=True, timeout=10)
            # look for an active 4-insn hook (cursor_capture's script len = 4)
            row_re = _re.compile(
                r"^\s*(\d+)\s+(\d)\s+0x[0-9a-fA-F]+\s+0x([0-9a-fA-F]+)\s+\d+\s+(\d+)",
                _re.M)
            for m in row_re.finditer(r.stdout):
                _slot, active, va_hex, insns = m.groups()
                if int(active) == 1 and int(insns) == 4:
                    return int(va_hex, 16)
            return None

        cursor_va = _find_cursor_hook()
        if cursor_va is None:
            print(f"[live] installing cursor_capture hook...")
            r = _sp.run([cursor_tool, "install"], capture_output=True, text=True, timeout=60)
            print(r.stdout)
            if r.returncode != 0:
                print(f"[live] cursor_capture install failed: {r.stderr}")
                return 1
            cursor_va = _find_cursor_hook()
            if cursor_va is None:
                print(f"[live] WARN: cursor_capture installed but not found in status; disabling gating")
                args.cursor_gate = False
        if cursor_va is not None:
            print(f"[live] cursor_capture hook at va=0x{cursor_va:X}")

    # -----------------------------------------------------------------
    # Dispatcher: wire cursor_fn and ho_getter if gating
    # -----------------------------------------------------------------
    poller_cursor_va = cursor_va if args.cursor_gate else None

    # Create dispatcher. If cursor gating is on, we'll re-create below with
    # cursor_fn/ho_getter wired in (after we have the poller running).
    dispatcher = Dispatcher(sink, press_lead_ms=args.press_lead_ms)

    print(f"[live] waiting for a CLOCK RESET (click Play in osu! now)...")
    # Watch for osu's `time` field to jump BACKWARD -- that's the song
    # restarting / a new play starting. Only then do we plan. Avoids
    # the "preview is playing at t=68s past end" trap.
    #
    # Fallback: if no reset seen in 60s but clock is advancing and well
    # before map end, accept it (user may have already started playing).
    last_time = None
    reset_detected = False
    tfirst_wait = time.perf_counter()
    bm_first_ho = min((h.t for h in reader._hit_objects), default=0.0)
    bm_last_ho  = max((h.t for h in reader._hit_objects), default=0.0)
    while True:
        state = reader.read()
        if last_time is not None and state.time_ms + 500 < last_time:
            reset_detected = True
            print(f"[live] clock RESET detected ({last_time:.0f} -> {state.time_ms:.0f})")
            break
        if state.time_ms < bm_first_ho + 500 and not state.is_paused:
            # We're near the start of the map. Good enough.
            print(f"[live] clock near map start (time={state.time_ms:.0f}ms, first_ho={bm_first_ho:.0f})")
            break
        if time.perf_counter() - tfirst_wait > 60.0:
            print(f"[live] no reset in 60s; proceeding at time={state.time_ms:.0f}")
            break
        last_time = state.time_ms
        time.sleep(0.02)

    print(f"[live] clock live (time={state.time_ms:.0f}ms, pause={state.is_paused}, "
          f"mode={'std' if state.is_playing else 'non-std'})")

    # Settle the time reader: osu's time field can briefly read a huge
    # (negative-signed, uninitialized) value immediately after a reset.
    # Wait until we see a sane value before planning.
    bm_end_est = max((h.t for h in reader._hit_objects), default=0.0) + 5000.0
    settle_start = time.perf_counter()
    while time.perf_counter() - settle_start < 2.0:
        s = reader.read()
        if 0 <= s.time_ms <= bm_end_est and not s.is_paused:
            state = s
            break
        time.sleep(0.05)
    else:
        print(f"[live] WARN: time field never settled in 2s; proceeding anyway")

    # Latency compensation: planning takes real time (model forward per HO,
    # 234 HOs at a few ms each = a second or more total). Snapshot the
    # planning-start wall time so we can offset by elapsed wall-ms before
    # submitting.
    planning_start_wall = time.perf_counter()
    planning_start_bt = state.time_ms

    # Plan ALL presses once from the static HO list. ModelService.plan with
    # lookahead=inf returns every remaining clickable press.
    decisions = model.plan(state, lookahead_ms=float("inf"))
    planning_wall_ms = (time.perf_counter() - planning_start_wall) * 1000.0
    print(f"[live] planned {len(decisions)} presses in {planning_wall_ms:.0f}ms wall")
    if len(decisions) == 0:
        print("[live] nothing to play -- clock may be past map end. exiting.")
        sink.detach()
        return 1

    # Drop any decisions whose press_time is already in osu's past
    # (planning took wall-ms; osu's clock moved during planning).
    # Current osu time approximately = planning_start_bt + planning_wall_ms.
    now_bt_est = planning_start_bt + planning_wall_ms
    pre = len(decisions)
    decisions = [d for d in decisions if d.press_time_ms > now_bt_est + 10.0]
    if pre != len(decisions):
        print(f"[live] dropped {pre - len(decisions)} decisions already past "
              f"(now_est={now_bt_est:.0f}ms)")

    dispatcher.submit(decisions)

    # ---- Tick loop with HookDataPoller anchor + wall-clock extrapolation ----
    #
    # HookDataPoller drains the shared ph_log ring and splits entries by
    # VA: key_inject hook B emits time, cursor_capture hook emits xy.
    # Main tick loop: extrapolate bt from latest time anchor; read latest
    # cursor on each tick for cursor-gating in Dispatcher.
    poll_hz = args.poll_hz
    tick_dt = 1.0 / poll_hz
    bm_end = max((h.t for h in reader._hit_objects), default=0.0) + 5000.0
    speed_mult = state.speed_mult

    # Resolve key_inject hook B VA (where time is logged from) via hook_tool
    # status — it's the 11-insn hook.
    import subprocess as _sp, re as _re
    _r = _sp.run([r"C:\Users\riley\projects\ao\cli\hook_tool.exe", "status"],
                 capture_output=True, text=True, timeout=10)
    _row_re = _re.compile(
        r"^\s*(\d+)\s+(\d)\s+0x[0-9a-fA-F]+\s+0x([0-9a-fA-F]+)\s+\d+\s+(\d+)",
        _re.M)
    time_hook_va = None
    for _m in _row_re.finditer(_r.stdout):
        _slot, active, va_hex, insns = _m.groups()
        if int(active) == 1 and int(insns) >= 8:
            time_hook_va = int(va_hex, 16)
            break
    if time_hook_va is None:
        print("[live] FATAL: couldn't find key_inject hook B VA in status")
        return 1

    hdp = HookDataPoller(
        time_va=time_hook_va,
        cursor_va=poller_cursor_va,
        max_valid_bt=bm_end,
        drain_interval_s=0.03,
    )
    pre_drained = hdp.drain_startup()
    print(f"[live] startup: discarded {pre_drained} stale log entries")
    hdp.start()
    print(f"[live] HookDataPoller started (time_va=0x{time_hook_va:X}, "
          f"cursor_va={'0x%X' % poller_cursor_va if poller_cursor_va else 'None'})")

    # Wait for first sane time sample
    tp_wait_start = time.perf_counter()
    while hdp.latest_time_ms() is None and time.perf_counter() - tp_wait_start < 2.0:
        time.sleep(0.02)
    first_tp = hdp.latest_time_ms()
    if first_tp is None:
        print("[live] WARN: HookDataPoller got no sane time samples in 2s. Falling back to state.")
        anchor_perf = time.perf_counter()
        anchor_bt   = float(state.time_ms)
        anchor_source = "state"
    else:
        tp_bt, tp_perf = hdp.latest_time()
        anchor_perf = tp_perf
        anchor_bt   = tp_bt
        anchor_source = "HookDataPoller"
        print(f"[live] first time sample: bt={tp_bt:.0f}ms "
              f"(discarded {hdp.discarded_count()} bad)")

    # Build ho_getter and cursor_fn for the Dispatcher
    if args.cursor_gate and poller_cursor_va is not None:
        from src.data.features import _circle_radius_px
        # CS from session_fields (hardcoded 4.0 under use_hom; OK for now)
        cs = reader._session_fields.get("cs", 4.0)
        circle_radius = _circle_radius_px(cs, args.mods)
        # Build (x, y, r) lookup by HO index
        ho_pos = {h.idx: (h.x, h.y, circle_radius) for h in reader._hit_objects}

        def _ho_getter(center_idx: int):
            return ho_pos.get(center_idx)
        def _cursor_fn():
            xy, _ = hdp.latest_cursor()
            return xy
        # Replace the plain Dispatcher with a cursor-aware one
        dispatcher = Dispatcher(
            sink,
            press_lead_ms=args.press_lead_ms,
            cursor_fn=_cursor_fn,
            ho_getter=_ho_getter,
            cursor_gate_window_ms=args.cursor_gate_window_ms,
            cursor_radius_mult=args.cursor_radius_mult,
        )
        # re-submit to the new dispatcher
        dispatcher.submit(decisions)
        print(f"[live] cursor gating ENABLED: window=±{args.cursor_gate_window_ms:.0f}ms, "
              f"radius_mult={args.cursor_radius_mult:.2f}, circle_r={circle_radius:.1f}px")

    last_anchor_tp_bt = anchor_bt
    start_bt = anchor_bt
    tick_count = 0
    last_dbg_wall = time.perf_counter()

    while True:
        wall = time.perf_counter()
        sleep_dt = tick_dt - (wall - anchor_perf) % tick_dt
        if sleep_dt > 0:
            time.sleep(sleep_dt)

        now_perf = time.perf_counter()
        tick_count += 1

        # Check HookDataPoller for a newer anchor
        tp_bt, tp_perf = hdp.latest_time()
        if tp_bt is not None and tp_bt != last_anchor_tp_bt:
            # New sample. Re-anchor if monotonic (ignore anomalies).
            if tp_bt >= last_anchor_tp_bt - 100:  # allow small jitter
                anchor_perf = tp_perf
                anchor_bt   = tp_bt
                last_anchor_tp_bt = tp_bt

        bt = anchor_bt + (now_perf - anchor_perf) * 1000.0 * speed_mult

        # Sanity clamp (shouldn't happen with poller but defensive).
        # Also: if the anchor itself is garbage (shouldn't happen since
        # TimePoller filters, but safety-net), treat as bad.
        if bt > bm_end or bt < 0:
            # Reset anchor to a safe value to prevent cascade of bad bt's.
            # Use the planning_start_bt (osu clock at plan time) as a floor.
            if anchor_bt > bm_end or anchor_bt < 0:
                anchor_perf = now_perf
                anchor_bt   = planning_start_bt
                if args.verbose:
                    print(f"[live] RESET anchor to planning_start_bt={planning_start_bt:.0f} "
                          f"(was garbage: {anchor_bt:.0f})")
            continue

        # Periodic status line
        if args.verbose and (now_perf - last_dbg_wall) > 1.0:
            last_dbg_wall = now_perf
            deltas = dispatcher._press_deltas
            if deltas:
                import statistics
                recent = deltas[-100:]
                mean_d = statistics.mean(recent)
                std_d = statistics.stdev(recent) if len(recent) > 1 else 0.0
                lag_str = f" lag_mean={mean_d:+.0f}ms std={std_d:.0f}ms"
            else:
                lag_str = ""
            print(f"[live] t={bt:>6.0f}ms  pending={len(dispatcher._pending)}  "
                  f"to_release={len(dispatcher._to_release)}  ticks={tick_count}  "
                  f"time_n={hdp.time_count()}  cursor_n={hdp.cursor_count()}  "
                  f"src={anchor_source}{lag_str}")

        dispatcher.tick(bt)

        if (not dispatcher._pending and not dispatcher._to_release
                and bt > start_bt + 2000):
            break

    hdp.stop()
    sink.detach()
    print(f"[live] done. ticks={tick_count} time_n={hdp.time_count()} cursor_n={hdp.cursor_count()}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--mock", action="store_true", help="offline mock run")
    mode.add_argument("--live", action="store_true", help="AoReader + AoSink (real osu!)")
    mode.add_argument("--dry-live", action="store_true",
                      help="AoReader + AoSink in dry-run mode (logs CFG writes, doesn't apply)")
    ap.add_argument("--bid", type=int, default=None,
                    help="beatmap id; optional if --use-hom (HOs come from memory)")
    ap.add_argument("--user", type=int, required=True)
    ap.add_argument("--mods", type=int, default=0)
    ap.add_argument("--use-hom", action="store_true",
                    help="(live only) read hit objects from osu!'s live HOM; "
                         "no .osu file required. osu! must be IN a map at attach time.")
    ap.add_argument("--speed", type=float, default=1.0,
                    help="(mock only) run faster than realtime")
    ap.add_argument("--start-offset-ms", type=float, default=-1000.0,
                    help="(mock only) where to start playback")
    ap.add_argument("--poll-hz", type=float, default=200.0,
                    help="cursor sampling + dispatcher tick rate")
    ap.add_argument("--plan-hz", type=float, default=30.0,
                    help="(mock only) how often to re-plan upcoming presses")
    ap.add_argument("--lookahead-ms", type=float, default=400.0)
    ap.add_argument("--press-lead-ms", type=float, default=0.0,
                    help="(live only) fire presses N ms earlier than planned. "
                         "With TimePoller the time anchor is kHz-fresh and "
                         "AoSink per-press latency is the only cost (~30-50ms). "
                         "Tune to taste; 0-50ms is the expected range.")
    ap.add_argument("--cursor-gate", action="store_true",
                    help="(live only) gate each press on cursor-is-on-HO. "
                         "Fires as soon as cursor enters the circle within the "
                         "gate window around the scheduled press_time. Requires "
                         "cursor_capture hook; auto-installs if missing.")
    ap.add_argument("--cursor-gate-window-ms", type=float, default=60.0,
                    help="(live only) half-width of cursor gate window around "
                         "press_time. Within this window we wait for cursor; "
                         "outside we fire anyway. Default 60ms.")
    ap.add_argument("--cursor-radius-mult", type=float, default=1.0,
                    help="(live only) multiplier on circle radius for the "
                         "'cursor on HO' check. 1.0 = exact radius; 1.5 = more "
                         "forgiving. Default 1.0.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    if args.mock:
        if args.bid is None:
            ap.error("--mock requires --bid")
        run_mock(args)
    elif args.dry_live:
        sys.exit(run_live(args, dry_run=True))
    elif args.live:
        sys.exit(run_live(args, dry_run=False))


if __name__ == "__main__":
    main()
