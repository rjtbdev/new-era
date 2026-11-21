# new-era-relax live inference service

End-to-end pipeline: osu! state → feature extraction → model inference →
scheduled key presses. Everything AI-side is Python+torch. The hemlock
side (memory reads and keyboard-input writes) is stubbed until the
in-game sigs are resolved.

## Files

```
__init__.py          overview
osu_reader.py        GameState dataclass + Reader protocol
mock_reader.py       drives state from a .osu + a reference cursor .bin
hemlock_reader.py    stub for live reads via hemlock hypercalls
feature_builder.py   17-dim per-hit-object feature window builder
model_service.py    timing + key heads → PressDecision list
dispatcher.py        schedules presses at the right beatmap time
sink.py              MockSink (logs) + HemlockSink (stub)
main.py              entry point with --mock mode
```

## Run (mock, offline validation)

```bash
python -u live/main.py --mock --bid 1851299 --user 673214 --mods 0 --verbose
```

This replays the 1851299 Harumachi Clover beatmap against Roro Rosset's
cursor trajectory in real time, running the model inference loop, and
logging every would-be K1/K2 press. No osu! or hemlock involved — pure
sanity check that the AI side of the pipeline works end-to-end.

Useful args:
- `--speed 2.0` — run 2x faster than realtime (test only)
- `--poll-hz 200` — cursor sampling + dispatcher tick rate
- `--plan-hz 30` — inference re-plan frequency
- `--lookahead-ms 400` — how far ahead to pre-plan presses

## Status

- [x] Reader/Sink interfaces defined
- [x] Mock reader working
- [x] Feature builder (ctx_past=8, ctx_future=8, 17-dim) matching training
- [x] Model service loading timing + key heads
- [x] Press scheduler + dispatcher
- [x] End-to-end smoke test: all 112 presses emit correctly on 1851299
- [ ] HemlockReader — needs sigs for: time, cursor xy, running_gamemode,
      hit_object_manager. Starting points in memory
      `project_aqn_osu_sigs_and_offsets.md`.
- [ ] HemlockSink — needs sig for: osu!'s current-frame keyboard-input
      field (write target), OR the function osu! calls to read keyboard
      state (hook target).
- [ ] Port style-matching (per-ref UR target) — generator has it; live
      service should look up `user_map_offset_std` the same way.
- [ ] Per-kind hold times in dispatcher (short for circles, long for
      sliders). Currently fixed 85ms default.
- [ ] HTTP+WS UI like osu_menu for toggle / intensity / status.

## Design notes

- **Latency budget.** Poll 200Hz = 5ms ticker. Model inference per HO is
  ~2-5ms on CPU. Planner runs at 30Hz = 33ms period. Total: a press
  decision is made 400ms ahead of ho_t, so even a 100ms planning stall
  wouldn't miss notes.
- **Presses are planned once per HO, sticky.** Prevents oscillation if
  the state changes between ticks.
- **Cursor history** is kept in a 500ms ring buffer inside FeatureBuilder
  so cursor-at-ho_t features can look back for velocity + approach rate.
