# new-era-relax

Research prototype: ML system that generates osu!standard replay **click events**
(timing, key, hold, outcome) statistically indistinguishable from human replays
against public detection tooling (Circleguard-style checks: cvUR, hold-time
histograms, edge-hit rate, miss distribution, same-key streaks).

**Scope:** offline research only. No live-game interaction. No injection. Pure
model outputs → structured click-event data.

See `BRIEF.md` for the full architecture and milestone plan; `docs/decisions.md`
for choices flagged for later review; `PROGRESS.md` for the work log.

## Layout

```
src/
  api/      v1 (rate-limited legacy key) + v2 (OAuth, public scope)
  data/     pull.py (bulk puller) · parse.py (replay frames) ·
            beatmap.py (.osu) · features.py (per-click feature extraction) ·
            build_dataset.py (parquet shards)
  models/   timing.py (transformer) · outcome.py (MLP) · key_choice.py (LSTM) ·
            datasets.py · train_timing.py · train_outcome.py · train_key.py ·
            generator.py (wires heads + cursor + hold)
  baselines/ gaussian.py (per-player fits) · baseline_relax.py (sampler)
  eval/     metrics.py · harness.py
scripts/   run_eval.py
```

Data on disk is gitignored; artifacts stay in `data/raw/`, `data/processed/`,
`models/`.

## Setup

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

`.env` is already populated with `OSU_API_V1_KEY`, `OSU_CLIENT_ID`,
`OSU_CLIENT_SECRET`. Don't commit it.

## Pipeline

```powershell
# 1. Pull top-N maps × top-50 replays (long running)
.\.venv\Scripts\python.exe -m src.data.pull 500

# 2. Build per-click parquet shards (incremental, re-runnable)
.\.venv\Scripts\python.exe -m src.data.build_dataset

# 3. Train heads
.\.venv\Scripts\python.exe -m src.models.train_timing --epochs 6
.\.venv\Scripts\python.exe -m src.models.train_outcome --epochs 4
.\.venv\Scripts\python.exe -m src.models.train_key --epochs 4

# 4. Evaluate
.\.venv\Scripts\python.exe scripts\run_eval.py --k-maps 5
```

Results land in `docs/results.md` and `data/eval_report.json`.

## The core architectural idea

The three heads are **decoupled**:

- `timing.py` sees **only** upcoming beatmap objects + player embedding. It has
  no cursor input. It fires on rhythm.
- `outcome.py` takes the sampled click time plus the user's actual cursor at
  that moment, and produces 300/100/50/miss. If the cursor is far from the
  circle, it says miss — just like the game does.
- `key_choice.py` is autoregressive on the key history, blind to cursor and
  rhythm specifics.

This is what defeats:
  - **edge-hit detection** — the model doesn't aim-snap to the center, but it
    also doesn't fire *because* the cursor is near the center. It fires on
    rhythm, so it rides whatever cursor trajectory it was handed.
  - **no-click-on-miss detection** — the model issues clicks even when the
    cursor was elsewhere, exactly like a human whose aim slipped.
  - **UR-floor detection** — the timing head's variance comes from the human
    replays it learned from, not from a constant jitter added on top.

## Notes

- Replay frames are in **beatmap time**, not real time. DT replays record at
  the sped-up tempo but with the clock running at the beatmap's own rate. Our
  cvUR converts back to real time by dividing by the speed multiplier.
- Replay-frames parser must apply **negative** dt (used by osu! for the
  "ready" seek-back frame). Earlier draft clamped at 0 and silently misaligned
  every HD+DT replay by multiple seconds.
- HR replays mirror hit-object Y (y' = 384 - y). We mirror the object positions
  before alignment (we don't mutate the beatmap).
