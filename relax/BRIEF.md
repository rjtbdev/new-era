# new-era-relax — autonomous build brief

## Goal

Research prototype: train an ML system that generates osu!standard replay click
events (timing, key choice, hold time, hit/miss outcome) indistinguishable from
real human replays under public detection tooling (Circleguard, osuReplayAnalyzer,
hold-time histograms).

**Scope:** offline research only. No live-game interaction. No injection
pipeline. Proof-of-concept, with an honest evaluation vs the detection signals
documented in [relax detection research](../../Desktop/deobf/scratchpad/aqn_extracted/) and summarized in
`docs/detection_targets.md` (to be written).

## Thesis (one sentence)

> Rhythm-conditioned imitation learning with architectural decoupling of
> click-timing from cursor position produces replay statistics (cvUR,
> hold-time distribution, edge-hit rate, miss distribution, same-key streaks)
> that are indistinguishable from human replays against public detection
> tooling.

## Architecture (three heads)

1. **Timing head** — `P(click_time | upcoming objects, beatmap features, player embedding)`.
   **No cursor input.** Fires on music, not cursor. This is the key architectural choice.
2. **Outcome head** — `P(hit|50|100|300|miss | click_time, cursor_pos_at_click, object, pattern_difficulty)`.
   Takes sampled click time + user's actual cursor. Misses when a human would miss.
3. **Key choice head** — `P(key | pattern history, recent key history, BPM)`.
   K1 vs K2 with realistic same-key streak distributions.

Why decoupled: the research says every detection failure of classic relax traces
back to "firing depends on cursor-over-circle." Removing that coupling defeats
edge-hit detection, no-click-on-miss detection, and the four-frame-double-tap
re-entry tell simultaneously. See `docs/architecture.md` (to be written).

## Data

- **osu! API v1** (legacy key, in `.env`) — `get_replay` for raw `.osr` bytes.
- **osu! API v2** (OAuth, user to fill) — leaderboards + beatmap metadata.
- Target collection: top ~500 maps by playcount × top 50 replays each ≈ 25k replays.
  Stratified across BPM / star-rating / length for pattern-type coverage.

**User explicitly declined using their own replays** — top-50 leaderboard only.

## Milestones

- **P0 — Scaffolding + deps.** Python venv, repo layout, requirements pinned.
- **P1 — Data pipeline.** API clients, replay+beatmap parsers, per-click feature
  extraction (offset_ms, hold_ms, cursor_pos_at_click, edge_bias, hit/miss, key).
  Features cached as parquet.
- **P2 — Baselines.** Per-player Gaussian fits for offset/hold. Metric
  implementations: cvUR, hold-time histograms, edge-hit rate, same-key streaks.
  `baseline_relax.py` that samples from fits. Evaluation harness.
- **P3 — Timing head.** Small transformer (~1-10M params), trained to match
  per-player per-pattern cvUR curves. PyTorch.
- **P4 — Outcome head.** Trained on `(click_time, cursor, object) → outcome`.
- **P5 — Key choice head.** LSTM over key history.
- **P6 — End-to-end generator.** Wire heads together. Generate replays for
  held-out maps. Run through eval harness. Report detection-likelihood by metric.

## Files to produce

```
new-era-relax/
  .env                      # API keys (gitignored)
  BRIEF.md                  # this file
  README.md                 # how to use once working
  requirements.txt
  pyproject.toml            # if useful
  src/
    api/
      v1.py                 # get_replay, rate limited
      v2.py                 # OAuth, beatmap, leaderboard
    data/
      pull.py               # orchestrate bulk replay pulls → cache
      parse.py              # .osr → structured frames
      beatmap.py            # .osu → hit objects, SV, timing
      features.py           # per-click feature extraction
    models/
      timing.py
      outcome.py
      key_choice.py
      generator.py          # wraps all three
    eval/
      metrics.py            # cvUR, hold distr, edge-hit, streaks
      harness.py            # end-to-end evaluation
    baselines/
      gaussian.py
  data/
    raw/                    # downloaded .osr + .osu, gitignored
    processed/              # feature parquet, gitignored
  models/                   # trained weights, gitignored
  notebooks/                # exploration
  docs/
    architecture.md
    detection_targets.md
    data_notes.md
    decisions.md            # log of judgment calls flagged for user review
```

## Decision-flagging policy

Per user: make reasonable calls, flag each in `docs/decisions.md` and in a
project memory (`project_new_era_relax_decisions.md`). User reviews on
resume and can override.

## Scope notes / what NOT to do

- **No live-game interaction.** No osu! process reads/writes. No oa.dll
  interaction. Honor [project_oa_dll_goal_and_constraint.md](../../.claude/projects/c--Users-riley-Desktop-deobf/memory/project_oa_dll_goal_and_constraint.md).
- **No injection pipeline** this run. Model outputs → structured click events.
  Someone else (or a later session) can bridge to an actual input injector.
- **No fine-tuning for cheating a specific scoring system.** Pure human-replay
  imitation. Avoids an "aimbot cosplay" that overfits.

## Open questions to leave in `docs/decisions.md` as flagged

- Mod handling: train separately per mod combination, or encode mods as features?
  (Leaning: features. Flag.)
- Mania/CTB/Taiko: ignore (standard-only) or support? (Leaning: standard-only
  for v1. Flag.)
- Player embedding: pretrained per-user embeddings or learned jointly?
  (Leaning: learned jointly, small dim. Flag.)
- Augmentation: cursor noise for distribution-shift robustness? (Leaning: yes,
  mild. Flag.)

## Resume path for next session

Save memory `project_new_era_relax_status.md` at each phase boundary:
- What was completed.
- What's in-progress (and where to pick up).
- Any blockers.
- Any decisions logged.

Next session reads that memory first, then `docs/decisions.md`, then continues.
