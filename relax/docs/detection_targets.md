# Detection targets

Public-tooling checks this project is explicitly trained to pass. Each is
implemented in `src/eval/metrics.py` with a specific threshold in
`src/eval/harness.py::DETECTION_THRESHOLDS`.

## cvUR (converted UR)

`cvUR = 10 * stddev(offset_ms_in_beatmap_time / speed_mult)` over all hits.

The `speed_mult` divisor is why we call it "converted": it expresses
tap-precision in effective real-time terms, so an HDDT play isn't flagged just
for being 1.5x tighter in beatmap-time.

Circleguard thresholds:
- cvUR < 60 → very suspicious
- cvUR < 50 → almost certainly relax

Humans in our top-50 data land ~70-160 depending on map difficulty.

## Hold-time histogram

Per-click `hold_ms = t_release - t_press`. Humans:
- distribution is heavy-tailed, lognormal-ish
- median 40-120ms, with a right tail to 300-500ms for slider holds
- bimodal is a bug tell (classic relax: all 2ms "holds")

Flag: `hold_mean < 20ms`.

## Edge-hit rate

Fraction of hits where cursor was at distance >= 0.85 * radius from circle
center at click time. Humans: typically 2-15% on streams, 20-50% on jump maps,
almost never >65%.

Flag: `edge_hit_rate@0.85 >= 0.65`.

## Miss distribution

Humans miss. The ratio of n_miss / n_total varies, but n_miss=0 on a hard map
with 300+ clicks is a flag. Perfect play is suspicious regardless of cvUR.

Flag: `miss_rate == 0 AND n_clicks > 200`.

## Same-key streaks

Humans alternate K1 <-> K2 in streams. Long same-key streaks (>= 20 in a row)
are a tell of a key-choice bug in a bot.

Flag: `max_same_key_streak >= 20` (per-replay, not concatenated).

## Attack-surface map vs our architecture

| detection             | classical-relax failure mode          | why our three-head arch beats it                  |
| --------------------- | ------------------------------------- | ------------------------------------------------- |
| cvUR floor            | constant jitter visible as narrow std | timing head learned mixture from humans           |
| edge_hit_rate         | fires when cursor is at center        | timing head is cursor-blind                       |
| no-click-on-miss      | no click when cursor leaves circle    | timing head fires anyway; outcome head says miss  |
| hold-time bimodal     | all 2ms clicks                        | hold sampled from per-user lognormal              |
| max_same_key_streak   | alternating K1/K2 forever             | key head LSTM learns burst/alternation structure  |
| perfect play          | 100% acc, 0 miss                      | outcome head learns stochastic miss under stress  |

## Attacks we don't yet defend against

- **Per-pattern cvUR curves** (circleguard's `UR` per-stream-vs-jump). Our
  single-number cvUR passes mean cvUR but doesn't ensure cvUR(streams) tracks
  cvUR(jumps) as a human's would. The timing head is pattern-aware (ctx window
  has gap info), so it probably does the right thing, but we haven't
  explicitly measured it.
- **Snaps-to-center**: an aimbot that writes to cursor_pos as well as clicks.
  We don't produce cursor (brief scope), so this check is trivially passed —
  because we reuse a HUMAN cursor.
- **Replay-replay**: duplicate frames across replays. We produce DIFFERENT
  click sequences each seed, so this is fine, but a defender comparing against
  a replay database would need many samples. No defense in-model.
- **Timestamp quantization**: osu! replay frames have a specific ms-integer
  quantization. We produce float offsets; a .osr writer would need to round.
  Not an issue for our evaluation (we score in float ms).
