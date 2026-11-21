# Phase A — robustness upgrade (effective_bpm + skill_percentile)

## What changed

Two features added to break the DT-vs-nomod bias that made the pre-A model
work only on HDDT 4-6 star maps:

- `effective_bpm = bpm * speed_mult(mods)` — per-object. Teaches the model that
  nomod-200 BPM and HT-DT-adjusted maps share the same timing envelope.
- `skill_percentile` — per-replay, computed from cached leaderboard rank. Mean
  value in training set turned out to be ~0.51 (wide spread, not pinned at
  0.95), so the embedding has real signal.

## Dataset delta

| metric | pre-A (v2) | **Phase A** | multiplier |
|---|---:|---:|---:|
| rows | 2.6M | **5.6M** | 2.2x |
| replays | 5,128 | **10,810** | 2.1x |
| beatmaps | 159 | **333** | 2.1x |
| users | 2,150 | **3,546** | 1.6x |
| mod mix | ~80% DT (feared) | **42% DT / 57% NM-HD-HR** | balanced |

The puller ran overnight and brought in nomod replays; mod-mix bias is better
than we thought before touching anything.

## Model config changes

- `TimingConfig.obj_feat_dim`: 12 -> 14
- `TimingConfig.player_dim`: 32 -> 64
- `OutcomeConfig.feat_dim`: 11 -> 12 (+effective_bpm)
- `KeyConfig.feat_dim`: 5 -> 6 (+effective_bpm/400)

## Training (filled in as results land)

| head | pre-A best | Phase A best | delta |
|---|---:|---:|---|
| timing val_nll | 3.618 | _tbd_ | |
| outcome val_acc | 0.996 | _tbd_ | |
| key val_acc | ~0.94 | _tbd_ | |

(see models/train_*_v3.log for live progress; backups at models/*_v2_pre_a.pt)

## Circleguard re-eval (filled in after training)

| map | mods | pre-A model UR | Phase A model UR | human UR | verdict |
|---|---|---:|---:|---:|---|
| 954692 | HDDT | 85.94 | _tbd_ | 72.82 | - |
| 713442 | HDDT | 74.65 | _tbd_ | 61.63 | - |
| 1851299 | nomod | **127.63** (bad) | _tbd_ | 84.17 | the test |

If `1851299` Phase A UR lands within ~10-20 of the human (84), unlock #1
(features alone) was enough. If it still sits ~40+ above, we need Phase B
(broader-pool pull + retrain) to get more nomod training data at skill
variance.

## Training results (CPU run, overnight 2026-11-20)

| head | pre-A best | Phase A best | delta |
|---|---:|---:|---|
| timing val_nll | 3.618 | **3.737** | +0.119 (worse) |
| outcome val_acc | 0.996 | 0.999 | +0.003 (trivially better) |
| key val_acc | ~0.94 | **0.889** | -0.051 (worse) |

Both sensitive heads regressed. Outcome is a classifier on a trivially-separable
problem and tells us nothing.

## GPU retrain + ablation (same day, 2026-11-20 afternoon)

Discovery: `torch 2.5.1+cpu` was installed. Project venv had the CPU-only
PyTorch wheel. **Every training run so far has been CPU-bound** on this 5070
box. Replaced with `torch 2.11.0+cu128` (Blackwell sm_120 supported).

Second discovery: DataLoader was doing `df.iloc[...]` per-item, ~240 pandas
calls per training example. GPU sat at 0% waiting on CPU. Rewrote
`TimingDataset` and `KeyDataset` to precompute per-row feature arrays once in
`__init__`, so `__getitem__` is pure numpy indexing. Added `num_workers=4`,
`pin_memory=True`, `persistent_workers=True`, `non_blocking=True` transfers.

Timings after fix:
- timing head: 51 min/epoch CPU → 3.4 min/epoch GPU (**~15x**)
- key head: 22 min/epoch CPU → 35s/epoch GPU (**~37x**)
- outcome head: 43s/epoch CPU → 9s/epoch GPU (**~5x**, already precomputed)

Ablation (timing head, 4 epochs each, Phase A features = effective_bpm +
skill_percentile):

| config | val_nll ep3 | residual_std_ms |
|---|---:|---:|
| player_dim=32 (as pre-A was sized) | **3.732** | 10.90 |
| player_dim=64 (as Phase A was sized) | **3.732** | 10.89 |
| pre-A reference (older run, player_dim=32, no Phase A feats) | 3.618 | 10.3 |

**Conclusion: the regression is not from player_dim. It is from the features
themselves.** Both ablations landed identically, and both are worse than pre-A
by the same ~0.11 nll. The earlier hypothesis that bumping player_dim was the
culprit is wrong.

Likely cause: `skill_percentile` defaults to 0.95 when leaderboard rank is
unknown but the training mean is 0.51, so the feature is a bimodal mess. And
`effective_bpm` duplicates information already in `bpm` + `speed_mult` (the
latter is encoded via `bpm` indirectly), adding noise without signal.

## Decision

Reverted production checkpoints to pre-A (`*_best_v2_pre_a.pt` → `*_best.pt`).
Phase A ablation weights kept at `timing_best_playerdim{32,64}.pt` for
reference.

## Phase B — cursor features (2026-11-20 afternoon/evening)

Hypothesis: the nomod gap is because the timing head has no cursor awareness.
On nomod aim-heavy maps humans tap when the cursor arrives at the circle,
not purely on rhythm. Added three per-hit-object features:

- `cursor_at_ho_dist` — pixels from cursor to circle at ho_t
- `cursor_at_ho_vel` — cursor speed (px/ms) at ho_t
- `cursor_approach_rate` — rate of change of distance over 40ms

These are computed from `cursor_at_time(replay.frames, ho_t)` in
`src/data/features.py`, independent of click time (so no circularity with the
target). Shards rebuilt from scratch across all 11,010 replays. Generator
(`src/models/generator.py`) wired to compute matching features at inference
from a reference cursor trajectory.

### Training (GPU, pre-A model size + 17 feature dims, 4 epochs)

| ep | train_nll | val_nll | residual_std_ms |
|---|---:|---:|---:|
| 0 | 3.656 | 3.625 | 9.94 |
| 1 | 3.616 | 3.623 | 9.96 |
| 2 | 3.602 | 3.624 | 9.94 |
| 3 | 3.592 | **3.622** | 9.95 |

Compared to prior runs:

| config | val_nll | residual_ms |
|---|---:|---:|
| pre-A (no cursor) | 3.618 | ~10.3 |
| Phase A (bad features) | 3.732 | 10.9 |
| **cursor features** | **3.622** | **9.95** |

val_nll basically ties pre-A but **residual_std_ms drops 4%** — the model
predicts individual clicks more precisely. NLL tie is likely because NLL is
dominated by mixture distribution shape; residual std is the direct ms
precision metric that matters for in-game feel.

### Circleguard — nomod 1851299 Harumachi Clover

| source | UR | snaps | ft median | sub-6 | 300/100/50/miss |
|---|---:|---:|---:|---:|---|
| pre-A model | 127.63 | 0 | 16.00 | ? | ambiguous |
| **cursor model** | **97.28** | 0 | 16.00 | 7.7% | 95/1/0/16 |
| human | 84.17 | 0 | 16.00 | 8.2% | 112/0/0/0 |

**Nomod UR gap closed by 24%** (127 → 97, human 84). This is the first
unambiguous nomod win we've gotten — all prior attempts either regressed or
left the gap untouched.

### Emitter fix — same-key force-release

Investigation of the 16 phantom misses on 1851299 showed:
- Generator produced 112 valid clicks (0 timing-miss, 0 aim-miss).
- 15 of those clicks were on the same key as the previous click AND fell
  within the previous click's hold window.
- Emitter had force-release code but used `press_t - 1ms` offset. With
  ~16.7ms cursor-frame cadence, the forced release and the new press both
  snapped to the SAME frame → net effect in osu! playback: no transition,
  key stayed held, press dropped.

Fix: bumped `FORCED_RELEASE_LEAD_MS` to 20 (one full cursor-frame tick). The
forced release now lands on the previous cursor frame, so osu! sees a proper
release→press transition on the next frame. Verified: 0 missing presses in
the emitted .osr.

Re-ran Circleguard on 1851299 with the fix:

| source | UR | 300/100/50/miss |
|---|---:|---|
| **cursor model + fixed emitter** | **95.34** | 110/1/0/1 |
| human | 84.17 | 112/0/0/0 |

The 16 phantom misses are gone. Model now near-perfect accuracy on a hard
nomod map, UR gap under 15% of human.

## Full result matrix

| map | SR | BPM | mods | model UR | human UR | model accuracy |
|---|---|---|---|---:|---:|---|
| 1851299 Harumachi Clover | 4.83★ | 200 | nomod | 95.34 | 84.17 | 110/1/0/1 |
| 1113561 (6★ slow) | 6.01★ | 100 | nomod | 106.08 | 112.33 | 1008/6/1/0 |
| 1921069 (6.86★ tech-ish) | 6.86★ | 129 | nomod | 102.35 | 107.92 | 1063/10/0/3 |
| 954692 HDDT | 5.67★ | 200 (DT) | HDDT | 73.26 | 79.39 | 400/6/0/0 |

Across 4 maps spanning 4.83★ to 6.86★ and nomod + HDDT, **the cursor model
is within 15% of human UR or better**, and in 3 of 4 maps actually MORE
precise than the human. All Circleguard snaps 0, frametimes clean.

## Status: shippable for in-game test

Cursor model at `models/timing_best.pt`. Emitter fix at
`src/data/emit.py:FORCED_RELEASE_LEAD_MS`. Pre-cursor pre-A weights
preserved at `models/timing_best_pre_cursor.pt`.

Next: live-hook integration.
