# Decisions — flagged for user review

Each entry = a call I made so the build could proceed. Each is reversible; flag
is in square brackets for grep-ability.

## [D-001] osu! API v2 search via /beatmapsets/search?sort=plays_desc

Choice: use the authoritative v2 endpoint `/beatmapsets/search` with
`sort=plays_desc` and `s=ranked` to enumerate the top maps. Pagination via
`cursor_string`.

Alternatives: hard-code a known-good beatmap-ID list; scrape the public ranking
JSON. Rejected because they require offline curation.

## [D-002] "Hardest std diff per set" for enumeration

Each beatmapset can hold multiple difficulties. I pick the **highest
difficulty_rating** osu!standard diff per set. Rationale: top-leaderboard play
is overwhelmingly on the hardest diff; easy diffs dilute the data.

Flag: if we care about easier maps, swap to "all osu!std diffs per set" and
raise `target_count`.

## [D-003] Store raw v1 replay payload, not synthesize a full .osr

v1's `get_replay` returns ONLY the LZMA'd frames block (base64-encoded). We
could synthesize a fake .osr header (version, map hash, user, mods, counts,
score, combo, etc.) but that header is irrelevant for our feature extraction —
we have the metadata from v2 already. We parse LZMA directly.

Flag: if we ever want to feed an existing osrparse-based tool, we'll need to
build a .osr constructor.

## [D-004] Minimal .osu parser instead of `slider`

`slider` pulls numpy-twisted dependencies and has stronger mode coverage than we
need. We only need: Difficulty values, TimingPoints (for BPM/SV), HitObjects
with times and positions. A ~150-line parser covers that.

Flag: slider's slider-path tail-time computation is more accurate than ours
(ours uses a simple length/velocity formula per spec). For slider-tail outcome
modeling this becomes relevant.

## [D-005] Mod handling: features, not per-mod models

Encode HD/DT/HT/HR/EZ as features on the input, plus a per-row speed_mult that
rescales cvUR back to effective-1x. One model handles all mods.

Flag: if we see divergence by mod combo (e.g. HD trains cleaner than DT), we
can shard by mod.

## [D-006] Standard-only (no mania/CTB/taiko)

Per brief. Puller filters `mode=osu` when picking diffs.

## [D-007] Player embedding: learned jointly, dim 16, top-2048 users

User IDs beyond the top-2048 clickers map to slot 0 (shared "unknown player"
embedding).

Flag: pretraining per-user embeddings separately could reduce leakage of
difficulty-specific timing into the player representation.

## [D-008] Hit-window formula

Per osu! wiki: HW_300 = 80-6*OD, HW_100 = 140-8*OD, HW_50 = 200-10*OD, all in
ms (beatmap-time). HR → OD*1.4 (cap 10). EZ → OD*0.5. DT/HT → beatmap-time
hit windows unchanged (they're already specified in beatmap time); only *real
clock* windows tighten, which affects the player's experience but not our
offset calculations.

## [D-009] Replay frames are in beatmap-time

Critical and easy to get wrong: the replay-frame time deltas are in the game's
"music time", not wall-clock. DT 1.5x replays record at wall-clock 1x but with
a sped-up clock, so frames arrive with the right spacing already. See
`README.md` and `src/data/features.py` comments.

## [D-010] Negative-dt frame is a backward seek

osu!'s "ready" sequence ends with a frame carrying a negative dt that resets
absolute time toward 0. Earlier code clamped the delta at 0 and silently
misaligned all replays by seconds. Now we clamp the ABSOLUTE time at 0 but
accept negative deltas.

## [D-011] Miss detection: cursor outside radius at click OR offset outside HW_50

Mirrors osu!'s rule. We don't yet distinguish slider heads vs clicks, so a
slider head "miss" could in fact be a held-through from a previous slider.

Flag: refine with slider state tracking.

## [D-012] Puller target: 500 maps, top-50 replays each

At ~50 req/min and ~70% replay-success rate, this is ~8h wall-clock. Session
starts the puller in the background and works with however much data lands.
State is persisted to `data/raw/pull_state.json`, so a next session can
continue from the same `idx`.

## [D-013] Train/val split by beatmap_id

Held-out BEATMAPS, not held-out clicks. This is strict: if your model can
memorize a map's layout, it still has to generalize to unseen maps. Seed=0,
10% val.

## [D-014] Hold time: lognormal from per-user mean/std

Hold time is NOT currently a learned head; we sample from a lognormal with
per-user mean/std from the dataset. P4 outcome head takes the sampled click
time and the cursor; P5 key head is autoregressive LSTM.

Flag: a proper hold-time head would condition on BPM, prev/next gap, and key
type. Mild overfitting risk, low downside.

## [D-015] Cursor is reused from a human replay for outcome

In `generator.generate`, we reuse one human replay's cursor frames as the
cursor trajectory for outcome sampling. This is a research-only shortcut: it
means the outcome is realistic (cursor was actually on-circle sometimes, off
other times) but the generated replay isn't fully synthetic — a cursor model
is explicitly out of scope for this project per the brief.

Flag: if we ever need a fully synthetic replay, we'd need a cursor head too.
Small bezier-spline model per hit-object trajectory + noise is a reasonable
first pass.

## [D-016] baseline sampler uses alternating-K fraction, not full K-sequence

Per-user K1 fraction is learned; the sequence is iid sampled. Real K-sequence
has strong temporal structure (bursts of K1, bursts of K2, alternation in
streams). The timing head doesn't care; the detection check looks at
max-same-key-streak, so iid sampling makes the baseline streaks too short.

Flag: baseline could be upgraded to a 1st-order Markov on K1↔K2 to mimic
humans better.

## [D-017] Detection thresholds

Pulled from circleguard defaults and related tooling: cvUR < 60 or 50 is
flagged, edge_hit_rate ≥ 0.65 is flagged, hold_mean < 20ms is flagged,
miss_rate == 0 on >200 clicks is flagged, max_same_key_streak ≥ 20 is flagged.

Flag: these are conservative; circleguard has per-pattern cvUR curves that we
don't replicate here. Future: port circleguard's exact per-pattern test.

## [D-018] Phase A: effective_bpm + skill_percentile features

Added two features to break the DT-vs-nomod bias:

- `effective_bpm = bpm * speed_mult(mods)` — per-object, lets the model see that
  nomod 240 BPM and DT-at-160 BPM share the same timing envelope.
- `skill_percentile` — per-replay, derived from leaderboard rank cached at
  `data/raw/leaderboards/`. Default 0.95 when not cached (top-tier assumption).

Feature column counts:
- object features: 12 -> 14
- outcome head: 11 -> 12
- key head: 5 -> 6
- timing player_dim: 32 -> 64 (expanded capacity to use new signals)

Why: v2 model (one before Phase A) worked on HDDT 4-6★ (Circleguard-clean on
bmaps 954692 and 713442) but broke on nomod same-SR (UR 128 vs human 84 on
1851299) and on 7★+ tech. Rebalanced dataset + effective_bpm should close most
of the nomod gap; skill_percentile is a lead-in to Phase C (user-tunable
skill target).

Backup: `models/{timing,outcome,key}_best_v2_pre_a.pt`.

Dataset delta from pre-A to Phase A build:
- 2.6M -> 5.6M rows (2.2x)
- 5,128 -> 10,810 replays (2.1x)
- 159 -> 333 beatmaps (2.1x)
- mod distribution: 42% DT, 57% NM/HD/HR (previously estimated ~80% DT; actual
  distribution is more balanced thanks to puller progress overnight).
