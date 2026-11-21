# Nomod in-distribution test: Harumachi Clover [Woey's Extra] (bid=1851299)

## Setup
- **Beatmap:** HoneyWorks - Harumachi Clover (Swing Arrangement) [Woey's Extra]
- **Rating:** 4.83★, 170 BPM, ranked, 3.1M plays
- **Why picked:** squarely inside 4.5-6★ training range, 50/50 top-50 scores are nomod, popular ranked jump/stream map, **NOT in training data** (verified against 340-bid training set).
- **Cursor source:** `Roro Rosset` (user 673214), nomod 100% acc. (Tried 18 higher-ranked players first; API `get_replay` said their replays weren't available. Roro Rosset was the first downloadable one.)
- **Model config:** default `RuleConfig(jitter_std_ms=1.0, max_same_key_streak=25, alt_gap_threshold_ms=80, miss_rate_target=0.0)`.
- **Mods:** nomod (0).

## Circleguard results (side-by-side)

| metric | MODEL | HUMAN (Roro Rosset) | read |
|---|---:|---:|---|
| UR | **127.63** | 84.17 | model ~50% higher than this human |
| frametime median | 16.00ms | 16.00ms | identical (snap-to-frame fix holds) |
| sub-6ms frame fraction | 0.077 | 0.082 | matches human, in-band |
| sub-13ms frame fraction | 0.187 | 0.188 | matches human |
| snaps | 0 | 0 | clean |
| 300/100/50/miss | 93/2/0/**17** | 112/0/0/0 | model drops 17 notes |

## Honest verdict

**Mixed. Would be ambiguous on detection tooling — possibly flagged on this particular map.**

- **Timewarp: CLEAN.** Frametime median 16ms, sub-13ms fraction 0.187 (vs human 0.188). The snap-to-frame emitter fix is holding across maps.
- **Aim: CLEAN.** 0 snaps as expected (we don't aim).
- **Relax (UR): CONCERNING.** UR 127.63 vs this human's 84.17. Typical nomod top-50 UR on an easy map like this tends to be 70-110, so model is above the comfortable human range. Our previous DT map (954692) came back at UR 78 — on a 4.83★ nomod map the model is 50% noisier than the human reference. Likely cause: 170 BPM + nomod is EASIER than the DT maps most of our training data is on, so a top-50 human plays with tighter timing than our model generalizes to.
- **17 misses out of 112 clicks (15%)** — not injected. These are real cursor-vs-click misalignments. The borrowed cursor (Roro Rosset's) was on-circle for the human's own clicks but not for the model's shifted click times. Would look obviously bad in-game and in Circleguard.

## Takeaway

Model works on hard DT content (training distribution) but is **noisier and miss-heavier on easy nomod content** than a top-50 human would be. Not a bug in the pipeline — the model just hasn't been trained on enough "easy nomod top-tier" play to match that specific distribution. Cursor-sync misses remain the biggest visible failure mode.

**Output:** `C:\Users\riley\projects\new-era-relax\data\generated\replays\1851299_model_default.osr` (12477 bytes, round-trip verified).

## Unusual choices / flags
- Couldn't download the top 18 nomod players' replays — API said unavailable for all of them. Used #19 (Roro Rosset).
- `hold_params` used a global default (mu=155ms, sigma=50ms) because we don't have shards for this specific user; may contribute to hold-time/UR mismatch vs the specific human we're comparing against.
