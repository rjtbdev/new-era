# Yomi Yori test — model on untrained-level difficulty

## Beatmap

- **Imperial Circus Dead Decadence - Yomi yori Kikoyu, Koukoku no Tou to Honoo no Shoujo.** [Reverberation] (bid=1218337) by PoNo
- Only diff in set 575330
- **SR 7.44★**, BPM 220, 8:10 long, 2,345 hit objects
- 10.7M plays, ranked

This is well above the 4-6★ range the model was trained on. Explicit extrapolation test.

## Mods

Request was nomod. **No nomod exists on the global leaderboard** (only 2 scores on it,
both HD-based). Fell back to **HD+HR (mods=24)**, cursor source `yodeston` (uid=15556397).

## Model generation

- Rules: `jitter=1.0ms`, `alt_gap_threshold=80ms`, `max_streak=25`, no miss injection
- Output: 2,335 clicks (out of 2,345 objects; covered 99.6%)
- Internal judgment counts from generator: 2308/24/0/3 (300/100/50/miss)

## Circleguard results (side-by-side)

| metric | model | human (yodeston) |
|---|---:|---:|
| UR | **239.75** | 77.80 |
| frametime median | 16.0ms | 16.0ms |
| snaps | 0 | 0 |
| judgments (Circleguard re-judged) | 1401 / 23 / 181 / **730** | 2313 / 22 / 0 / 0 |

## Honest verdict

**Model would be heavily flagged on this map.** Two problems:

1. **UR 240 vs human 78.** Model timing fell apart — triple the human UR. On a 7.44★ 220 BPM map, the model's clicks are landing much later/earlier than the hit windows accommodate. Public detection would see this as either a very low-tier player OR, more suspiciously, a bot that doesn't understand the map's timing.
2. **730 misses out of 2335 clicks (31% miss rate).** Circleguard re-judges the replay against the beatmap and finds the cursor wasn't in the circle at click time for 730 notes. This is the cursor-sync / timing-mismatch issue from before but *much worse* because the map is harder.

**Root cause (most likely):** the model extrapolates badly outside its 4-6★ training range. Reverberation is a long, dense, tech-heavy map at 220 BPM — nothing like the pop maps in the training set. The timing head doesn't know what to do with it.

**Clean signals (unchanged):**
- Snaps: 0 — no aim-cheat signature (cursor is human's)
- Frametime median: 16.0ms — matches human exactly (the snap-to-frame fix holds up)

## What this tells us

- The model does NOT generalize to maps far outside the training distribution.
- On familiar-style maps (like 954692, jump map ~5.6★) it produces clean replays.
- For a user who wants to play maps in the 4-6★ range, the model is ready.
- For 7★+ or very unusual maps, model needs either (a) training data from that range, or (b) a different approach.

## File paths

- Model replay: `data/generated/replays/1218337_model_default.osr`
- Human wrapped: `data/generated/replays/1218337_human_wrapped.osr`
- Full Circleguard JSON: `data/yomi_cg_results.json`
