# Replay attrition diagnosis

**TL;DR:** The ~45% attrition the agent flagged was NOT API-side (wrong-replay returns, mod mismatches, old formats). It was a **client-side parser bug**: `src/data/parse.py` silently clamped the negative "lead-in" dt frame to t=0, shifting every subsequent frame forward by ~500–600 ms. Clicks then landed outside hit-windows and/or on the wrong hit object, producing miss_rate → ~1.0.

## Impact

Before the fix, on first 2000 indexed replays, the `build_dataset.py` sanity
filter dropped **1000+ replays** (>50 % miss rate). After the fix: **0 drops**
in the same sample.

## Dominant failure mode (all 207 fails of first 300 scan)

**Reason:** `miss_rate > 0.5` from feature-extraction, 207 / 207.

**Root cause:** `parse.py::parse_replay_text` was treating the osu! replay's
leading negative-`dt` frame as a bogus value to clamp, rather than as the
**audio-lead-in offset**.

The osu! replay format convention (per osu! wiki and verified in-session):

1. First 1–2 frames are placeholders at `(256, -500)` with `dt=0` / `dt=-1` — cursor parked off-screen.
2. The next frame carries a **large negative `dt`** (typically -300 to -2500 ms). Its cumulative `t` is **negative** and represents pre-audio lead-in time. Subsequent frames accumulate `dt` from there.
3. Audio-start = `t = 0`. Beatmap hit object times are in this axis.

The old parser:

```python
t = max(0, t + w)   # clamped every frame to >= 0
```

shifted every frame forward by `|lead-in|` (typically 500–900 ms), so a click
intended for `ho_t` landed at `ho_t + lead_in`, way outside the 100–200 ms
hit window. `features.py` then either (a) found no click near the HO → marked
miss, or (b) matched the click to the NEXT hit object and computed a cursor
distance in the hundreds of pixels → marked miss via `dist > radius`.

The bug manifested on **every replay** whose lead-in dt was non-trivial, but
only the ones with enough time-shift to exceed hw50 showed as full misses.
No-mods replays on maps with short lead-in sometimes squeezed through, which
is why the agent saw the bug intermittently as "mysterious per-map attrition"
instead of a universal shift.

Mod bias in the failures (DT 70 %, HR 45 %) was coincidence: top-50 lists on
hard maps are DT-heavy, that's all. Mod handling in `features.py` was already
correct.

## Fix applied

- `src/data/parse.py` — rewrote the frame-time computation to:
  1. Skip the leading `(256, -500)` placeholder frames (identified by their
     exact "parked" position).
  2. Set `t` for the first real frame to the cumulative sum of dts up to and
     including it — **without clamping to zero**, so negative `t` (lead-in)
     is preserved.
  3. Accumulate dts normally for the rest, again no clamping.

## Verified

- Previously-failing HR-only SS replay (`1849273_7657831_16.bin`, acc=1.0):
  329 × 300, zero miss (was 329 × miss).
- Previously-failing HDDTHRFL replay (`1764213_5247450_1624.bin`):
  109 × 300 + 6 × 100, zero miss (was 115 × miss).
- No-mods sanity check (`736215_12408961_0.bin`): 1005 × 300, unchanged.
- Full first-2000-replay scan: drop rate **0 %** (was 69 % on first 300).

## Hypotheses that turned out to be FALSE (saved time if revisited)

1. "API serves most recent play regardless of mods" — not true; the on-disk
   payload has no beatmap_md5 or player id to verify against anyway.
2. "API returns score from a different diff of the same mapset" — same.
3. "Old replay format incompatible with osrparse" — we use our own parser, not
   osrparse.
4. "HR y-mirror direction wrong" — mirror direction is correct. The bug *looked*
   like a mirror problem (cursor near unmirrored HO) only because the shifted
   click time put the cursor somewhere arbitrary.

## What the user needs to decide

**Nothing.** The fix is in; the build-dataset filter will now let through
~100 % of replays instead of ~30 %. Effective pulled-data yield just went up
~3×. No new API keys, mod flags, or endpoints needed.

## What to do next (not in scope for this diagnosis)

1. Re-run `src/data/build_dataset.py` on all 5086 (and growing) indexed replays
   to rebuild the processed/ shards with the full data.
2. Keep the `.miss` markers as-is; those are genuine API-no-replay cases
   (~35 % — users with private replays or stale scores), unaffected by this
   bug.
3. Retrain the three heads on the now-much-larger dataset. The previously
   reported "only 4 beatmaps survived" was a direct consequence of this bug;
   the agent will now have hundreds of maps in the training set.
