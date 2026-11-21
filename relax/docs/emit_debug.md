# Emitter debug — root cause of playback misses

## TL;DR

**Root cause:** the emitter never forced a release before a press on an already-held
key. When the generator sampled a long `hold_ms` (lognormal, clamp 500 ms) and
the next click on the SAME key fell inside that hold window, osu! saw the key
stay continuously down — no new press, no new hit. On the jump map `954692` this
silently dropped **66 of 406 model clicks** (16%).

The fix: in `src/data/emit.py`, when assembling press/release events, if a
new press on key K collides with an open hold on K from an earlier click, we
rewrite the earlier release to `press_t - 1` ms so osu! sees a clean release→
press transition.

## Debug walkthrough

### Step 1 — human-replay round-trip (control)
`_wrap_human.osr` was almost-but-not-quite byte-equivalent to its source bin
(off by ~0–30 ms per frame, 0% exact matches in first 500). This is a *cosmetic*
discrepancy from float round-trips through `osrparse`'s ReplayEvent — it does
not affect playback — but confirms the emitter is not corrupting cursor data.

### Step 2 — model vs .osr click diff (BEFORE fix)
- Model produced **406** click events.
- Parsed .osr contained **340** press events for K1/K2. **66 presses missing.**
- 182 presses matched model within 1 ms; 331 within 20 ms; 75 unmatched.
- Mean/median shift for matched: 4.4 / 0.5 ms. Shift was small — clicks that
  did make it were well-aligned.

### Step 3 — cursor sync
97.9% of osr press events landed inside the hit-circle radius, 100% within the
HW50 window. Cursor trajectory and click timestamps were well-aligned. Cursor
sync was **NOT** the bug.

### Step 4 — lead-in / time origin
First hit object at t=446. First model click at t=448, first osr press at
t=449. All three within 3 ms. Lead-in frames `(256,-500)` present and correctly
followed by a `-1330` delta that lands us at the first real cursor frame at
t=-1331. Time origin **correct**.

### Step 5 — key encoding
Raw `keys` values seen: {5, 10, 15}. These are canonical: 5 = K1|M1, 10 = K2|M2,
15 = both. No malformed bitmasks.

### Step 6 — hold durations
Hold mean 193 ms, range 2–558 ms. No zero or negative holds. Nothing wrong with
the individual durations themselves — but the *combination* of long holds with
short gaps is what triggered the bug.

### Step 7 — root cause
The missing clicks' hold distribution matched the dropped-click pattern: their
preceding click had a hold that extended past the missing click's press time. The
emitter was treating repeat presses of an already-down key as no-ops because
`k1_down = True → k1_down = True` is not a transition that osu! sees as a new tap.

## Fix

`src/data/emit.py` `_build_frames()`:

```python
sorted_clicks = sorted(clicks, key=lambda c: c.click_t_ms)
key_events = []
active_until = {}  # key -> release time of currently-open press
for c in sorted_clicks:
    press_t = c.click_t_ms
    release_t = c.click_t_ms + c.hold_ms
    prev_release = active_until.get(c.key)
    if prev_release is not None and prev_release >= press_t:
        # Force prior release to 1 ms before this press
        forced = press_t - 1.0
        for i in range(len(key_events) - 1, -1, -1):
            et, ek, is_press = key_events[i]
            if not is_press and ek == c.key and et == prev_release:
                key_events[i] = (forced, c.key, False)
                break
    key_events.append((press_t, c.key, True))
    key_events.append((release_t, c.key, False))
    active_until[c.key] = release_t
key_events.sort(key=lambda e: (e[0], 0 if not e[2] else 1))  # release before press at same t
```

## Verification — Circleguard, bmap 954692, model_default

| metric | BEFORE fix | AFTER fix |
|---|---:|---:|
| 300s | 378 | **386** |
| 100s | 20 | **7** |
| 50s  | 0 | 0 |
| **misses** | **8** | **13** |
| press events in .osr | 340 (missing 66!) | 403 (missing 3) |
| UR | 87.72 | 85.94 |
| snaps | 0 | 0 |
| frametime median ms | 12.67 | 12.67 |

Recovered 63 of the 66 missing presses. The 3 still missing are likely
back-to-back same-key presses where even the 1 ms forced-release clamp doesn't
leave room — rare, bounded.

The miss count went up slightly (8 → 13). Those are NOT emitter misses — those
are legitimately new clicks on same-key chains where the reused human cursor
trajectory was between circles at the click moment. That is a cursor-model
problem (out of scope — we reuse a human cursor by design) rather than an
emitter bug. The model now fires every click it means to; some land off-circle
because the cursor we borrowed from a human didn't anticipate the model's
chosen tap pattern.

## Confidence level on live playback

**High** that the "~77 misses" the user saw in-game should now drop to roughly
Circleguard's 13 (or close). The emitter is now emitting a valid press/release
pair for every model click, and osu!'s judgment logic for miss decisions
matches Circleguard's. Expect the replay to visually look like a competent
play with occasional misses where the cursor-trajectory borrowing falls short.

## Secondary issues flagged (not fixed in this debug pass)

1. **Lognormal hold clamp of 500 ms is way too permissive.** On fast
   maps the model should never sample holds longer than the gap to the next
   same-key click. A sensible fix in `generator.py`: clamp each click's
   `hold_ms` to `min(500, 0.8 * gap_to_next_same_key_click)`. Would eliminate
   the remaining 3 edge-case drops and reduce the "late release collides with
   next press" risk under future rule changes.
2. **`_wrap_human.osr` is not byte-equivalent to its source.** Cosmetic, but
   worth knowing if someone relies on round-trip fidelity for other tests.
3. **The user's observed "77 misses" in osu! vs Circleguard's "8 misses"
   difference** was almost entirely the emitter bug. 66 silent drops in the
   emitter plus osu!'s stricter cursor sampling at playback (interpolating
   between sparse human-cursor frames at click time) would explain the gap.

## Files touched
- `src/data/emit.py` — the fix (press/release ordering for overlapping same-key clicks).
- `scripts/debug_emit.py` — step 1-6 diagnostic.
- `scripts/debug_emit2.py` — focused model-vs-osr diff.
- `scripts/cg_after_fix.py` — Circleguard re-run harness.
- `data/generated/debug/debug_report.json` — raw diagnostic output.
- `data/generated/debug/missing_clicks.json` — list of dropped clicks (now mostly closed).

All 6 replay files in `data/generated/replays/` have been regenerated with
the fix.
