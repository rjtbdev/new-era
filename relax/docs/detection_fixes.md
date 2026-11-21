# Detection fixes: timewarp + cvUR

## Context
User's detection tool flagged `954692_model_default.osr` on two signals:
- **Timewarp** (frametime median 12.7ms, tool floor 16.7ms, flag <13ms)
- **Relax** (cvUR 85.94 suspicious; human replay on same map comes back CLEAN)

## Fix 1: frametime density (timewarp)
Changed `src/data/emit.py::_build_frames` to SNAP press/release events onto
the nearest existing cursor frame (window ±8ms = half a native osu! frame)
instead of inserting new frames at every click timestamp.

Old behavior packed sub-6ms deltas around every click; new behavior preserves
the cursor stream's native ~16.7ms cadence.

### Verification (954692_model_default, pre-fix vs post-fix)
| metric | pre-fix | post-fix (jitter 0) |
| --- | ---: | ---: |
| frame count | ~same | 5044 |
| frametime median (raw file) | ~12.7ms | **23ms** (p25=11, p75=25) |
| frac frames <6ms | ~26.7% | **10.3%** |
| frac frames <13ms | high | 27.0% |
| Circleguard frametime median | 12.67 | **15.33** |
| Circleguard snaps | 0 | 0 |
| 300/100/50/miss | 378/20/0/8 | 378/20/0/8 (unchanged) |

Circleguard's frametime median went from 12.67 → **15.33ms**. Still below the
tool's claimed 16.7ms floor but above the 13ms flag threshold. **Timewarp
flag should clear.** The 15.33 is already in the human-comparison band
(human was 14.67).

## Fix 2: cvUR tune (jitter sweep)
Ran three jitter settings (0.0, 1.0, 2.0 ms). Circleguard UR per file:

| jitter_std_ms | UR | 300/100/50/miss |
| ---: | ---: | --- |
| 0.0 | 78.43 | 378/20/0/8 |
| **1.0** | **78.39** | 378/20/0/8 |
| 2.0 | 79.45 | 378/20/0/8 |

All three sit near 78-79 UR — close to human (72.82) and comfortably inside
human band for HDDT. Note: the pre-fix UR of 85.94 was measured on the OLD
.osr format with the frame-insertion bug; snap-to-nearest reduced UR as a
side-effect (fewer artificial cursor frames jittering Circleguard's UR math).

**Picked: jitter 1.0.** Lowest UR of the three, and 1ms gaussian jitter is a
sensible default even outside this specific detection concern. Written into
`954692_model_default.osr` (the file the user tests against).

## Final per-file Circleguard snapshot
`954692_model_default.osr` (post both fixes):
- 300: 378
- 100: 20
- 50: 0
- miss: 8
- UR: 78.39
- frametime median (cg): 15.33ms
- snaps: 0

## Confidence
- **Timewarp flag — moderate-to-high confidence CLEAN.** Circleguard's
  frametime median is 15.33ms, well above the 13ms flag floor, and the sub-6ms
  fraction dropped from ~26% to ~10% (within range of the human comparison's
  ~14%). If the tool uses the same metric Circleguard does, this clears. If
  the tool measures something subtly different (e.g. fraction below a cutoff),
  it may still flag — but the underlying cause (packed sub-frame intervals) is
  fixed.
- **Relax flag — moderate confidence CLEAN.** UR 78.39 is close to the
  human's 72.82 and well inside typical HDDT human range (60-110). If the
  tool's "cvUR" is just coefficient-of-variation of UR, we haven't directly
  measured it, but the underlying UR is now close to human. Caveat: the tool
  flagged at cvUR 85.94 last time with threshold unknown; possible but
  unlikely it still fires at ~78.

## Files changed
- `src/data/emit.py` — snap key events onto nearest cursor frame instead of
  inserting new frames.
- `scripts/regen_and_test.py` — regenerate + CG measurement harness (new).
- `data/generated/replays/954692_model_default.osr` — overwritten with
  jitter=1.0 best-tuned file.
- `data/generated/replays/954692_model_default_jitter{0p0,1p0,2p0}.osr` —
  the three test variants kept for reference.
- `data/regen_results.json` — raw per-jitter measurements.

## Not addressed
- Only tested on 954692. Other maps (1059894, 2449781) were NOT regenerated.
  If the user wants them regenerated with fix 1 + jitter 1.0, re-run
  `scripts/generate_replays.py` after this emitter change.
- `_model_human` variants still have the pressure-miss heuristic bug (77
  misses on jump maps). Not in scope.
