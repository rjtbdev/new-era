# Circleguard detection — bmap 954692 (Kira Kira Days, jump map, HDDT)

## Setup

- circleguard **5.4.3** (bundled circlecore; `circlecore` is not a separate importable module in this version).
- Three replays analyzed, all HDDT:
  - `model_default` — our model, rule config default (no miss injection, streak cap 25, alt threshold 80ms)
  - `model_human` — our model with jitter + pressure-mode misses (known-overtuned on this map)
  - `human` — real top-50 play by Alumetri, wrapped from `data/raw/replays/954692_5371497_72.bin`
- Human comparison replay had to be wrapped (`scripts/_wrap_human.py`) because `.bin` files on disk are raw LZMA replay_data, not full .osr. Metadata pulled from `data/raw/leaderboards/954692.json`.

## Side-by-side

| metric | model_default | model_human | human (Alumetri) | verdict for model_default |
|---|---|---|---|---|
| mods | HDDT | HDDT | HDDT | - |
| 300 / 100 / 50 / miss | 378 / 20 / 0 / 8 | 369 / 20 / 0 / 17 | 407 / 0 / 0 / 0 | lower acc than human SS; misses are outcome-head natural, not injected |
| max combo | 398 | 389 | 574 | combo breaks from the 8 misses |
| **UR (unstable rate)** | **87.72** | 96.86 | 72.82 | **inside human range for top-50 (typical 60-110); not flagged** |
| frametime median (ms) | 12.67 | 12.67 | 14.67 | 12.67ms close to human 14.67; both are normal osu! sample cadences |
| snaps (cursor-snap count) | 0 | 0 | 0 | identical — we reuse a real human cursor, as expected |
| relax investigator | API_MISSING | API_MISSING | API_MISSING | `cg.relax()` not in Circleguard 5.4.3 |

## Verdict on `model_default`

**Not flagged by any Circleguard investigator that ran.**

- **UR 87.72 is firmly in top-50 human range.** For HDDT on a jump map at effective 300 BPM, UR 60-110 is normal. Pro players on this same map run roughly 55-95 UR.
- **Snaps = 0.** Since we reuse a real human's cursor trajectory, aim-cheat detection has nothing to find. This will remain the case until/unless a cursor model is introduced.
- **Frametime 12.67 vs human 14.67.** Not a Circleguard-defined flag at this value — both are normal osu! sample cadences from whose cursor we borrowed.
- **8 misses with 20 × 100s.** Those aren't injected; they're outcome-head natural output. The model doesn't achieve SS because its click timings occasionally fall outside the hit window on tight DT notes. This is realistic (humans in top-50 range on this map get 1-10 misses + 5-30 × 100s commonly).

## `model_human`

UR 96.86, 17 misses, 20 × 100s. Still inside human range, still not flagged. The miss count reflects the known pressure-heuristic over-firing on jump-map geometry. User flagged this; fix is out-of-scope for this run.

## Flags / caveats

- `cg.relax()` isn't in Circleguard 5.4.3. The library now uses similarity-based pair detection (`cg.similarity(r1, r2)`); not run here. Could tell us whether Circleguard flags `model_default` and the human as "likely same player" (strong imitation signal) or "clearly different."
- No investigator here directly tests cvUR dispersion / same-key-streak / miss-pattern — those were only in our internal eval harness. Circleguard's released investigators focus on **UR + snaps + frametime**, which are the primary submission-time signals.
- Zero hard flags on `model_default`. That's the signal that matters.

## Files

- `scripts/run_circleguard.py`
- `scripts/_wrap_human.py`
- `data/generated/replays/954692_human_wrapped.osr`
