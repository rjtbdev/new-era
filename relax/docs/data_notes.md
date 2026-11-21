# Data notes

## Sources

- **osu! API v2** (`src/api/v2.py`) — OAuth client_credentials, scope=public.
  Endpoints used:
  - `GET /beatmapsets/search?s=ranked&sort=plays_desc&m=0` — enumerate top-ranked
    osu!standard beatmapsets by playcount, paginated with `cursor_string`.
  - `GET /beatmaps/{bid}` — per-beatmap metadata (OD/AR/CS/HP, star-rating,
    etc.).
  - `GET /beatmaps/{bid}/scores?mode=osu&limit=50&type=global` — global top-50
    leaderboard. Returns each score with `user.id`, `mods` as a string list.

- **osu! API v1** (`src/api/v1.py`) — legacy key, rate-limited 50 req / 60s.
  - `GET /api/get_replay?k=KEY&b={bid}&u={uid}&m=0&mods={BITS}&type=id` — returns
    JSON `{"content": "<base64-lzma-frames>"}` on success; `{"error": "..."}` or
    404/403 on failure. Mods must match the exact bit set that score was played
    with.

- **ppy.sh static** — `https://osu.ppy.sh/osu/{bid}` returns the raw .osu file.
  No auth, no rate limit beyond their general fair-use.

## Mod-bit conversion

v2 scores return `mods` as a string list, e.g. `["HD","DT"]`. v1 `get_replay`
needs the bit sum (`8 | 64 = 72`). `src/data/pull.MOD_BITS` has the table. NC
implies DT (64|512 = 576).

## Cached artifacts

- `data/raw/beatmapsets_search.jsonl` — every search-page response, one JSON
  object per line. Lets us resume enumeration without re-fetching.
- `data/raw/beatmaps/{bid}.json` — v2 beatmap metadata
- `data/raw/beatmaps/{bid}.osu` — raw .osu file
- `data/raw/leaderboards/{bid}.json` — v2 top-50 scores for that beatmap
- `data/raw/replays/{bid}_{uid}_{mods}.bin` — LZMA'd replay-frames block
- `data/raw/replays/{bid}_{uid}_{mods}.miss` — zero-byte marker for "we tried
  and the replay was unavailable". Prevents retries on every re-run.
- `data/raw/replay_index.jsonl` — append-only successful-pull index.
- `data/raw/pull_state.json` — persistent cursor `{bids, idx, last_stats}`.

## Replay-frames gotchas (lessons learned)

1. **Frame times are in beatmap-time.** DT/HT doesn't change the frame timeline.
   A click at beatmap_t=2397 lines up directly with HO.t=2397. When comparing
   offsets across mods, divide by `speed_mult` to express in real-time-equivalent
   ms ("converted" UR).

2. **Negative dt on the third frame.** osu! records a "wait, then seek back to
   near zero" sequence to represent the player loading in. Example:
   `0|256|-500|0, 9146|256|-500|0, -9124|216.9|131.1|1, ...`. If you clamp
   negative dt at 0 (we did initially), absolute time runs 9146ms ahead for the
   rest of the replay, and every alignment is wrong by that offset.

3. **HR mirrors Y, not X.** `y' = 384 - y` on the hit-object side (and the
   cursor data is recorded post-mirror, so it matches). We mirror object
   positions when computing cursor_dx/dy.

4. **Key bits.** `M1=1, M2=2, K1=4 (sets M1 implicitly to 5), K2=8 (sets M2
   implicitly to 10), Smoke=16`. Keyboard-and-mouse-only players exist; pure
   M1/M2 play shows up with no K1/K2 bits ever. For training key-choice we
   filter to K1/K2 only.

5. **RNG seed row.** The LAST row of the frames block is literally
   `-12345|0|0|<seed>`. If you don't special-case it, you'll try to int() the
   seed as a key bitmask.

## Dataset layout

Per-click rows written to `data/processed/shard_{unix}.parquet`. Columns:

```
beatmap_id, user_id, mods, ho_idx, ho_t, ho_kind, ho_x, ho_y,
ho_new_combo, prev_gap_ms, next_gap_ms, bpm, circle_radius_px,
hw300, hw100, hw50, speed_mult,
click_t, click_x, click_y, hold_ms, key_type, offset_ms,
cursor_dx, cursor_dy, cursor_dist, edge_bias, outcome
```

Rows with `outcome=="miss"` have `click_t/click_x/.../offset_ms/hold_ms =
None` since no key-down was matched.

## Scale

- Pulled (as of last writeup): ~300 replays across 10 beatmaps.
- Target: 500 maps × ~70% success × top-50 = ~17.5k replays
- At 50 API calls/min, this takes ~6-8 hours wall-clock. Puller is resumable.
