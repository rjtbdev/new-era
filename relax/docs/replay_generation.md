# Replay generation for detection testing

Generated `.osr` replay files from the trained model for a held-out set of
beatmaps. Output lives at `data/generated/replays/`.

## Beatmaps picked

Selected from the held-out validation set (10% of training beatmaps, not seen
during training). Chosen to span the difficulty spectrum:

| bid | title | diff | stars | BPM | HOs | OD | why |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 954692 | Kira Kira Days | Shiawase!! | 5.60 | 200 | 407 | 9.0 | hard, dense stream map |
| 1059894 | Harebare Fanfare | Bright & Cheerful | 5.04 | 168 | 347 | 8.0 | middle difficulty, varied patterns |
| 2449781 | Angel With A Shotgun (Sped Up) | A r M i N's Extra | 4.59 | 142 | 254 | 8.7 | easier, lower density |

No no-mod replays were available in our dataset — top-50 leaderboards for these
maps are all HDDT / HDHR / HDHRDT. The generated files inherit the mods from
the reference human whose cursor we reused.

## Generation configs

Two configs per map:

**`<bid>_model_default.osr`** — neutral rule layer:
- `alt_gap_threshold_ms=80` (force alternation under ~190 BPM 1/4)
- `max_same_key_streak=25` (cap LSTM over-commits)
- `miss_rate_target=0.0` (no injected misses; model outputs natural misses when cursor is off-circle)
- `jitter_std_ms=0.0`
- `preferred_key=None`

**`<bid>_model_human.osr`** — human-flavor:
- Same as default, plus:
- `jitter_std_ms=1.5` (adds small gaussian noise on click times)
- `miss_rate_target=0.015, miss_mode="pressure"` (injects ~1.5% misses during
  streams, end-game, or long same-key streaks — the places humans actually choke)

## Output files

```
data/generated/replays/954692_model_default.osr
data/generated/replays/954692_model_human.osr
data/generated/replays/1059894_model_default.osr
data/generated/replays/1059894_model_human.osr
data/generated/replays/2449781_model_default.osr
data/generated/replays/2449781_model_human.osr
```

A machine-readable report with per-file metadata is at
`data/generated/generation_report.json`.

## Round-trip verification

Every file was read back with `osrparse.Replay.from_path(...)` after writing.
All six replays passed:

| bid | tag | md5 ok | counts ok | mods ok | n_frames | t range (ms) | covers map |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 954692 | default | ✓ | ✓ | HDDT (72) | 5620 | [-1331, 91114] | ✓ (ends 89621) |
| 954692 | human | ✓ | ✓ | HDDT (72) | 5620 | [-1331, 91114] | ✓ |
| 1059894 | default | ✓ | ✓ | HDDT (72) | 5145 | [-1298, 88615] | ✓ (ends 87637) |
| 1059894 | human | ✓ | ✓ | HDDT (72) | 5128 | [-1298, 88615] | ✓ |
| 2449781 | default | ✓ | ✓ | HDHRDTNC (600) | 4423 | [-2001, 89620] | ✓ (ends 87345) |
| 2449781 | human | ✓ | ✓ | HDHRDTNC (600) | 4414 | [-2001, 89620] | ✓ |

The t_min field is negative — this preserves the audio lead-in period (osu!
starts cursor rendering before audio t=0, which is required for detection tools
that look at pre-audio frames).

## .osr format gotchas we hit

- **`ReplayEventOsu.keys` is typed `osrparse.utils.Key` (an `IntFlag`).**
  Passing a plain `int` crashes `Replay.write_file` at pack time
  (`'int' object has no attribute 'value'`). Wrap: `keys=Key(int(v))`.
- **`Key()` accepts arbitrary ints, not just the defined bits.** Needed for
  the RNG seed trailer, which puts the seed value in the `keys` field alongside
  `time_delta == -12345` as a sentinel.
- **Lead-in frames must be synthesized.** osu! replays begin with two seed
  frames at `(256, -500)` with `dt` 0 and -1, then the true lead-in `dt` places
  the stream at `t = lead_in_offset`. Our emitter prepends these automatically.
  Skipping them causes some detection tools to reject the file.
- **K1 press = keys 5 (M1|K1); K2 press = keys 10 (M2|K2).** The dual-bit rule
  matches what real osu!stable writes. Writing bare K1=4 or K2=8 confuses tools
  that distinguish keyboard-vs-mouse presses.
- **Reused cursor keeps its original mods bits.** The generated replay inherits
  the mod combo that the cursor-source human was playing. Combining a HDDT
  cursor trajectory with a NoMod click stream would misalign timing. Documented
  per-file in the report.

## Running Circleguard

Install (CPython 3.10-3.12 recommended; same venv we use):

```
cd C:\Users\riley\projects\new-era-relax
.venv\Scripts\python.exe -m pip install circleguard
```

Minimal invocation — check cvUR + frame time + snaps + relax on one of our files:

```
.venv\Scripts\python.exe -c "
from circleguard import Circleguard, ReplayPath
cg = Circleguard('<osu-api-v1-key>')
r = ReplayPath('data/generated/replays/954692_model_default.osr')
cg.load(r)
print('ur:', cg.ur(r))
print('frametime:', cg.frametime(r))
print('snaps:', cg.snaps(r))
print('relax:', cg.relax(r))
"
```

(Legacy osu! API v1 key is already in `.env` as `OSU_API_V1_KEY`. Pass it as
the Circleguard constructor arg. The `relax(r)` call returns a confidence
score; `snaps(r)` returns suspected aim snaps; `frametime(r)` returns
sub-16ms frame anomalies; `ur(r)` returns the unstable rate on that replay.)

To batch all six:

```
.venv\Scripts\python.exe -c "
import os, pathlib
from circleguard import Circleguard, ReplayPath
cg = Circleguard(os.environ['OSU_API_V1_KEY'])
for fn in sorted(pathlib.Path('data/generated/replays').glob('*.osr')):
    r = ReplayPath(str(fn))
    cg.load(r)
    print(fn.name, 'ur=', cg.ur(r), 'ft=', cg.frametime(r), 'snaps=', len(cg.snaps(r)))
"
```

Circleguard needs the beatmap file to compare against, which it fetches from
osu! via the API. If the API call fails for a map (deleted, restricted), it
cannot compute UR — the error message will say so. All three beatmaps picked
here are ranked and should fetch fine.

## Things to flag for the user

- **Reference cursor leakage.** We reused one human's cursor per map (the one
  with the most clicks in our dataset for that map). The cursor was captured
  under HDDT for all three maps. That means:
  - The cursor's **mods field** matches (the emitter inherits it).
  - The cursor's **timing** reflects where that human aimed — our model's
    click timing is layered on top. If the human's cursor was at pixel (256,100)
    at song-time 32040 ms, and the model chose to click at 32045 ms, we emit a
    click at the cursor's interpolated position at 32045 ms. That's realistic.
  - Detection tools might notice that the cursor path is *identical across
    multiple replay submissions* if we generated many model replays off the
    same cursor source. We only did one default + one human-flavor variant
    per map, so cursor is reused twice (not a submission-tracking risk
    because we don't submit).
- **Score field is placeholder.** The `score`, `max_combo`, and life bar
  fields are approximations. Circleguard ignores these; replay-match tooling
  on bancho would not.
- **No submission.** Per constraints these files stay local. Do not upload them.
