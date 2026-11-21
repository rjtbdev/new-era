# Post-process rule layer

Trained models are frozen. Rules in `src/models/rules.py` run after the three
heads produce click events and modify (click_time, key, outcome) before the
generator returns. Tune these without retraining.

Default `RuleConfig()` is close to neutral. The one non-neutral default is
`max_same_key_streak=25` — the LSTM over-commits to one key (eval observed
streaks of 49-52), so the cap is an intervention baked in.

## Rules

### 1. Alternation threshold — `alt_gap_threshold_ms` (default 80.0)

At high BPM humans can't single-tap. When the physical gap between consecutive
hit objects falls below this threshold, force alternation regardless of what
the key head chose. 80ms is roughly 1/4 at 187 BPM.

- Set higher (e.g. 100) if a player single-taps up to higher BPM.
- Set lower (e.g. 60) if alternation feels forced too early.

### 2. Preferred key bias — `preferred_key`, `bias_logit`

`preferred_key` ∈ {"K1", "K2", None}. When `None`, model decides. When set, in
the non-forced-alternation zone, flip to the preferred key with probability
`sigmoid(bias_logit)`. Default `bias_logit=1.0` → 73% flip-to-preferred when
the model picked the other key. Set `bias_logit` higher for a stronger bias.

### 3. Miss injection — `miss_rate_target`, `miss_mode`

Default 0.0 (no misses — the model plays cleanly). Flip the switch if you want
human-like miss patterns in the generated replay.

- `miss_mode="random"`: roll `miss_rate_target` chance per click.
- `miss_mode="pressure"`: concentrate misses where pressure is high — streams
  (`pressure_stream_gap_ms`, `pressure_stream_min_len`), endgame
  (`pressure_endgame_frac`), or after long same-key streaks
  (`pressure_streak_threshold`).

Misses are implemented by shifting the click +`miss_shift_ms` late (default
100ms, outside all hit windows on osu!standard) and marking the outcome as
"miss".

### 4. Same-key streak limiter — `max_same_key_streak` (default 25)

After this many consecutive presses of one key, force a switch on the next
click. Caps the streak detection flag without retraining the key head.

### 5. Timing jitter — `jitter_std_ms` (default 0.0)

Adds `N(0, jitter_std_ms)` to each click's offset. Model's residual std is
~9.7ms already, which is human-tier. Set >0 only if a specific map's cvUR
looks suspiciously tight.

## Using the rules

### Programmatic

```python
from src.models.rules import RuleConfig
from src.models.generator import generate

cfg = RuleConfig(
    alt_gap_threshold_ms=80.0,
    preferred_key="K1",
    max_same_key_streak=25,
    miss_rate_target=0.0,
)
gen_df = generate(bm, mods=0, ..., rules=cfg)
```

### Eval CLI

```
python scripts/run_eval.py --rules "alt_gap_threshold_ms=80,preferred_key=K1,max_same_key_streak=25" --out-suffix _with_rules
```

Unknown rule keys raise `ValueError` so typos surface instead of being silently
ignored.
