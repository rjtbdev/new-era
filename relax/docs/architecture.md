# Architecture

Three heads, decoupled by design.

## Timing head (`src/models/timing.py`)

```
input  :  (B, 17, 12)   -- 17-object window, 12-dim per-object features
          player_id (B,)
          pad_mask (B, 17)
hidden :  d_model = 128, 4 transformer layers, 4 heads
output :  4-component mixture of Gaussians over click_offset_ms (beatmap time)
loss   :  NLL of observed offset under the mixture
```

**Features per object (relative to the row we're predicting for):**

1. dt (seconds since the target object)
2. sign(dt) * log1p(|dt|)
3. x / 512 (playfield x, normalized)
4. y / 384
5. is_circle
6. is_slider
7. is_spinner
8. new_combo
9. bpm / 300
10. prev_gap_ms / 1000
11. next_gap_ms / 1000
12. hw300 / 50

**What it sees:** the music. Not the cursor.

**Why this matters:** every classical detection failure of "relax" tracers back to
"the click fires when the cursor crosses into the circle." If that's true, you
*always* fire when cursor is on-circle, so edge-hit rate collapses toward 0 and
your misses are perfectly correlated with cursor-leaves-circle events. If
instead you fire on music, your edge-hit rate inherits the cursor's own
geometry and your misses inherit the cursor's own slip-ups.

## Outcome head (`src/models/outcome.py`)

```
input  :  11-dim vector (offset_ms, cursor_dx, cursor_dy, hit windows,
          radius, bpm, is_circle, prev_gap, next_gap)
hidden :  2-layer MLP, 128 hidden
output :  softmax over {300, 100, 50, miss}
loss   :  cross-entropy
```

Mostly deterministic given (offset, cursor_distance). The network earns its
place by learning the stochastic miss-rate under difficulty pressure: tight
BPM, short gaps, small circles.

## Key choice head (`src/models/key_choice.py`)

```
input  :  (B, 8, 5)  -- last 8 clicks: (prev_key_onehot(2), bpm/300, prev_gap, new_combo)
hidden :  LSTM, 64 hidden
output :  softmax over {K1, K2}
loss   :  cross-entropy
```

Captures the K1-burst / K2-burst / alternation structure that humans show.
Pure Bernoulli would fail the "max_same_key_streak" detection check.

## Generator (`src/models/generator.py`)

Wires them together on a beatmap:

1. Parse the .osu to a sequence of hit objects.
2. For each object, build the 17-object context, run timing → sample offset.
3. Look up cursor_x/y at (ho_t + offset) from a reference human cursor
   trajectory. (A cursor model is explicitly out of scope — see
   docs/decisions.md D-015.)
4. Run outcome → sample {300,100,50,miss}.
5. Run key head autoregressively → sample K1/K2.
6. Sample hold_ms from a per-user lognormal.

Output: a DataFrame with the same shape as `src/data/features.py` produces for
humans. The eval harness treats both the same way.
