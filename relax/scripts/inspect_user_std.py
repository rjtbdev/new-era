"""Per-user and per-replay offset variance from the parquet shards."""
import torch
import pandas as pd, glob
paths = sorted(glob.glob("data/processed/shard_*.parquet"))
df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
df = df.dropna(subset=["offset_ms"])

by_rep = df.groupby(["user_id","beatmap_id","mods"])["offset_ms"].agg(["std","count"]).reset_index()
print("Per-replay offset std:")
print(f"  mean={by_rep['std'].mean():.2f}  median={by_rep['std'].median():.2f}")
for q in [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]:
    print(f"  q{int(q*100):02d}  std = {by_rep['std'].quantile(q):.2f}")

by_user = df.groupby("user_id")["offset_ms"].std().reset_index()
by_user = by_user.rename(columns={"offset_ms":"ustd"})
print("\nPer-user offset std (collapsed):")
print(f"  mean={by_user['ustd'].mean():.2f}  median={by_user['ustd'].median():.2f}")

# Per-(user,beatmap,mods) offset-std lookup — the primary key the generator
# should use. Captures the player's actual performance on THIS map, which is
# what we want to imitate. Fall back to per-user when (user,map,mods) unknown.
per_rep = df.groupby(["user_id","beatmap_id","mods"])["offset_ms"].std().reset_index()
per_rep = per_rep.rename(columns={"offset_ms":"offset_std_ms"})
per_rep.to_parquet("data/processed/user_map_offset_std.parquet", index=False)
print(f"\nwrote data/processed/user_map_offset_std.parquet ({len(per_rep)} (user,map,mods) rows)")

# Per-user fallback (collapsed).
out = df.groupby("user_id")["offset_ms"].std().reset_index()
out = out.rename(columns={"offset_ms":"offset_std_ms"})
out.to_parquet("data/processed/user_offset_std.parquet", index=False)
print(f"wrote data/processed/user_offset_std.parquet ({len(out)} users)")

# Also compute a hold-time lookup
h = df.dropna(subset=["hold_ms"]).copy()
h = h[h["hold_ms"].between(20, 400)]  # drop obvious outliers
hold = h.groupby("user_id")["hold_ms"].agg(["mean","std"]).reset_index()
hold.columns = ["user_id","hold_mean_ms","hold_std_ms"]
hold.to_parquet("data/processed/user_hold.parquet", index=False)
print(f"wrote data/processed/user_hold.parquet ({len(hold)} users)")
print(f"  hold_mean  median = {hold['hold_mean_ms'].median():.1f}ms")
print(f"  hold_std   median = {hold['hold_std_ms'].median():.1f}ms")
