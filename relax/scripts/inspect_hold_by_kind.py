"""Hold-time distribution split by circle vs slider."""
import torch
import pandas as pd, glob, numpy as np
paths = sorted(glob.glob("data/processed/shard_*.parquet"))
df = pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
df = df.dropna(subset=["hold_ms","ho_kind"])
df = df[df["hold_ms"].between(10, 1000)]

for kind in ("circle", "slider"):
    sub = df[df["ho_kind"] == kind]
    vals = sub["hold_ms"].to_numpy()
    print(f"\n[{kind}]  n={len(vals)}")
    print(f"  mean={np.mean(vals):.1f}  median={np.median(vals):.1f}  std={np.std(vals):.1f}")
    print(f"  q10={np.percentile(vals,10):.1f}  q25={np.percentile(vals,25):.1f}  q75={np.percentile(vals,75):.1f}  q90={np.percentile(vals,90):.1f}")

# write per-kind lookup
out = {}
for kind in ("circle","slider"):
    sub = df[df["ho_kind"] == kind]
    out[kind] = {"mean": float(np.mean(sub["hold_ms"])), "std": float(np.std(sub["hold_ms"]))}
import json
with open("data/processed/hold_by_kind.json","w") as f:
    json.dump(out, f, indent=2)
print(f"\nwrote data/processed/hold_by_kind.json")
