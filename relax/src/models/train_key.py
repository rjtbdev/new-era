"""Train key-choice head.

For each click, build a context of the previous `ctx` clicks (key_type, bpm,
prev_gap, new_combo) → predict next key_type (K1 vs K2).

Rows where key_type is M1/M2-only or NaN are skipped (M1/M2 only players exist
but we train the model on the dominant K1/K2 convention).
"""

from __future__ import annotations

import pathlib
import sys
import time
from dataclasses import asdict

import torch  # must precede numpy/pandas on Windows (DLL init order)
from torch.utils.data import DataLoader, Dataset

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.models.datasets import load_all_shards  # noqa: E402
from src.models.key_choice import KeyConfig, KeyModel  # noqa: E402


PROJECT = pathlib.Path(__file__).resolve().parents[2]
CKPT_DIR = PROJECT / "models"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


class KeyDataset(Dataset):
    """Precomputes per-row features once so __getitem__ is pure array slicing."""

    def __init__(self, df: pd.DataFrame, ctx: int = 8):
        self.ctx = ctx
        df = df.dropna(subset=["key_type"]).copy()
        df = df[df["key_type"].isin(["K1", "K2"])].copy()
        df["k"] = (df["key_type"] == "K2").astype(np.int64)
        df.sort_values(["beatmap_id", "user_id", "mods", "ho_t"], inplace=True)
        df = df.reset_index(drop=True)

        n = len(df)
        # per-row feature components (not yet one-hot of prev_key — we fill that on the fly from k)
        k_arr = df["k"].to_numpy(dtype=np.int64)
        bpm = df["bpm"].fillna(120).to_numpy(dtype=np.float32) / 300.0
        prev_gap = df["prev_gap_ms"].fillna(0).to_numpy(dtype=np.float32) / 1000.0
        new_combo = df["ho_new_combo"].fillna(False).astype(np.float32)
        if "effective_bpm" in df.columns:
            eff = df["effective_bpm"].fillna(df["bpm"].fillna(120)).to_numpy(dtype=np.float32)
        else:
            sm = df["speed_mult"].fillna(1.0).to_numpy(dtype=np.float32) if "speed_mult" in df.columns else np.ones(n, dtype=np.float32)
            eff = df["bpm"].fillna(120).to_numpy(dtype=np.float32) * sm
        eff /= 400.0

        self.k_arr = k_arr
        self.bpm = bpm
        self.prev_gap = prev_gap
        self.new_combo = new_combo
        self.eff = eff

        # Group boundaries
        sizes = df.groupby(["beatmap_id", "user_id", "mods"], sort=False).size().to_numpy()
        self.group_start = np.empty(len(sizes), dtype=np.int64)
        self.group_end = np.empty(len(sizes), dtype=np.int64)
        s = 0
        for i, sz in enumerate(sizes):
            self.group_start[i] = s
            self.group_end[i] = s + sz
            s += sz

        # Row index → (group, local), skipping local==0 (no history)
        self.row_group_arr = []
        self.row_local_arr = []
        for gi in range(len(sizes)):
            a, b = self.group_start[gi], self.group_end[gi]
            g_len = b - a
            if g_len <= 1:
                continue
            self.row_group_arr.append(np.full(g_len - 1, gi, dtype=np.int64))
            self.row_local_arr.append(np.arange(1, g_len, dtype=np.int64))
        self.row_group_arr = np.concatenate(self.row_group_arr) if self.row_group_arr else np.zeros(0, dtype=np.int64)
        self.row_local_arr = np.concatenate(self.row_local_arr) if self.row_local_arr else np.zeros(0, dtype=np.int64)

    def __len__(self): return len(self.row_group_arr)

    def __getitem__(self, idx: int):
        gi = self.row_group_arr[idx]
        local = self.row_local_arr[idx]
        gs = self.group_start[gi]
        center = gs + local

        feats = np.zeros((self.ctx, 6), dtype=np.float32)
        # Window of ctx prev rows ending at (center-1); map j in [0..ctx-1]
        # to k = local - (ctx - j). Only fill valid (k >= 0) slots.
        first_valid_j = max(0, self.ctx - local)  # smallest j where k >= 0
        src_lo = gs + (local - (self.ctx - first_valid_j))
        src_hi = center
        if src_hi > src_lo:
            w = src_hi - src_lo
            slot = slice(first_valid_j, first_valid_j + w)
            prev_k = self.k_arr[src_lo:src_hi]
            feats[slot, 0] = (prev_k == 0).astype(np.float32)
            feats[slot, 1] = (prev_k == 1).astype(np.float32)
            feats[slot, 2] = self.bpm[src_lo:src_hi]
            feats[slot, 3] = self.prev_gap[src_lo:src_hi]
            feats[slot, 4] = self.new_combo[src_lo:src_hi]
            feats[slot, 5] = self.eff[src_lo:src_hi]

        target = int(self.k_arr[center])
        return torch.from_numpy(feats), torch.tensor(target, dtype=torch.long)


def train(epochs: int = 4, batch_size: int = 512, lr: float = 1e-3, device: str = "cpu"):
    df = load_all_shards()
    if df.empty:
        print("[train_key] no data"); return
    print(f"[train_key] rows={len(df)}")
    rng = np.random.default_rng(1)
    bmaps = df["beatmap_id"].unique(); rng.shuffle(bmaps)
    val_bmaps = set(bmaps[:max(1, len(bmaps)//10)].tolist())
    train_df = df[~df["beatmap_id"].isin(val_bmaps)]
    val_df = df[df["beatmap_id"].isin(val_bmaps)]
    cfg = KeyConfig()
    tds = KeyDataset(train_df, ctx=cfg.ctx)
    vds = KeyDataset(val_df, ctx=cfg.ctx)
    print(f"[train_key] train={len(tds)} val={len(vds)}")
    if len(tds) == 0 or len(vds) == 0:
        print("[train_key] not enough data"); return
    pin = device.startswith("cuda")
    nw = 4 if pin else 0
    tdl = DataLoader(tds, batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=pin, persistent_workers=(nw > 0))
    vdl = DataLoader(vds, batch_size=batch_size, num_workers=nw, pin_memory=pin, persistent_workers=(nw > 0))
    model = KeyModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best = float("inf")
    for ep in range(epochs):
        t0 = time.time()
        model.train()
        tot=0; n=0
        for x, y in tdl:
            x, y = x.to(device, non_blocking=pin), y.to(device, non_blocking=pin)
            logits = model(x)
            loss = model.nll(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item()*len(y); n += len(y)
        model.eval(); vt=0; vn=0; cor=0
        with torch.no_grad():
            for x, y in vdl:
                x, y = x.to(device, non_blocking=pin), y.to(device, non_blocking=pin)
                logits = model(x)
                loss = model.nll(logits, y)
                vt += loss.item()*len(y); vn += len(y)
                cor += (logits.argmax(-1) == y).sum().item()
        print(f"[key ep {ep}] train={tot/max(1,n):.3f} val={vt/max(1,vn):.3f} val_acc={cor/max(1,vn):.3f} ({time.time()-t0:.1f}s)")
        if vt/max(1,vn) < best:
            best = vt/max(1,vn)
            torch.save({"model": model.state_dict(), "cfg": asdict(cfg)}, CKPT_DIR / "key_best.pt")
    torch.save({"model": model.state_dict(), "cfg": asdict(cfg)}, CKPT_DIR / "key_final.pt")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    print(f"[train_key] device={args.device}")
    train(epochs=args.epochs, batch_size=args.batch, lr=args.lr, device=args.device)
