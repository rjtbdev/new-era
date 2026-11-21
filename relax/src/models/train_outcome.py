"""Train the outcome head on per-click rows."""

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
from src.models.outcome import CLASS_NAMES, OutcomeConfig, OutcomeModel  # noqa: E402

PROJECT = pathlib.Path(__file__).resolve().parents[2]
CKPT_DIR = PROJECT / "models"


class OutcomeDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        df = df.copy()
        df["cls"] = df["outcome"].map({c: i for i, c in enumerate(CLASS_NAMES)})
        df = df.dropna(subset=["cls"]).reset_index(drop=True)
        # Fill missing cursor/offset with sentinel 0s and flag as miss (they'll be miss anyway)
        for c in ("offset_ms", "cursor_dx", "cursor_dy"):
            df[c] = df[c].fillna(0.0)
        self.df = df
        # Phase A: +effective_bpm (bpm × speed_mult)
        if "effective_bpm" in df.columns:
            eff_bpm = df["effective_bpm"].fillna(df["bpm"]).to_numpy()
        else:
            spd = df["speed_mult"].fillna(1.0).to_numpy() if "speed_mult" in df.columns else np.ones(len(df))
            eff_bpm = df["bpm"].to_numpy() * spd
        feats = np.stack([
            df["offset_ms"].to_numpy(),
            df["cursor_dx"].to_numpy(),
            df["cursor_dy"].to_numpy(),
            df["hw300"].to_numpy(),
            df["hw100"].to_numpy(),
            df["hw50"].to_numpy(),
            df["circle_radius_px"].to_numpy(),
            df["bpm"].to_numpy(),
            (df["ho_kind"] == "circle").astype(np.float32).to_numpy(),
            df["prev_gap_ms"].fillna(0).to_numpy(),
            df["next_gap_ms"].fillna(0).to_numpy(),
            eff_bpm,
        ], axis=1).astype(np.float32)
        self.feats = feats
        self.y = df["cls"].to_numpy().astype(np.int64)

    def __len__(self): return len(self.y)
    def __getitem__(self, i): return torch.from_numpy(self.feats[i]), torch.tensor(self.y[i])


def train(epochs: int = 4, batch_size: int = 1024, lr: float = 1e-3, device: str = "cpu"):
    df = load_all_shards()
    if df.empty:
        print("[train_outcome] no data"); return
    print(f"[train_outcome] rows={len(df)}")
    # Split by beatmap_id
    rng = np.random.default_rng(1)
    bmaps = df["beatmap_id"].unique(); rng.shuffle(bmaps)
    val_bmaps = set(bmaps[:max(1, len(bmaps)//10)].tolist())
    train_df = df[~df["beatmap_id"].isin(val_bmaps)]
    val_df = df[df["beatmap_id"].isin(val_bmaps)]
    tds = OutcomeDataset(train_df); vds = OutcomeDataset(val_df)
    pin = device.startswith("cuda")
    nw = 4 if pin else 0
    tdl = DataLoader(tds, batch_size=batch_size, shuffle=True, num_workers=nw, pin_memory=pin, persistent_workers=(nw > 0))
    vdl = DataLoader(vds, batch_size=batch_size, num_workers=nw, pin_memory=pin, persistent_workers=(nw > 0))

    cfg = OutcomeConfig(feat_dim=tds.feats.shape[1])
    model = OutcomeModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best_val = float("inf")
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
        # val
        model.eval(); vt=0; vn=0; cor=0
        with torch.no_grad():
            for x, y in vdl:
                x, y = x.to(device, non_blocking=pin), y.to(device, non_blocking=pin)
                logits = model(x)
                loss = model.nll(logits, y)
                vt += loss.item()*len(y); vn += len(y)
                cor += (logits.argmax(-1) == y).sum().item()
        tr = tot/max(1,n); vl = vt/max(1,vn); acc = cor/max(1,vn)
        print(f"[outcome ep {ep}] train={tr:.3f} val={vl:.3f} val_acc={acc:.3f} ({time.time()-t0:.1f}s)")
        if vl < best_val:
            best_val = vl
            torch.save({"model": model.state_dict(), "cfg": asdict(cfg)}, CKPT_DIR / "outcome_best.pt")
    torch.save({"model": model.state_dict(), "cfg": asdict(cfg)}, CKPT_DIR / "outcome_final.pt")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    print(f"[train_outcome] device={args.device}")
    train(epochs=args.epochs, batch_size=args.batch, lr=args.lr, device=args.device)
