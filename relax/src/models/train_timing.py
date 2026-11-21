"""Train the timing head on the current parquet shards.

Resumable — writes a checkpoint after each epoch. Separates train/val by
(beatmap_id) so we validate on fully held-out maps (prevents leakage of a map's
difficulty-specific timing into val).
"""

from __future__ import annotations

import pathlib
import sys
import time
from dataclasses import asdict
from typing import Dict

import torch  # must precede numpy/pandas on Windows (DLL init order)
from torch.utils.data import DataLoader, Subset

import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from src.models.datasets import (  # noqa: E402
    OBJ_FEAT_DIM, PlayerIndex, TimingDataset, collate, load_all_shards,
)
from src.models.timing import TimingConfig, TimingModel  # noqa: E402


PROJECT = pathlib.Path(__file__).resolve().parents[2]
CKPT_DIR = PROJECT / "models"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


def _train_val_split(df: pd.DataFrame, val_frac: float = 0.1, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    bmaps = df["beatmap_id"].unique()
    rng.shuffle(bmaps)
    n_val = max(1, int(len(bmaps) * val_frac))
    val_bmaps = set(bmaps[:n_val].tolist())
    val = df[df["beatmap_id"].isin(val_bmaps)].reset_index(drop=True)
    train = df[~df["beatmap_id"].isin(val_bmaps)].reset_index(drop=True)
    return train, val


def train(
    *,
    epochs: int = 6,
    batch_size: int = 256,
    lr: float = 3e-4,
    device: str = "cpu",
    max_clicks: int | None = None,
    log_every: int = 50,
    tag: str = "",
    player_dim: int = 64,
    use_phase_a_feats: bool = True,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 4,
) -> Dict:
    df = load_all_shards()
    if df.empty:
        print("[train_timing] no data yet, exiting")
        return {"status": "no_data"}
    print(f"[train_timing] loaded {len(df)} rows from parquet shards; n_replays={df.groupby(['beatmap_id','user_id','mods']).ngroups}")

    # Drop misses (no offset)
    df = df.dropna(subset=["offset_ms"]).reset_index(drop=True)
    if max_clicks:
        df = df.sample(n=min(max_clicks, len(df)), random_state=0).reset_index(drop=True)

    train_df, val_df = _train_val_split(df, val_frac=0.1)
    print(f"[train_timing] train={len(train_df)}  val={len(val_df)}")

    player_index = PlayerIndex.build(df, max_players=4096)
    cfg = TimingConfig(
        d_model=d_model, n_heads=n_heads, n_layers=n_layers,
        ctx_past=8, ctx_future=8, n_mix=4,
        dropout=0.1, max_players=player_index.max_players,
        player_dim=player_dim, obj_feat_dim=OBJ_FEAT_DIM,
    )

    model = TimingModel(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    print(f"[train_timing] param count: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    tds = TimingDataset(train_df, player_index, ctx_past=cfg.ctx_past, ctx_future=cfg.ctx_future)
    vds = TimingDataset(val_df, player_index, ctx_past=cfg.ctx_past, ctx_future=cfg.ctx_future)
    pin = device.startswith("cuda")
    # num_workers=0: datasets are too large to pickle through Windows spawn
    # (precomputed feature arrays per worker > 1 GB). DataLoader runs in the
    # main process; since __getitem__ is now pure numpy indexing (no pandas),
    # this is still fast enough to keep the GPU fed.
    nw = 0
    tdl = DataLoader(tds, batch_size=batch_size, shuffle=True, collate_fn=collate, num_workers=nw, pin_memory=pin)
    vdl = DataLoader(vds, batch_size=batch_size, shuffle=False, collate_fn=collate, num_workers=nw, pin_memory=pin)

    history = []
    best_val = float("inf")
    for ep in range(epochs):
        t0 = time.time()
        model.train()
        tot = 0.0; n = 0
        for i, batch in enumerate(tdl):
            batch = {k: v.to(device, non_blocking=pin) for k, v in batch.items()}
            pred = model(batch["obj_feats"], batch["player_id"], pad_mask=batch["pad_mask"])
            loss = model.nll(pred, batch["target"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * batch["target"].numel()
            n += batch["target"].numel()
            if i % log_every == 0:
                print(f"[ep {ep}] it {i}  loss={loss.item():.3f}")
        tr_loss = tot / max(1, n)

        # val
        model.eval()
        vtot = 0.0; vn = 0
        preds_mu = []; targets = []
        with torch.no_grad():
            for batch in vdl:
                batch = {k: v.to(device, non_blocking=pin) for k, v in batch.items()}
                pred = model(batch["obj_feats"], batch["player_id"], pad_mask=batch["pad_mask"])
                loss = model.nll(pred, batch["target"])
                vtot += loss.item() * batch["target"].numel()
                vn += batch["target"].numel()
                # mixture mean for a point estimate
                pi = torch.softmax(pred["pi_logits"], dim=-1)
                mu = (pi * pred["mu"]).sum(-1)
                preds_mu.append(mu.cpu().numpy())
                targets.append(batch["target"].cpu().numpy())
        vl = vtot / max(1, vn)
        preds_mu = np.concatenate(preds_mu) if preds_mu else np.array([])
        targets = np.concatenate(targets) if targets else np.array([])
        res_std = float(np.std(targets - preds_mu)) if len(targets) else float("nan")
        print(f"[ep {ep}] train_nll={tr_loss:.3f} val_nll={vl:.3f} residual_std_ms={res_std:.2f} ({time.time()-t0:.1f}s)")
        history.append({"epoch": ep, "train_nll": tr_loss, "val_nll": vl, "res_std": res_std})
        if vl < best_val:
            best_val = vl
            ckpt = {
                "model": model.state_dict(),
                "cfg": asdict(cfg),
                "player_index": player_index.user_to_idx,
                "history": history,
            }
            best_name = f"timing_best_{tag}.pt" if tag else "timing_best.pt"
            torch.save(ckpt, CKPT_DIR / best_name)
            print(f"[train_timing] saved best to models/{best_name} (val_nll={vl:.3f})")
    # Save final
    final_name = f"timing_final_{tag}.pt" if tag else "timing_final.pt"
    torch.save({
        "model": model.state_dict(),
        "cfg": asdict(cfg),
        "player_index": player_index.user_to_idx,
        "history": history,
    }, CKPT_DIR / final_name)
    return {"status": "ok", "history": history, "best_val": best_val}


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-clicks", type=int, default=None)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--player-dim", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=128)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--n-layers", type=int, default=4)
    args = ap.parse_args()
    print(f"[train_timing] device={args.device} tag={args.tag!r} player_dim={args.player_dim} "
          f"d_model={args.d_model} n_heads={args.n_heads} n_layers={args.n_layers}")
    train(epochs=args.epochs, batch_size=args.batch, lr=args.lr, device=args.device,
          max_clicks=args.max_clicks, tag=args.tag, player_dim=args.player_dim,
          d_model=args.d_model, n_heads=args.n_heads, n_layers=args.n_layers)
