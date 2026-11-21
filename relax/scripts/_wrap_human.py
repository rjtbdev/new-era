"""Wrap a human .bin (raw LZMA replay_data) into a full .osr Circleguard can load."""
from __future__ import annotations

import json
import lzma
import pathlib
import sys
import datetime as _dt
from typing import List

from osrparse import GameMode, Key, Mod, Replay, ReplayEventOsu

ROOT = pathlib.Path(r"C:\Users\riley\projects\new-era-relax")

def wrap(bin_path: pathlib.Path, out_path: pathlib.Path, beatmap_md5: str, username: str,
         mods_bits: int, counts: dict) -> None:
    raw = bin_path.read_bytes()
    text = lzma.decompress(raw).decode("ascii", errors="replace")
    events: List[ReplayEventOsu] = []
    for part in text.rstrip(",").split(","):
        if not part:
            continue
        pcs = part.split("|")
        if len(pcs) != 4:
            continue
        w_s, x_s, y_s, k_s = pcs
        try:
            w = int(w_s); x = float(x_s); y = float(y_s); k = int(k_s)
        except ValueError:
            try:
                w = int(float(w_s)); x = float(x_s); y = float(y_s); k = int(k_s)
            except ValueError:
                continue
        events.append(ReplayEventOsu(time_delta=w, x=x, y=y, keys=Key(k)))

    rp = Replay(
        mode=GameMode.STD,
        game_version=20210520,
        beatmap_hash=beatmap_md5,
        username=username,
        replay_hash="",
        count_300=counts.get("count_300", 0),
        count_100=counts.get("count_100", 0),
        count_50=counts.get("count_50", 0),
        count_geki=counts.get("count_geki", 0),
        count_katu=counts.get("count_katu", 0),
        count_miss=counts.get("count_miss", 0),
        score=counts.get("score", 0),
        max_combo=counts.get("max_combo", 0),
        perfect=counts.get("perfect", False),
        mods=Mod(mods_bits),
        life_bar_graph=None,
        timestamp=_dt.datetime.utcnow(),
        replay_data=events,
        replay_id=0,
        rng_seed=0,
    )
    rp.write_path(str(out_path))
    print(f"wrote {out_path}  frames={len(events)}  mods={int(rp.mods)}")


def main() -> int:
    # Find any leaderboard entry for 954692 to get the real beatmap_md5 + a user's counts
    lbdir = ROOT / "data/raw/leaderboards"
    lb_files = sorted(lbdir.glob("*954692*.json")) + sorted(lbdir.glob("954692*.json"))
    lb = None
    for f in lb_files:
        try:
            lb = json.loads(f.read_text())
            if lb:
                break
        except Exception:
            continue
    if lb is None:
        print("no leaderboard found for 954692", file=sys.stderr)
        return 1

    # Beatmap md5 comes from the beatmap file
    beatmaps_dir = ROOT / "data/raw/beatmaps"
    import hashlib
    bm_file = beatmaps_dir / "954692.osu"
    beatmap_md5 = hashlib.md5(bm_file.read_bytes()).hexdigest() if bm_file.exists() else ""
    print(f"beatmap_md5={beatmap_md5}  n_lb_entries={len(lb)}")

    scores = lb.get("scores", lb) if isinstance(lb, dict) else lb
    MOD_BITS = {"NF":1,"EZ":2,"TD":4,"HD":8,"HR":16,"SD":32,"DT":64,"RX":128,"HT":256,"NC":512,"FL":1024,"AT":2048,"SO":4096,"AP":8192,"PF":16384}
    def mods_to_bits(m):
        if isinstance(m, int): return m
        if isinstance(m, list):
            v = 0
            for x in m:
                v |= MOD_BITS.get(x, 0)
            return v
        return 0

    target = None
    for s in scores:
        if mods_to_bits(s.get("mods", s.get("enabled_mods"))) == 72:
            target = s; break
    if target is None:
        print("no HDDT entry; using first", file=sys.stderr)
        target = scores[0]

    uid = int(target.get("user", {}).get("id", target.get("user_id", 0)))
    mods_bits = mods_to_bits(target.get("mods", target.get("enabled_mods", 0)))
    username = target.get("user", {}).get("username", target.get("username", f"uid_{uid}"))
    st = target.get("statistics", target)
    counts = {
        "count_300": int(st.get("count_300", st.get("count300", 0))),
        "count_100": int(st.get("count_100", st.get("count100", 0))),
        "count_50":  int(st.get("count_50",  st.get("count50", 0))),
        "count_geki": int(st.get("count_geki") or 0),
        "count_katu": int(st.get("count_katu") or 0),
        "count_miss": int(st.get("count_miss", st.get("countmiss", 0))),
        "score": int(target.get("score", 0)),
        "max_combo": int(target.get("max_combo", target.get("maxcombo", 0))),
        "perfect": bool(target.get("perfect", False)),
    }
    print(f"target: user={username} uid={uid} mods={mods_bits}")

    bin_path = ROOT / f"data/raw/replays/954692_{uid}_{mods_bits}.bin"
    if not bin_path.exists():
        for p in (ROOT / "data/raw/replays").glob("954692_*_72.bin"):
            bin_path = p
            break
    out_path = ROOT / "data/generated/replays/954692_human_wrapped.osr"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wrap(bin_path, out_path, beatmap_md5, username, mods_bits, counts)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
