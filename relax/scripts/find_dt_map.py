"""Find a popular ranked osu!std map 5-6★ not in training data with a usable DT leaderboard score.

Prints candidate bid; follow-up script uses it.
"""
from __future__ import annotations
import pathlib, sys, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.api.v2 import OsuApiV2

PROJECT = pathlib.Path(__file__).resolve().parents[1]
INDEX = PROJECT / "data" / "raw" / "replay_index.jsonl"
SEARCH = PROJECT / "data" / "raw" / "beatmapsets_search.jsonl"

def main():
    # Load training bids
    train_bids = set()
    for line in INDEX.read_text().splitlines():
        try:
            r = json.loads(line)
            train_bids.add(int(r["beatmap_id"]))
        except Exception:
            pass
    print(f"[find] training bids: {len(train_bids)}")

    # Walk the search cache, collect 5-6★ osu!std ranked diffs not in training
    candidates = []
    for line in SEARCH.read_text().splitlines():
        try:
            page = json.loads(line)
        except Exception:
            continue
        for bs in page.get("beatmapsets") or []:
            if bs.get("status") != "ranked":
                continue
            for d in bs.get("beatmaps") or []:
                if d.get("mode") != "osu":
                    continue
                sr = d.get("difficulty_rating") or 0
                if not (4.8 <= sr <= 6.0):
                    continue
                bid = d.get("id")
                if not bid or int(bid) in train_bids:
                    continue
                candidates.append({
                    "bid": int(bid),
                    "sr": sr,
                    "title": bs.get("title"),
                    "artist": bs.get("artist"),
                    "version": d.get("version"),
                    "bpm": d.get("bpm") or bs.get("bpm"),
                    "playcount": bs.get("play_count"),
                })
    # Sort by playcount desc
    candidates.sort(key=lambda c: c.get("playcount") or 0, reverse=True)
    print(f"[find] {len(candidates)} candidates")
    for c in candidates[:20]:
        print(f"  bid={c['bid']}  sr={c['sr']:.2f}  bpm={c['bpm']}  plays={c['playcount']}  {c['artist']} - {c['title']} [{c['version']}]")

    # Pull leaderboards until we find one with a DT score
    api2 = OsuApiV2()
    for c in candidates[:40]:
        try:
            lb = api2.get_beatmap_scores(c["bid"], limit=50)
        except Exception as e:
            print(f"[find] bid={c['bid']} exception: {e}")
            continue
        scores = (lb or {}).get("scores") or []
        name_to_bit = {"NF":1,"EZ":2,"HD":8,"HR":16,"SD":32,"DT":64,"RX":128,"HT":256,"NC":512,"FL":1024,"SO":4096,"AP":8192,"PF":16384}
        dt_count = 0
        best_dt = None
        for s in scores:
            mods = s.get("mods") or []
            b = 0
            for m in mods:
                b |= name_to_bit.get(m, 0)
            if b & 64:
                dt_count += 1
                if best_dt is None:
                    best_dt = {"user_id": s.get("user_id"), "mods_bits": b, "mods_list": mods, "accuracy": s.get("accuracy"), "username": (s.get("user") or {}).get("username")}
        print(f"[find] bid={c['bid']} n_scores={len(scores)} dt={dt_count} best_dt={best_dt}")
        if dt_count >= 1 and best_dt:
            pick = {**c, **best_dt, "n_scores": len(scores), "dt_count": dt_count}
            out = PROJECT / "data" / "raw" / "dt_pick.json"
            out.write_text(json.dumps(pick, indent=2))
            print(f"[find] PICKED: bid={c['bid']} user={best_dt['user_id']} mods={best_dt['mods_bits']} ({best_dt['mods_list']})")
            print(f"[find] written → {out}")
            return

    print("[find] no candidates with DT leaderboard scores found")
    sys.exit(1)

if __name__ == "__main__":
    main()
