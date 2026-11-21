"""Wrap the Yomi Yori human replay and run Circleguard on both model+human."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import lzma
import os
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from osrparse import GameMode, Key, Mod, Replay, ReplayEventOsu


ROOT = pathlib.Path(r"C:\Users\riley\projects\new-era-relax")
BID = 1218337
USER_ID = 15556397       # yodeston (HD+HR cursor source)
MODS_INT = 24            # HD + HR
MODEL_OSR = ROOT / "data/generated/replays/1218337_model_default.osr"
HUMAN_OSR = ROOT / "data/generated/replays/1218337_human_wrapped.osr"
BIN_PATH = ROOT / f"data/raw/replays/{BID}_{USER_ID}_{MODS_INT}.bin"
BM_PATH = ROOT / f"data/raw/beatmaps/{BID}.osu"


def wrap_human():
    raw = BIN_PATH.read_bytes()
    text = lzma.decompress(raw).decode("ascii", errors="replace")
    events = []
    for part in text.rstrip(",").split(","):
        if not part:
            continue
        pcs = part.split("|")
        if len(pcs) != 4:
            continue
        try:
            w = int(float(pcs[0])); x = float(pcs[1]); y = float(pcs[2]); k = int(pcs[3])
        except ValueError:
            continue
        events.append(ReplayEventOsu(time_delta=w, x=x, y=y, keys=Key(k)))
    beatmap_md5 = hashlib.md5(BM_PATH.read_bytes()).hexdigest()
    rp = Replay(
        mode=GameMode.STD, game_version=20210520, beatmap_hash=beatmap_md5,
        username="yodeston", replay_hash="",
        count_300=2300, count_100=40, count_50=0, count_geki=0, count_katu=0, count_miss=5,
        score=100000000, max_combo=2000, perfect=False,
        mods=Mod(MODS_INT), life_bar_graph=None,
        timestamp=_dt.datetime.utcnow(), replay_data=events, replay_id=0, rng_seed=0,
    )
    HUMAN_OSR.parent.mkdir(parents=True, exist_ok=True)
    rp.write_path(str(HUMAN_OSR))
    print(f"[wrap] wrote {HUMAN_OSR}  frames={len(events)}  md5={beatmap_md5}")
    return beatmap_md5


def run_cg():
    load_dotenv(dotenv_path=ROOT / ".env")
    api_key = os.environ["OSU_API_V1_KEY"]
    from circleguard import Circleguard, ReplayPath

    cg = Circleguard(api_key)
    out = {}
    for label, path in (("model", MODEL_OSR), ("human", HUMAN_OSR)):
        r = ReplayPath(str(path))
        cg.load(r)
        row = {"path": str(path)}
        try: row["ur"] = cg.ur(r)
        except Exception as e: row["ur_err"] = str(e)
        try: row["frametime_median"] = cg.frametime(r)
        except Exception as e: row["frametime_err"] = str(e)
        try:
            fts = cg.frametimes(r)
            if hasattr(fts, "__len__"):
                import statistics
                row["frametimes_n"] = len(fts)
                row["frametimes_median"] = statistics.median(fts) if len(fts) else None
                lt6 = sum(1 for x in fts if x < 6)
                lt13 = sum(1 for x in fts if x < 13)
                row["frametimes_lt6_frac"] = lt6 / len(fts) if fts else 0
                row["frametimes_lt13_frac"] = lt13 / len(fts) if fts else 0
        except Exception as e: row["frametimes_err"] = str(e)
        try:
            snaps = cg.snaps(r)
            row["n_snaps"] = len(snaps) if hasattr(snaps, "__len__") else "?"
        except Exception as e: row["snaps_err"] = str(e)
        try:
            j = cg.judgments(r)
            row["judgments_summary"] = {
                "n": len(j) if hasattr(j, "__len__") else "?",
            }
            # classify
            c300=c100=c50=cmiss=0
            for jd in j:
                t = getattr(jd, "type", None)
                tn = getattr(t, "name", str(t)) if t else ""
                if "300" in tn: c300 += 1
                elif "100" in tn: c100 += 1
                elif "50" in tn: c50 += 1
                elif "MISS" in tn.upper() or "miss" in tn.lower(): cmiss += 1
            row["judgments"] = {"300": c300, "100": c100, "50": c50, "miss": cmiss}
        except Exception as e: row["judgments_err"] = str(e)
        print(f"--- {label} ---")
        for k, v in row.items():
            if "err" in k:
                print(f"  {k}: {v[:200]}")
            else:
                print(f"  {k}: {v}")
        out[label] = row

    (ROOT / "data" / "yomi_cg_results.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[cg] results saved to data/yomi_cg_results.json")


def main():
    wrap_human()
    run_cg()


if __name__ == "__main__":
    main()
