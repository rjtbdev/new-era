"""Wrap human replay for bid=713442 then run Circleguard on both model and human."""
from __future__ import annotations
import pathlib, sys, json, os, lzma, datetime as _dt
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from osrparse import GameMode, Key, Mod, Replay, ReplayEventOsu

PROJECT = pathlib.Path(__file__).resolve().parents[1]
META = json.loads((PROJECT / "data" / "generated" / "replays" / "713442_meta.json").read_text())
BID = META["bid"]
USER_ID = META["chosen_user"]
MODS = META["chosen_mods"]
MD5 = META["beatmap_md5"]
MODEL_OSR = pathlib.Path(META["model_osr"])
HUMAN_BIN = pathlib.Path(META["human_replay_bin"])
HUMAN_OSR = PROJECT / "data" / "generated" / "replays" / f"{BID}_human_wrapped.osr"


def wrap_human():
    print(f"[wrap] wrapping {HUMAN_BIN}")
    raw = HUMAN_BIN.read_bytes()
    text = lzma.decompress(raw).decode("ascii", errors="replace")
    events = []
    for part in text.rstrip(",").split(","):
        if not part:
            continue
        pcs = part.split("|")
        if len(pcs) != 4:
            continue
        try:
            w = int(pcs[0]); x = float(pcs[1]); y = float(pcs[2]); k = int(pcs[3])
        except ValueError:
            try:
                w = int(float(pcs[0])); x = float(pcs[1]); y = float(pcs[2]); k = int(pcs[3])
            except ValueError:
                continue
        events.append(ReplayEventOsu(time_delta=w, x=x, y=y, keys=Key(k)))

    rp = Replay(
        mode=GameMode.STD,
        game_version=20210520,
        beatmap_hash=MD5,
        username=f"human_{USER_ID}",
        replay_hash="",
        count_300=0, count_100=0, count_50=0, count_geki=0, count_katu=0, count_miss=0,
        score=0, max_combo=0, perfect=False,
        mods=Mod(MODS), life_bar_graph=None,
        timestamp=_dt.datetime.utcnow(),
        replay_data=events, replay_id=0, rng_seed=0,
    )
    rp.write_path(str(HUMAN_OSR))
    print(f"[wrap] wrote {HUMAN_OSR}  frames={len(events)}  mods={int(rp.mods)}")


def run_cg(path: pathlib.Path, label: str) -> dict:
    from circleguard import Circleguard, ReplayPath
    cg = Circleguard(os.environ["OSU_API_V1_KEY"])
    r = ReplayPath(str(path))
    cg.load(r)
    out = {"label": label, "path": str(path)}
    try: out["ur"] = cg.ur(r)
    except Exception as e: out["ur_err"] = str(e)
    try:
        ft = cg.frametime(r)
        out["frametime_median"] = ft
    except Exception as e: out["frametime_err"] = str(e)
    try:
        fts = cg.frametimes(r)
        import statistics
        out["frametime_mean"] = statistics.mean(fts)
        out["frametime_count"] = len(fts)
        out["sub6_frac"] = sum(1 for x in fts if x < 6) / len(fts) if fts else 0
        out["sub13_frac"] = sum(1 for x in fts if x < 13) / len(fts) if fts else 0
    except Exception as e: out["frametimes_err"] = str(e)
    try:
        sn = cg.snaps(r)
        out["snap_count"] = len(sn) if sn is not None else 0
    except Exception as e: out["snaps_err"] = str(e)
    try:
        j = cg.judgments(r)
        c300 = sum(1 for x in j if hasattr(x,'type') and str(x.type).endswith('HIT_300'))
        c100 = sum(1 for x in j if hasattr(x,'type') and str(x.type).endswith('HIT_100'))
        c50  = sum(1 for x in j if hasattr(x,'type') and str(x.type).endswith('HIT_50'))
        cmiss = sum(1 for x in j if hasattr(x,'type') and str(x.type).endswith('MISS'))
        out["j_300"] = c300; out["j_100"] = c100; out["j_50"] = c50; out["j_miss"] = cmiss
        out["j_total"] = len(j)
    except Exception as e: out["judgments_err"] = str(e)
    return out


def main():
    wrap_human()

    from dotenv import load_dotenv
    load_dotenv(dotenv_path=PROJECT / ".env")

    print(f"\n[cg] running Circleguard on MODEL: {MODEL_OSR}")
    m = run_cg(MODEL_OSR, "model")
    print(json.dumps(m, indent=2, default=str))

    print(f"\n[cg] running Circleguard on HUMAN: {HUMAN_OSR}")
    h = run_cg(HUMAN_OSR, "human")
    print(json.dumps(h, indent=2, default=str))

    report = {"bid": BID, "mods_int": MODS, "user_id": USER_ID, "model": m, "human": h}
    (PROJECT / "data" / "cg_713442.json").write_text(json.dumps(report, indent=2, default=str))

    md = PROJECT / "docs" / "dt_result2.md"
    with md.open("w", encoding="utf-8") as f:
        f.write(f"# DT confirmation — bmap {BID}\n\n")
        f.write(f"Map: **Reol - No title [deetz' Expert]** (bid={BID}, 5.53★, 200 BPM)\n")
        f.write(f"Mods: HDDT ({MODS}). Human cursor source: user_id={USER_ID}.\n\n")
        f.write(f"## Circleguard side-by-side\n\n")
        f.write(f"| metric | MODEL | HUMAN |\n|---|---:|---:|\n")
        f.write(f"| UR | {m.get('ur','?'):.2f} | {h.get('ur','?'):.2f} |\n" if isinstance(m.get('ur'), (int,float)) and isinstance(h.get('ur'), (int,float)) else f"| UR | {m.get('ur','?')} | {h.get('ur','?')} |\n")
        f.write(f"| frametime median | {m.get('frametime_median','?'):.2f} | {h.get('frametime_median','?'):.2f} |\n" if isinstance(m.get('frametime_median'),(int,float)) and isinstance(h.get('frametime_median'),(int,float)) else "")
        f.write(f"| sub-6ms frac | {m.get('sub6_frac','?'):.3f} | {h.get('sub6_frac','?'):.3f} |\n" if isinstance(m.get('sub6_frac'),(int,float)) and isinstance(h.get('sub6_frac'),(int,float)) else "")
        f.write(f"| snaps | {m.get('snap_count','?')} | {h.get('snap_count','?')} |\n")
        f.write(f"| 300/100/50/miss | {m.get('j_300','?')}/{m.get('j_100','?')}/{m.get('j_50','?')}/{m.get('j_miss','?')} | {h.get('j_300','?')}/{h.get('j_100','?')}/{h.get('j_50','?')}/{h.get('j_miss','?')} |\n")
        f.write(f"\n## Comparison to previous DT test\n\n")
        f.write(f"954692 (jump, 5.6★, 200BPM, HDDT): model UR 78 / human 73, misses 8/0 → clean\n\n")
        f.write(f"1851299 (nomod, 4.83★, 170BPM): model UR 128 / human 84, misses 17/0 → noisier\n\n")
        f.write(f"\n## Files\n- Model: `{MODEL_OSR}`\n- Human (wrapped): `{HUMAN_OSR}`\n")
    print(f"\n[cg] wrote {md}")


if __name__ == "__main__":
    main()
