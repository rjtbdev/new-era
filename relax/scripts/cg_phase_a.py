"""Run Circleguard on each Phase A-generated replay and the corresponding
human reference, print a side-by-side comparison, write JSON."""
import os, json, pathlib, re, sys
from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

from circleguard import Circleguard, ReplayPath
from src.data.emit import write_osr
from src.data.parse import parse_replay_bytes

sys.path.insert(0, str(ROOT))


MODEL_DIR = ROOT / "data" / "generated" / "replays"
REPLAY_DIR = ROOT / "data" / "raw" / "replays"
BM_DIR = ROOT / "data" / "raw" / "beatmaps"

CASES = [
    ("954692_model_phase_a.osr",  954692),
    ("713442_model_phase_a.osr",  713442),
    ("1851299_model_phase_a.osr", 1851299),
]


def find_cursor_src(bid):
    cands = sorted(REPLAY_DIR.glob(f"{bid}_*.bin"))
    return cands[0] if cands else None


def judgments_breakdown(j):
    h300 = sum(1 for x in j if str(x.type).endswith("Hit300"))
    h100 = sum(1 for x in j if str(x.type).endswith("Hit100"))
    h50  = sum(1 for x in j if str(x.type).endswith("Hit50"))
    hm   = sum(1 for x in j if str(x.type).endswith("Miss"))
    return {"300": h300, "100": h100, "50": h50, "miss": hm}


def wrap_human_osr(bid, src_bin, out_path):
    """Build a plain .osr from the cached binary replay data.
    The raw .bin is just an LZMA-compressed replay_data; we need a full .osr
    with header for Circleguard to load."""
    import re, osrparse
    m = re.match(rf"{bid}_(\d+)_(\d+)\.bin", src_bin.name)
    uid = int(m.group(1)); mods = int(m.group(2))
    # Parse frames from our internal format
    parsed = parse_replay_bytes(src_bin.read_bytes())
    cursor_frames = [{"t": f.t, "x": f.x, "y": f.y} for f in parsed.frames]
    # Rebuild clicks + judgments by scanning frames.k for press edges
    clicks = []
    prev_k = 0
    for f in parsed.frames:
        for bit in (5, 10):  # M1|K1, M2|K2
            was = prev_k & bit
            is_ = f.k & bit
            if is_ and not was:
                clicks.append({"t": f.t, "key": "K1" if bit == 5 else "K2"})
        prev_k = f.k
    # Dummy judgments (CG re-judges anyway)
    judg = {"300": len(clicks), "100": 0, "50": 0, "miss": 0}
    # Need md5: read from beatmap file
    osu_path = BM_DIR / f"{bid}.osu"
    import hashlib
    md5 = hashlib.md5(osu_path.read_bytes()).hexdigest()
    write_osr(
        out_path, beatmap_md5=md5, player_name="human_src",
        mods_int=mods, click_events=clicks, cursor_frames=cursor_frames, judgments=judg,
    )
    return out_path


def main():
    cg = Circleguard(os.environ["OSU_API_V1_KEY"])
    results = []
    for model_name, bid in CASES:
        model_path = MODEL_DIR / model_name
        if not model_path.exists():
            print(f"SKIP: {model_path} missing"); continue
        human_bin = find_cursor_src(bid)
        if not human_bin:
            print(f"SKIP bid={bid}: no human cursor src"); continue
        human_wrapped = MODEL_DIR / f"{bid}_human_phase_a.osr"
        try:
            wrap_human_osr(bid, human_bin, human_wrapped)
        except Exception as e:
            print(f"wrap error bid={bid}: {e}")
            continue

        row = {"bid": bid, "model_file": str(model_path), "human_file": str(human_wrapped)}
        for tag, path in (("model", model_path), ("human", human_wrapped)):
            try:
                r = ReplayPath(str(path))
                cg.load(r)
                ur = float(cg.ur(r))
                ft = float(cg.frametime(r))
                sn = len(cg.snaps(r))
                row[f"{tag}_ur"] = ur
                row[f"{tag}_frametime_med"] = ft
                row[f"{tag}_snaps"] = sn
                row[f"{tag}_judg"] = judgments_breakdown(cg.judgments(r))
            except Exception as e:
                row[f"{tag}_error"] = str(e)
        print("---", bid, "---")
        print(json.dumps(row, indent=2))
        results.append(row)
    out = ROOT / "data" / "cg_phase_a.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
