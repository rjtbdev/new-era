"""Run Circleguard on the fixed model_default replay and print judgment counts."""
import os, pathlib, sys
from dotenv import load_dotenv

ROOT = pathlib.Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

from circleguard import Circleguard, ReplayPath

cg = Circleguard(os.environ["OSU_API_V1_KEY"])

for tag in ("default", "human"):
    r = ReplayPath(str(ROOT / f"data/generated/replays/954692_model_{tag}.osr"))
    cg.load(r)
    j = cg.judgments(r)
    h300 = sum(1 for x in j if str(x.type).endswith("Hit300"))
    h100 = sum(1 for x in j if str(x.type).endswith("Hit100"))
    h50  = sum(1 for x in j if str(x.type).endswith("Hit50"))
    hm   = sum(1 for x in j if str(x.type).endswith("Miss"))
    print(f"=== 954692 model_{tag} ===")
    print(f"ur={cg.ur(r):.2f}")
    print(f"frametime median={cg.frametime(r)}")
    print(f"snaps n={len(cg.snaps(r))}")
    print(f"judgments: 300={h300} 100={h100} 50={h50} miss={hm}")
    print()
