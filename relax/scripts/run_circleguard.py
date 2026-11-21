"""Run every Circleguard investigator on three replays and emit a side-by-side table."""
from __future__ import annotations

import os
import pathlib
import sys
from typing import Any, Dict, List

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=pathlib.Path(__file__).resolve().parents[1] / ".env")
except Exception:
    pass

from circleguard import Circleguard, ReplayPath

API_KEY = os.environ.get("OSU_API_V1_KEY")
if not API_KEY:
    print("ERROR: OSU_API_V1_KEY missing from env", file=sys.stderr)
    sys.exit(1)

ROOT = pathlib.Path(r"C:\Users\riley\projects\new-era-relax")
TARGETS = {
    "model_default": ROOT / "data/generated/replays/954692_model_default.osr",
    "model_human":   ROOT / "data/generated/replays/954692_model_human.osr",
    "human":         ROOT / "data/generated/replays/954692_human_wrapped.osr",
}


def run_one(name: str, path: pathlib.Path, cg: Circleguard) -> Dict[str, Any]:
    r = ReplayPath(str(path))
    out: Dict[str, Any] = {"path": str(path), "name": name}
    try:
        cg.load(r)
        out["mods"] = str(r.mods)
        out["username"] = str(r.username)
        out["count_300"] = r.count_300
        out["count_100"] = r.count_100
        out["count_50"]  = r.count_50
        out["count_miss"] = r.count_miss
        out["max_combo"] = r.max_combo
    except Exception as e:
        out["load_error"] = repr(e)
        return out

    # Each investigator wrapped so one failure doesn't kill the whole run.
    for fn_name in ("ur", "frametime", "frametimes"):
        try:
            v = getattr(cg, fn_name)(r)
            if isinstance(v, (list, tuple)):
                out[fn_name] = {"count": len(v), "sample": list(v)[:5]}
            else:
                out[fn_name] = v
        except Exception as e:
            out[fn_name] = f"ERR: {e!r}"

    try:
        snaps = cg.snaps(r)
        out["snaps"] = {"count": len(snaps), "sample": [str(s) for s in list(snaps)[:3]]}
    except Exception as e:
        out["snaps"] = f"ERR: {e!r}"

    # Older circleguard has cg.relax(), newer removed it in favor of 2-replay similarity
    for fn_name in ("relax",):
        try:
            v = getattr(cg, fn_name)(r)
            out[fn_name] = v
        except AttributeError:
            out[fn_name] = "API_MISSING"
        except Exception as e:
            out[fn_name] = f"ERR: {e!r}"

    return out


def main() -> int:
    cg = Circleguard(API_KEY)
    results: Dict[str, Dict[str, Any]] = {}
    for name, path in TARGETS.items():
        if not path.exists():
            print(f"[{name}] MISSING: {path}", file=sys.stderr)
            continue
        print(f"[{name}] analyzing {path.name} ...", flush=True)
        results[name] = run_one(name, path, cg)
        print(f"[{name}] done", flush=True)

    import json
    print("\n=== RAW RESULTS (json) ===")
    print(json.dumps(results, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
