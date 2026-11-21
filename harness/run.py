"""new-era harness — single entry point for all validations.

Usage:
    python harness/run.py <name>              run one named validation
    python harness/run.py list                list every available validation

Output lands in artifacts/<YYYY-MM-DD-HHMMSS>/<name>/.

Mirrors the ao harness; see ao/harness/run.py for the canonical version.
"""
from __future__ import annotations

import argparse
import datetime
import importlib.util
import json
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent
HARNESS_DIR = pathlib.Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
VALIDATIONS_DIR = HARNESS_DIR / "validations"


def _discover() -> dict[str, pathlib.Path]:
    VALIDATIONS_DIR.mkdir(parents=True, exist_ok=True)
    out: dict[str, pathlib.Path] = {}
    for p in sorted(VALIDATIONS_DIR.glob("*.py")):
        if p.name.startswith("_"):
            continue
        out[p.stem] = p
    return out


def _load_module(path: pathlib.Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_one(name: str, mod_path: pathlib.Path) -> int:
    ts = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    out_dir = ARTIFACTS / ts / name
    out_dir.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    mod = _load_module(mod_path)
    if not hasattr(mod, "run"):
        print(f"[harness] {name}: no run() function", file=sys.stderr)
        return 2

    stdout_path = out_dir / "stdout.log"
    stderr_path = out_dir / "stderr.log"
    with open(stdout_path, "w", encoding="utf-8") as fo, \
         open(stderr_path, "w", encoding="utf-8") as fe:
        class Tee:
            def __init__(self, *s): self.s = s
            def write(self, x):
                for st in self.s: st.write(x)
            def flush(self):
                for st in self.s: st.flush()
        orig_o, orig_e = sys.stdout, sys.stderr
        sys.stdout = Tee(orig_o, fo); sys.stderr = Tee(orig_e, fe)
        try:
            result = mod.run()
        except SystemExit as e:
            result = (int(e.code) if isinstance(e.code, int) else 1, f"SystemExit: {e.code}", {})
        except Exception as exc:
            import traceback; traceback.print_exc()
            result = (1, f"exception: {exc!r}", {})
        finally:
            sys.stdout = orig_o; sys.stderr = orig_e

    duration = time.perf_counter() - start
    if isinstance(result, tuple) and len(result) == 3:
        exit_code, summary_line, metrics = result
    else:
        exit_code, summary_line, metrics = (int(result) if isinstance(result, int) else 0), "", {}

    status = "PASS" if exit_code == 0 else "FAIL"
    with open(out_dir / "summary.txt", "w", encoding="utf-8") as f:
        f.write(f"{status}  {name}  ({duration:.2f}s)\n")
        if summary_line: f.write(summary_line.rstrip() + "\n")
        for k, v in metrics.items(): f.write(f"  {k}: {v}\n")
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump({"name": name, "exit_code": exit_code, "duration_s": duration,
                   "python": sys.version.split()[0], "timestamp": ts,
                   "metrics": metrics}, f, indent=2)
    print(f"[{status}] {name} ({duration:.2f}s)  →  artifacts/{ts}/{name}/")
    if summary_line: print(f"       {summary_line}")
    return exit_code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("name", nargs="?")
    args = ap.parse_args()
    v = _discover()
    if not args.name or args.name == "list":
        print("available validations:")
        for slug in sorted(v): print(f"  {slug}")
        return 0
    if args.name not in v:
        print(f"[harness] unknown validation: {args.name}", file=sys.stderr)
        print(f"          available: {sorted(v)}", file=sys.stderr)
        return 2
    return _run_one(args.name, v[args.name])


if __name__ == "__main__":
    sys.exit(main())
