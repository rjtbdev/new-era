"""relax-live-mock — end-to-end smoke test of the relax live pipeline.

Runs the relax live/main.py in --mock mode against a known beatmap +
reference cursor. Asserts the expected number of key presses + K1
dominance. Pipeline health check; should complete in ~60s at speed=2x.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
RELAX = ROOT / "relax"
PYTHON = RELAX / ".venv" / "Scripts" / "python.exe"

# Known-good fixture — a map + user + mods we've shipped for.
BID = 1851299
USER = 673214
MODS = 0


def run():
    if not PYTHON.exists():
        print(f"missing relax venv python at {PYTHON}")
        return (1, "no relax venv", {})
    main = RELAX / "live" / "main.py"
    if not main.exists():
        print(f"missing {main}")
        return (1, "no live/main.py", {})

    cmd = [
        str(PYTHON), "-u", str(main), "--mock",
        "--bid", str(BID), "--user", str(USER), "--mods", str(MODS),
        "--speed", "4.0",
    ]
    print("[relax-live-mock] running:", " ".join(cmd))

    # Run with the relax dir as cwd so its relative paths work
    proc = subprocess.run(cmd, cwd=str(RELAX), capture_output=True, text=True,
                          timeout=180)
    print("---- stdout ----")
    print(proc.stdout)
    if proc.stderr:
        print("---- stderr ----")
        print(proc.stderr)

    if proc.returncode != 0:
        return (proc.returncode, f"child exited {proc.returncode}", {})

    # Parse expected summary line: "[live] presses: K1=... K2=... total=N"
    presses_line = next((ln for ln in proc.stdout.splitlines() if "presses: K1=" in ln), None)
    if not presses_line:
        return (1, "no 'presses:' summary line in output", {})
    try:
        after_colon = presses_line.split("presses:", 1)[1]
        parts = {kv.split("=")[0].strip(): int(kv.split("=")[1])
                 for kv in after_colon.strip().split()}
    except Exception as e:
        return (1, f"could not parse presses line: {e}", {})

    k1 = parts.get("K1", 0); k2 = parts.get("K2", 0); total = parts.get("total", 0)
    print(f"[relax-live-mock] K1={k1} K2={k2} total={total}")

    metrics = {"k1": k1, "k2": k2, "total": total}
    # Expect ~112 presses for 1851299 (113 hit objects, 1 spinner).
    if total < 100:
        return (1, f"too few presses: {total}", metrics)
    # Expect K1-dominance from rules.
    if k1 < k2:
        return (1, f"K2 outweighs K1 (K1={k1} K2={k2}) — K1-dominance rule broken", metrics)
    return (0, f"{total} presses, K1={k1} K2={k2} (K1 dominant: OK)", metrics)
