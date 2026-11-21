"""tools-build — rebuild new-era/tools/* against the migrated ao host/.

Shells out to each tool's build.bat and verifies the exe lands.
Currently covers osu_menu and osu_overlay; add more as tools are added.
"""
from __future__ import annotations

import pathlib
import subprocess
import time

ROOT = pathlib.Path(__file__).resolve().parents[2]
TOOLS_DIR = ROOT / "tools"

TARGETS = [
    ("osu_menu",    TOOLS_DIR / "osu-menu"    / "build.bat", TOOLS_DIR / "osu-menu"    / "osu_menu.exe"),
    ("osu_overlay", TOOLS_DIR / "osu-overlay" / "build.bat", TOOLS_DIR / "osu-overlay" / "osu_overlay.exe"),
]

MIN_SIZE_BYTES = 20_000


def run():
    results = []
    out_metrics = {}
    failed = []

    for name, bat, exe in TARGETS:
        if not bat.exists():
            failed.append(f"{name}: missing {bat}")
            continue
        print(f"[+] building {name} via {bat.relative_to(ROOT)}")
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                ["cmd.exe", "/c", str(bat)],
                cwd=str(ROOT),
                capture_output=True,
                text=True,
                timeout=300,
            )
        except subprocess.TimeoutExpired:
            failed.append(f"{name}: timed out")
            continue
        dur = time.perf_counter() - t0

        for line in (proc.stdout or "").splitlines()[-15:]:
            print(f"    {line}")
        if proc.returncode != 0:
            for line in (proc.stderr or "").splitlines()[-10:]:
                print(f"    stderr: {line}")
            failed.append(f"{name}: build.bat returned {proc.returncode}")
            continue

        if not exe.exists():
            failed.append(f"{name}: exe missing after build")
            continue
        size = exe.stat().st_size
        if size < MIN_SIZE_BYTES:
            failed.append(f"{name}: only {size} bytes")
            continue

        out_metrics[f"{name}_size"] = size
        out_metrics[f"{name}_dur_s"] = round(dur, 2)
        results.append(f"{name} {size}B/{dur:.1f}s")
        print(f"    OK {size} bytes, {dur:.1f}s")

    if failed:
        for f in failed:
            print(f"FAIL  {f}")
        return (1,
                f"{len(failed)}/{len(TARGETS)} tools failed: {', '.join(failed)}",
                out_metrics)

    summary = "; ".join(results)
    print(f"OK — {summary}")
    return (0, summary, out_metrics)
