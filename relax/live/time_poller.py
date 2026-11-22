"""Low-latency osu! beatmap time, piggybacked on the key_inject hook.

The key_inject hook B (post-call WRITE at kgs_match+5) emits a log entry
every time osu! polls GetKeyboardState (observed ~1kHz). Entries are
`ph_log_entry { cr3, va, a=time_ms, b=0 }`.

Drain the log ring and we get fresh osu time with zero per-read HV
shell-out — much faster than calling AoReader.read() which spawns 3
hook_tool subprocesses per call.

Usage:
    tp = TimePoller()
    tp.start()                 # background thread, drains log ring
    bt = tp.latest_time_ms()   # None if no data yet, else float ms
    tp.stop()
"""
from __future__ import annotations

import ctypes
import subprocess
import threading
import time
import re
from dataclasses import dataclass
from typing import Optional


# We shell out to hook_tool because Python can't issue hypercalls directly
# (no ctypes binding yet). Each drain command is one spawn (~50ms) but
# each spawn returns ALL buffered entries, so the per-time-sample cost is
# tiny when the hook fires at 1kHz.
HOOK_TOOL = r"C:\Users\riley\projects\ao\cli\hook_tool.exe"


class TimePoller:
    """Background thread that keeps latest osu time fresh.

    The key_inject hook-B script emits `ph_log_entry.a = osu_time_ms,
    .b = 0` on every keyboard poll. We drain the ring periodically and
    track the newest `a` value as the current osu time.

    Thread-safe: `latest_time_ms()` and `latest_perf()` are atomic reads
    of a (value, timestamp) tuple protected by a lock.
    """

    def __init__(self, drain_interval_s: float = 0.05):
        self._drain_interval = drain_interval_s
        self._lock = threading.Lock()
        self._latest_bt: Optional[float] = None
        self._latest_perf: Optional[float] = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._drain_count = 0

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="TimePoller")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def latest(self) -> tuple[Optional[float], Optional[float]]:
        """Return (bt_ms, perf_at_reception). None/None if no data yet."""
        with self._lock:
            return (self._latest_bt, self._latest_perf)

    def latest_time_ms(self) -> Optional[float]:
        with self._lock:
            return self._latest_bt

    def drain_count(self) -> int:
        return self._drain_count

    # --- private ---

    # hook_tool `log` output per row (after header):
    #   "  0x<cr3-16hex> 0x<va-16hex> 0x<a-16hex> 0x<b-16hex>"
    _LOG_LINE_RE = re.compile(
        r"^\s*0x([0-9a-fA-F]{16})\s+0x([0-9a-fA-F]{16})\s+0x([0-9a-fA-F]{16})\s+0x([0-9a-fA-F]{16})\s*$",
        re.M,
    )

    def _drain_once(self) -> Optional[float]:
        """Shell to `hook_tool log` and return the latest 'a' value seen."""
        try:
            p = subprocess.run(
                [HOOK_TOOL, "log"],
                capture_output=True, text=True, timeout=2.0,
            )
        except subprocess.TimeoutExpired:
            return None
        if p.returncode != 0:
            return None
        # hook_tool log output: "[N] cr3=0x... va=0x... a=0x... b=0x..."
        # We want the newest entry's `a` (= osu time ms).
        matches = list(self._LOG_LINE_RE.finditer(p.stdout))
        if not matches:
            return None
        last = matches[-1]
        # Groups: 1=cr3, 2=va, 3=a, 4=b. `a` is the osu time (we set it that
        # way in the hook's PH_LOG). Fits in uint32; mask off.
        a_val = int(last.group(3), 16)
        self._drain_count += len(matches)
        return float(a_val & 0xFFFFFFFF)

    def _loop(self) -> None:
        while self._running:
            bt = self._drain_once()
            if bt is not None:
                perf = time.perf_counter()
                with self._lock:
                    self._latest_bt = bt
                    self._latest_perf = perf
            time.sleep(self._drain_interval)
