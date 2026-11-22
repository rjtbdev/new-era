"""Background thread that drains the shared ph_log ring and routes entries
to the correct consumer by VA.

Multiple hooks (key_inject, cursor_capture, etc.) emit into one ring.
Each entry carries the hook's install VA, so we can tell them apart:
    va == key_inject_hook_b_va     -> time entry (a = osu time ms)
    va == cursor_capture_va        -> cursor entry (a = x bits, b = y bits)

Usage:
    hdp = HookDataPoller(
        time_va=0xDEADBEEF,      # kgs_match + 5
        cursor_va=0xCAFEBABE,    # mouse_apply_handler match
        max_valid_bt=66817+5000,
    )
    hdp.drain_startup()          # clear stale pre-reset entries
    hdp.start()
    bt, perf = hdp.latest_time()       # (osu time ms, perf_counter at receipt)
    xy, perf = hdp.latest_cursor()     # ((x, y), perf_counter at receipt)
    hdp.stop()
"""
from __future__ import annotations

import re
import struct
import subprocess
import threading
import time
from typing import Optional, Tuple


HOOK_TOOL = r"C:\Users\riley\projects\ao\cli\hook_tool.exe"


class HookDataPoller:
    _LOG_LINE_RE = re.compile(
        r"^\s*0x([0-9a-fA-F]{16})\s+0x([0-9a-fA-F]{16})\s+0x([0-9a-fA-F]{16})\s+0x([0-9a-fA-F]{16})\s*$",
        re.M,
    )

    def __init__(
        self,
        *,
        time_va: int,
        cursor_va: Optional[int] = None,
        max_valid_bt: float = 10_000_000.0,
        drain_interval_s: float = 0.03,
    ):
        self._time_va = time_va & 0xFFFFFFFF
        self._cursor_va = (cursor_va & 0xFFFFFFFF) if cursor_va else None
        self._max_valid_bt = max_valid_bt
        self._drain_interval = drain_interval_s

        self._lock = threading.Lock()
        self._time_bt: Optional[float] = None
        self._time_perf: Optional[float] = None
        self._cursor_xy: Optional[Tuple[float, float]] = None
        self._cursor_perf: Optional[float] = None

        self._time_count = 0
        self._cursor_count = 0
        self._discarded = 0

        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name="HookDataPoller")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)

    def drain_startup(self) -> int:
        """Drop all entries currently in the ring. Call once right after
        detecting osu clock reset."""
        p = subprocess.run([HOOK_TOOL, "log"], capture_output=True, text=True, timeout=2.0)
        n = len(self._LOG_LINE_RE.findall(p.stdout))
        return n

    # --- accessors ---

    def latest_time(self) -> Tuple[Optional[float], Optional[float]]:
        with self._lock:
            return (self._time_bt, self._time_perf)

    def latest_cursor(self) -> Tuple[Optional[Tuple[float, float]], Optional[float]]:
        with self._lock:
            return (self._cursor_xy, self._cursor_perf)

    def latest_time_ms(self) -> Optional[float]:
        with self._lock:
            return self._time_bt

    def time_count(self) -> int:
        return self._time_count

    def cursor_count(self) -> int:
        return self._cursor_count

    def discarded_count(self) -> int:
        return self._discarded

    # --- private ---

    def _loop(self) -> None:
        while self._running:
            self._drain_once()
            time.sleep(self._drain_interval)

    def _drain_once(self) -> None:
        try:
            p = subprocess.run([HOOK_TOOL, "log"], capture_output=True, text=True, timeout=2.0)
        except subprocess.TimeoutExpired:
            return
        if p.returncode != 0:
            return

        matches = list(self._LOG_LINE_RE.finditer(p.stdout))
        if not matches:
            return

        # Track the newest valid entry per consumer. We walk oldest → newest
        # and overwrite as we go, so after the loop we have the latest of each.
        now = time.perf_counter()
        latest_time_bt: Optional[int] = None
        latest_cursor_bits: Optional[Tuple[int, int]] = None

        for m in matches:
            va = int(m.group(2), 16) & 0xFFFFFFFF
            a = int(m.group(3), 16) & 0xFFFFFFFF
            b = int(m.group(4), 16) & 0xFFFFFFFF
            if va == self._time_va:
                if 0 <= a <= self._max_valid_bt:
                    latest_time_bt = a
                    self._time_count += 1
                else:
                    self._discarded += 1
            elif self._cursor_va is not None and va == self._cursor_va:
                latest_cursor_bits = (a, b)
                self._cursor_count += 1
            else:
                # Entry from some other hook; ignore
                pass

        if latest_time_bt is not None or latest_cursor_bits is not None:
            with self._lock:
                if latest_time_bt is not None:
                    self._time_bt = float(latest_time_bt)
                    self._time_perf = now
                if latest_cursor_bits is not None:
                    xb, yb = latest_cursor_bits
                    x = struct.unpack("<f", xb.to_bytes(4, "little"))[0]
                    y = struct.unpack("<f", yb.to_bytes(4, "little"))[0]
                    self._cursor_xy = (float(x), float(y))
                    self._cursor_perf = now
