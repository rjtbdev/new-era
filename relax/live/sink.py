"""Sink abstractions. A sink outputs a key press/release event.

Concrete sinks:
    MockSink  — writes a (time, key, action) row to a log. For testing.
    AoSink    — drives osu! input via ao HV. Requires key_inject hook pair
                installed (key_inject.exe install). This sink only flips CFG
                slots on hook B; the hook itself does the WRITE1 inside osu!.
"""
from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Protocol


class KeySink(Protocol):
    def press(self, key: str) -> None: ...
    def release(self, key: str) -> None: ...


@dataclass
class MockSink:
    events: List[tuple] = field(default_factory=list)  # (wall_ms, action, key)
    verbose: bool = False
    t0: float = field(default_factory=time.perf_counter)

    def press(self, key: str) -> None:
        t = (time.perf_counter() - self.t0) * 1000.0
        self.events.append((t, "press", key))
        if self.verbose:
            print(f"[{t:9.2f}ms] PRESS   {key}")

    def release(self, key: str) -> None:
        t = (time.perf_counter() - self.t0) * 1000.0
        self.events.append((t, "release", key))
        if self.verbose:
            print(f"[{t:9.2f}ms] RELEASE {key}")


class AoSink:
    """Drives osu! key presses via the ao HV key_inject hook pair.

    CFG layout on hook B (set by `key_inject install`):
        CFG0 = K1 enable (0 pass-through, nonzero override)
        CFG1 = K1 value  (0x80 pressed, 0x00 released)
        CFG2 = K2 enable
        CFG3 = K2 value

    Shells out to hook_tool.exe to set CFG. For the inner loop this is one
    subprocess per press / release; at ~10-20 presses/sec in osu! that's
    ~20-40 process spawns/sec worst case. Acceptable for a first wire-up.
    If latency becomes a problem, add a ctypes binding to ao_call.obj so
    we issue AO_CMD_PH_SET_CFG directly from Python.
    """

    HOOK_TOOL_DEFAULT = r"C:\Users\riley\projects\ao\cli\hook_tool.exe"
    KEY_INJECT_DEFAULT = r"C:\Users\riley\projects\new-era\tools\key-inject\key_inject.exe"

    def __init__(
        self,
        *,
        hook_tool: Optional[str] = None,
        key_inject: Optional[str] = None,
        verbose: bool = False,
        dry_run: bool = False,
    ):
        self.hook_tool = Path(hook_tool or self.HOOK_TOOL_DEFAULT)
        self.key_inject = Path(key_inject or self.KEY_INJECT_DEFAULT)
        self.verbose = verbose
        self.dry_run = dry_run
        self._slot_b: Optional[int] = None
        self._t0 = time.perf_counter()

    def attach(self) -> None:
        """Locate hook B's slot. Caller must have already run `key_inject
        install` before this. We parse `hook_tool status` to find the slot
        whose VA matches `kgs_match + 5` — but we don't re-resolve the sig
        here, so we fall back to: the second active hook with insn_count
        == 8 (the write hook). If none matches, raise.
        """
        if not self.hook_tool.exists():
            raise FileNotFoundError(f"hook_tool.exe not at {self.hook_tool}")
        if self.dry_run:
            self._slot_b = -1  # sentinel
            print("[AoSink] dry run; not touching hooks")
            return

        out = subprocess.run([str(self.hook_tool), "status"],
                             capture_output=True, text=True, timeout=10)
        # Parse status table:
        #   slot active cr3                va                 hits       insns  fault        fault_pc
        #   1    1      0x...              0x...              1234       8      none         0
        slot_re = re.compile(
            r"^\s*(\d+)\s+(\d)\s+0x[0-9a-fA-F]+\s+0x([0-9a-fA-F]+)\s+\d+\s+(\d+)",
            re.M,
        )
        candidates = []  # (slot, va, insn_count)
        for m in slot_re.finditer(out.stdout):
            slot, active, va_hex, insn_count = m.groups()
            if int(active) != 1:
                continue
            candidates.append((int(slot), int(va_hex, 16), int(insn_count)))

        # Hook B is the write hook; hook A is a 1-insn passthrough.
        # B's size changed over time as we added features (8 -> 11 with
        # time-logging). Find it as "any active hook that isn't the 1-insn
        # passthrough".
        writers = [c for c in candidates if c[2] > 1]
        if not writers:
            raise RuntimeError(
                "AoSink: no multi-insn hook found (hook B). "
                "Run `key_inject install` first.\n"
                f"hook_tool status output:\n{out.stdout}"
            )
        if len(writers) > 1:
            print(f"[AoSink] WARNING: multiple 8-insn hooks: {writers}. Using first.")
        self._slot_b = writers[0][0]
        if self.verbose:
            print(f"[AoSink] bound to hook B at slot {self._slot_b} (va=0x{writers[0][1]:X})")

    def detach(self) -> None:
        """Clear any lingering force overrides. Doesn't uninstall the hooks."""
        if self._slot_b is None or self._slot_b < 0:
            return
        if self.dry_run:
            return
        # Best-effort: zero all four CFG slots. We can't do this via
        # hook_tool directly (no set_cfg subcommand yet); use a tiny
        # ctypes shim or just install a release-both press via key_inject.
        # For now we just rely on the dispatcher firing explicit release()
        # calls for every press. If paranoid, `key_inject press k1 1` then
        # the dispatcher's clear-CFG will zero it below.
        for k in ("K1", "K2"):
            self.release(k)

    # --- KeySink protocol ---

    def _set_cfg(self, cfg_idx: int, value: int) -> None:
        if self.dry_run or self._slot_b is None or self._slot_b < 0:
            return
        # hook_tool doesn't expose set_cfg directly today; use the small
        # ctypes shim below to call ao_ph_set_cfg via the dll-less hypercall.
        _ao_ph_set_cfg(self._slot_b, cfg_idx, value)

    def press(self, key: str) -> None:
        t = (time.perf_counter() - self._t0) * 1000.0
        if self.verbose:
            print(f"[{t:9.2f}ms] PRESS   {key}")
        if key == "K1":
            self._set_cfg(0, 1)      # enable
            self._set_cfg(1, 0x80)   # value: pressed
        elif key == "K2":
            self._set_cfg(2, 1)
            self._set_cfg(3, 0x80)
        else:
            raise ValueError(f"AoSink only knows K1/K2, got {key}")

    def release(self, key: str) -> None:
        t = (time.perf_counter() - self._t0) * 1000.0
        if self.verbose:
            print(f"[{t:9.2f}ms] RELEASE {key}")
        if key == "K1":
            self._set_cfg(0, 0)      # disable override (pass-through)
        elif key == "K2":
            self._set_cfg(2, 0)
        else:
            raise ValueError(f"AoSink only knows K1/K2, got {key}")


# ----- low-level hypercall binding (Python -> ao HV, no process spawn) -----
#
# We use ctypes to load a tiny stub DLL built from host/ao_call.asm +
# a C wrapper. But rather than ship a DLL, we can compile at import time
# or just shell out. For zero-DLL correctness and reasonable speed, we
# use a small MinGW-inline approach: call hook_tool via a dedicated
# `set_cfg` subcommand. That doesn't exist yet in hook_tool — we'll add
# it as a one-liner so AoSink has a direct path.
#
# For now: shell to hook_tool with a new `set_cfg` command. We fall back
# to a helpful error if the command isn't available.

_HOOK_TOOL_PATH = r"C:\Users\riley\projects\ao\cli\hook_tool.exe"


def _ao_ph_set_cfg(slot: int, cfg_idx: int, value: int) -> None:
    """Shell to hook_tool set_cfg. Command added in ao this session."""
    r = subprocess.run(
        [_HOOK_TOOL_PATH, "set_cfg", str(slot), str(cfg_idx), f"0x{value:X}"],
        capture_output=True, text=True, timeout=5,
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"ao_ph_set_cfg(slot={slot}, cfg={cfg_idx}, val=0x{value:X}) failed: "
            f"rc={r.returncode} stdout={r.stdout!r} stderr={r.stderr!r}"
        )
