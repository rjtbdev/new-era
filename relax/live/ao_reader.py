"""AoReader: reads osu!'s live state via the ao HV.

Shells out to `hook_tool.exe` (sigscan + read) rather than linking against
ao_client.h directly, so Python stays dependency-free. Later: a proper
hypercall binding so each read isn't a process spawn.

For each ASLR-shifted field we:
    1. sigscan the osu! module for an AOB from AQN
    2. read 4 bytes at match_va + recipe.target_offset
    3. interpret those bytes as the field's session VA
Resolved VAs are stable for the rest of the osu! process lifetime; if osu!
is restarted, call detach() then attach() again.

See memory project_aqn_osu_sigs_and_offsets.md for the sig provenance.
"""
from __future__ import annotations

import os
import pathlib
import re
import struct
import subprocess
from dataclasses import dataclass
from typing import List, Optional

from .osu_reader import GameState, HitObject


# --- HitObjectManager (HOM) layout, discovered live 2026-11-21 ---
#
# Walk: running_gamemode -> GameMode_Player -> HOM -> _items[] -> HO[i]
#
# GameMode_Player + 0x44 = HOM object pointer
# HOM + 0x90           = int32 count         (how many HOs on this map)
# HOM + 0xB8           = ptr to _items[]     (.NET T[] backing array)
# _items + 0x04        = int32 capacity      (not count; may be > count)
# _items + 0x08 + i*4  = HitObject pointer i
#
# HitObject layout:
#   +0x10  int32 start_t_ms
#   +0x14  int32 end_t_ms   (equal to start_t for circles; differs for sliders/spinners)
#   +0x18  int32 type       (bit 0=circle, bit 1=slider, bit 2=new_combo, bit 3=spinner)
#   +0x38  float32 x        (osu field 0..512)
#   +0x3C  float32 y        (osu field 0..384)
GM_OFF_HOM        = 0x44
HOM_OFF_COUNT     = 0x90
HOM_OFF_ITEMS     = 0xB8
ARR_OFF_LENGTH    = 0x04
ARR_OFF_DATA      = 0x08
HO_OFF_START_T    = 0x10
HO_OFF_END_T      = 0x14
HO_OFF_TYPE       = 0x18
HO_OFF_X          = 0x38
HO_OFF_Y          = 0x3C

# osu HO type bits (matches .osu-file spec)
HO_TYPE_CIRCLE    = 0x01
HO_TYPE_SLIDER    = 0x02
HO_TYPE_NEWCOMBO  = 0x04
HO_TYPE_SPINNER   = 0x08


def _ho_type_to_kind(type_byte: int) -> str:
    """Decode the raw type byte to our kind string."""
    if type_byte & HO_TYPE_SPINNER:
        return "spinner"
    if type_byte & HO_TYPE_SLIDER:
        return "slider"
    return "circle"


HOOK_TOOL_ENV = "AO_HOOK_TOOL"
DEFAULT_HOOK_TOOL = r"C:\Users\riley\projects\ao\cli\hook_tool.exe"

_MATCH_RE = re.compile(r"^\[\+\] MATCH:\s*0x([0-9a-fA-F]+)\s*$", re.M)
_READ_RE  = re.compile(r"^VA=0x[0-9a-fA-F]+\s+SIZE=\d+\s+HEX=([0-9a-fA-F]+)\s*$", re.M)
# sigscan's `[+] osu!.exe cr3=0x... osu!.exe base=0x... ...` header line
_CR3_RE   = re.compile(r"cr3[=\s]+0x([0-9a-fA-F]+)", re.I)


class _HypercallError(RuntimeError):
    pass


def _hook_tool_path() -> str:
    p = os.environ.get(HOOK_TOOL_ENV, DEFAULT_HOOK_TOOL)
    if not pathlib.Path(p).exists():
        raise _HypercallError(f"hook_tool.exe not found at {p}. "
                              f"Set {HOOK_TOOL_ENV}=... or build cli/.")
    return p


def _run_hook_tool(*args: str, timeout: float = 15.0) -> subprocess.CompletedProcess:
    cmd = [_hook_tool_path(), *args]
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise _HypercallError(f"hook_tool {args[0]} timed out after {timeout}s") from e


@dataclass
class SigRecipe:
    """AQN-style AOB sig + how to extract a field VA from the match.

    `aob` is space-separated hex bytes with '?' wildcards, verbatim from AQN.
    `target_offset` is the byte-offset from the match-start to the 4-byte
    displacement that encodes the field VA.
    """
    name: str
    aob: str
    target_offset: int
    scan_mb: int = 2000  # x86 full user-mode address space (osu! is 32-bit)
    note: str = ""

    def to_pattern_mask(self) -> tuple[str, str]:
        tokens = self.aob.strip().split()
        pat, mask = [], []
        for t in tokens:
            if t in ("?", "??"):
                pat.append("00"); mask.append("?")
            else:
                pat.append(t.lower().zfill(2)); mask.append("x")
        return ("".join(pat), "".join(mask))


# AQN sig recipes — see project_aqn_osu_sigs_and_offsets memory.
SIGS = {
    "time": SigRecipe(
        name="time",
        aob="74 12 a1 ? ? ? ? 2b 05 ? ? ? ? a3",
        target_offset=0x24,
        note="AQN time_hook: field VA at match+0x24",
    ),
    "is_paused": SigRecipe(
        name="is_paused",
        aob="56 53 8b f1 80 3d ? ? ? ? 00 75 05",
        target_offset=6,
        note=("Real pause-check site: `push esi / push ebx / mov esi, ecx / "
              "cmp byte ptr [is_paused], 0 / jne +5`. AQN literal `02 75 0E 80 3D ...` "
              "no longer matches on current osu! build. The shortened `80 3D ...` "
              "sig matched a DIFFERENT, dead `cmp byte ptr` site. Field VA at "
              "match+6 (skip `56 53 8B F1 80 3D`, next 4 bytes = displacement)."),
    ),
    # play_mode is derived: play_mode VA = is_paused VA - 0x21
    "mode": SigRecipe(
        name="mode",
        aob="89 45 f0 a1 ? ? ? ? a3 ? ? ? ? a1",
        target_offset=0x13,
    ),
    "running_gamemode": SigRecipe(
        name="running_gamemode",
        aob="89 45 f0 a1 ? ? ? ? a3 ? ? ? ? a1",
        target_offset=0x19,
    ),
}


class AoReader:
    def __init__(
        self,
        process_name: str = "osu!.exe",
        osu_path: Optional[pathlib.Path | str] = None,
        mods: int = 0,
        beatmap_id: Optional[int] = None,
        player_id: Optional[int] = None,
        use_hom: bool = False,
    ):
        """
        Args:
            osu_path: if given, HOs come from this .osu file (parse once).
            use_hom:  if True, HOs come from osu!'s live HOM on each attach.
                      Requires osu! to be IN a map at attach time. Does NOT
                      require osu_path. Session-level defaults (CS/OD/etc.)
                      still need the .osu for now (not read from memory yet).
        """
        self.process_name = process_name
        self.osu_path = pathlib.Path(osu_path) if osu_path else None
        self.mods = mods
        self.beatmap_id = beatmap_id
        self.player_id = player_id
        self.use_hom = use_hom

        self._cr3: Optional[int] = None
        self._hit_objects: List[HitObject] = []
        self._session_fields: dict = {}
        self._va: dict[str, int] = {}
        # HOM walk cache (set on attach when use_hom=True)
        self._hom_ptr: Optional[int] = None
        self._hom_items_ptr: Optional[int] = None
        self._hom_count: int = 0

    # --- Hypercall wrappers (shell out to hook_tool.exe) ---

    def _hc_check_present(self) -> None:
        """Confirm the HV is alive and the process exists; capture CR3."""
        # Run a sigscan of a tiny no-match pattern — produces the header line
        # "[+] <proc> cr3=0x... <proc> base=0x..." BEFORE scanning, then
        # "[!] not found" (expected). Side-effect: we parse CR3 from the header.
        p = _run_hook_tool(
            "sigscan", self.process_name, self.process_name,
            "90909090", "xxxx", "1",
            timeout=20.0,
        )
        out = (p.stdout or "") + (p.stderr or "")
        if "ao HV not detected" in out:
            raise _HypercallError("ao HV not loaded on this boot")
        if "process not found" in out:
            raise _HypercallError(f"process {self.process_name!r} not running")
        m = _CR3_RE.search(out)
        if not m:
            raise _HypercallError(
                f"could not resolve CR3 for {self.process_name!r}: {out.strip()[:200]}")
        self._cr3 = int(m.group(1), 16)

    def _hc_sigscan(self, module: str, recipe: SigRecipe) -> int:
        pattern, mask = recipe.to_pattern_mask()
        p = _run_hook_tool(
            "sigscan", self.process_name, module, pattern, mask, str(recipe.scan_mb),
            timeout=120.0,
        )
        out = (p.stdout or "")
        m = _MATCH_RE.search(out)
        if not m:
            raise _HypercallError(
                f"sig {recipe.name!r} did not match in {module}: "
                f"rc={p.returncode} out={out.strip()[:200]}")
        return int(m.group(1), 16)

    def _hc_read_bytes(self, va: int, size: int) -> bytes:
        p = _run_hook_tool("read", self.process_name, f"0x{va:X}", str(size))
        out = (p.stdout or "")
        m = _READ_RE.search(out)
        if not m:
            raise _HypercallError(
                f"read va=0x{va:X} size={size} failed: "
                f"rc={p.returncode} out={out.strip()[:200]} err={(p.stderr or '').strip()[:200]}")
        hex_str = m.group(1)
        return bytes.fromhex(hex_str)

    def _resolve_sig(self, module: str, recipe: SigRecipe) -> int:
        match_va = self._hc_sigscan(module, recipe)
        disp_va = match_va + recipe.target_offset
        disp_bytes = self._hc_read_bytes(disp_va, 4)
        return int(struct.unpack("<I", disp_bytes)[0])

    # --- Public reader interface ---

    def attach(self) -> None:
        self._hc_check_present()

        for name, recipe in SIGS.items():
            self._va[name] = self._resolve_sig(self.process_name, recipe)

        # play_mode is at is_paused + 0x21F on current osu!stable build.
        # AQN's init.h recipe is "is_paused - 0x21", but that derivation was
        # anchored to a different is_paused address on an older build. On this
        # build play_mode lives at +0x21F from the real is_paused static — found
        # by diffing a 512-byte window around is_paused across mode switches
        # (std=0, taiko=1, catch=2, mania=3). The sig+offset for is_paused is
        # the reliable anchor; the play_mode offset is tied to the actual
        # struct layout at runtime, not to an AQN constant.
        if "is_paused" in self._va:
            self._va["play_mode"] = self._va["is_paused"] + 0x21F

        if self.osu_path is not None:
            # Lazy import — src/ is on sys.path when main.py adds it.
            from src.data.beatmap import parse_osu
            from src.data.features import _circle_radius_px, _effective_od, _speed_mult
            bm = parse_osu(self.osu_path)
            self._hit_objects = [
                HitObject(
                    idx=i, t=float(h.t), x=float(h.x), y=float(h.y),
                    kind=h.kind, new_combo=bool(h.new_combo),
                    end_t=float(h.end_t) if getattr(h, "end_t", None) is not None else None,
                )
                for i, h in enumerate(bm.hit_objects)
                if h.kind in ("circle", "slider", "spinner")
            ]
            eod = _effective_od(bm.od, self.mods)
            self._session_fields = {
                "cs": float(bm.cs),
                "od": float(bm.od),
                "hw300": 80.0 - 6.0 * eod,
                "hw100": 140.0 - 8.0 * eod,
                "hw50": 200.0 - 10.0 * eod,
                "circle_radius_px": _circle_radius_px(bm.cs, self.mods),
                "speed_mult": _speed_mult(self.mods),
            }

        if self.use_hom:
            # Walk to HOM via running_gamemode. osu! must be IN a map now.
            self._hom_attach()
            hos = self._read_hit_objects_from_hom()
            if self.osu_path is None:
                # No .osu file; use HOM entirely. Session fields fall back
                # to osu default CS/OD values; will be approximate until we
                # also pull those from memory.
                self._hit_objects = hos
                # Defaults: CS=4, OD=8 (common osu!std ranked values).
                # Keep hard-coded for now; TODO sig/offset for live CS/OD.
                cs, od = 4.0, 8.0
                eod = od  # mods=0 assumption; apply speed mods later
                self._session_fields = {
                    "cs": cs, "od": od,
                    "hw300": 80.0 - 6.0 * eod,
                    "hw100": 140.0 - 8.0 * eod,
                    "hw50":  200.0 - 10.0 * eod,
                    "circle_radius_px": 54.4 - 4.48 * cs,  # osu std formula
                    "speed_mult": 1.0,
                }
            else:
                # .osu was also parsed; compare against HOM as a sanity check.
                if len(hos) != len(self._hit_objects):
                    print(f"[AoReader] HOM count ({len(hos)}) != .osu count "
                          f"({len(self._hit_objects)}); using .osu-parsed HOs.")

    def read(self) -> GameState:
        if self._cr3 is None:
            raise _HypercallError("attach() first")
        t = self._read_u32(self._va["time"])
        is_paused = bool(self._read_u8(self._va["is_paused"]))
        play_mode = self._read_u32(self._va["play_mode"])
        # Cursor TBD — AQN captures via hooked mouse_apply_handler, not a
        # static field. Next step: PH script that stashes xy into CFG slots.
        cx, cy = 256.0, 192.0
        return GameState(
            time_ms=float(t), cursor_x=float(cx), cursor_y=float(cy),
            mods=self.mods,
            hit_objects=self._hit_objects,
            is_playing=(play_mode == 0),
            is_paused=is_paused,
            beatmap_id=self.beatmap_id, player_id=self.player_id,
            **self._session_fields,
        )

    def detach(self) -> None:
        self._cr3 = None
        self._va.clear()
        self._hom_ptr = None
        self._hom_items_ptr = None
        self._hom_count = 0

    # --- HitObjectManager (HOM) walk ---

    def _hom_attach(self) -> None:
        """Resolve GameMode_Player -> HOM -> _items. Fails if no live map."""
        if "running_gamemode" not in self._va:
            raise _HypercallError("running_gamemode not resolved; attach() first")
        gm_ptr = self._read_u32(self._va["running_gamemode"])
        if gm_ptr == 0:
            raise _HypercallError(
                "GameMode_Player ptr is NULL — osu! is not in a map. "
                "Start a map first, then call attach() with use_hom=True.")
        hom_ptr = self._read_u32(gm_ptr + GM_OFF_HOM)
        if hom_ptr == 0:
            raise _HypercallError(f"HOM ptr at GM+0x{GM_OFF_HOM:X} is NULL")
        items_ptr = self._read_u32(hom_ptr + HOM_OFF_ITEMS)
        if items_ptr == 0:
            raise _HypercallError(f"HOM._items ptr at HOM+0x{HOM_OFF_ITEMS:X} is NULL")
        count = self._read_u32(hom_ptr + HOM_OFF_COUNT)
        self._hom_ptr = hom_ptr
        self._hom_items_ptr = items_ptr
        self._hom_count = count

    def _read_hit_objects_from_hom(self) -> List[HitObject]:
        """Iterate the HOM backing array, read each HO's fields."""
        if self._hom_items_ptr is None:
            raise _HypercallError("_hom_attach() first")
        out: List[HitObject] = []
        # Bulk-read all HO pointers in one call for speed:
        # items[0..count] = _items + 8, 4 bytes each
        ptrs_bytes = self._hc_read_bytes(
            self._hom_items_ptr + ARR_OFF_DATA, self._hom_count * 4
        )
        for i in range(self._hom_count):
            ho_ptr = int.from_bytes(ptrs_bytes[i*4:(i+1)*4], "little")
            if ho_ptr == 0:
                continue
            # Each HO: read a 64-byte window covering +0x10..+0x3F
            # (start, end, type at +0x10..+0x1B; x,y at +0x38..+0x3F).
            win = self._hc_read_bytes(ho_ptr + HO_OFF_START_T, 0x30)
            start_t  = int.from_bytes(win[0:4], "little")
            end_t    = int.from_bytes(win[4:8], "little")
            type_    = int.from_bytes(win[8:12], "little")
            # +0x38 is 40 bytes past start_t base (we started reading at
            # HO_OFF_START_T=0x10; x is at 0x38 so offset 0x28 in `win`)
            x = struct.unpack("<f", win[0x28:0x2C])[0]
            y = struct.unpack("<f", win[0x2C:0x30])[0]
            kind = _ho_type_to_kind(type_)
            new_combo = bool(type_ & HO_TYPE_NEWCOMBO)
            out.append(HitObject(
                idx=i,
                t=float(start_t),
                x=float(x),
                y=float(y),
                kind=kind,
                new_combo=new_combo,
                end_t=float(end_t) if end_t != start_t else None,
            ))
        return out

    def read_hit_objects(self) -> List[HitObject]:
        """Return the currently-loaded HOs from the live HOM.

        Requires attach() was called with use_hom=True. Re-walks the HOM
        pointers on every call (map-change-safe).
        """
        self._hom_attach()
        self._hit_objects = self._read_hit_objects_from_hom()
        return list(self._hit_objects)

    # --- Typed primitive reads (go through hook_tool read) ---
    def _read_u8(self, va: int) -> int:
        return self._hc_read_bytes(va, 1)[0]

    def _read_u32(self, va: int) -> int:
        return int(struct.unpack("<I", self._hc_read_bytes(va, 4))[0])

    def _read_f32(self, va: int) -> float:
        return float(struct.unpack("<f", self._hc_read_bytes(va, 4))[0])

    def resolved_vas(self) -> dict[str, int]:
        """For diagnostics: the field VAs resolved during attach()."""
        return dict(self._va)
