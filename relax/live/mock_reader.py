"""MockReader: drives GameState from a parsed .osu + a reference cursor .bin.

Simulates real-time playback. Used for offline validation of the whole live
pipeline without needing hemlock or osu! running. The cursor trajectory is
read from a real human replay; the time axis is driven by wall-clock from
when `attach()` is called.
"""
from __future__ import annotations

import pathlib
import time
from typing import List, Optional

import torch  # DLL init order on Windows

from src.data.beatmap import parse_osu
from src.data.features import _circle_radius_px, _effective_od, _speed_mult
from src.data.parse import parse_replay_bytes

from .osu_reader import GameState, HitObject, Reader


class MockReader:
    """Runs a .osu beatmap against a reference cursor at real-time speed.

    Use for testing the inference pipeline end-to-end. Not a Reader for a
    live-play scenario; use HemlockReader there.
    """

    def __init__(
        self,
        osu_path: pathlib.Path | str,
        cursor_bin_path: pathlib.Path | str,
        mods: int = 0,
        beatmap_id: Optional[int] = None,
        player_id: Optional[int] = None,
        start_offset_ms: float = -1000.0,
        speed: float = 1.0,
    ):
        self.osu_path = pathlib.Path(osu_path)
        self.cursor_bin_path = pathlib.Path(cursor_bin_path)
        self.mods = mods
        self.beatmap_id = beatmap_id
        self.player_id = player_id
        self.start_offset_ms = start_offset_ms
        self.speed = speed

        self._bm = None
        self._cursor_frames = None
        self._hit_objects: List[HitObject] = []
        self._session_fields = {}
        self._wall_t0: Optional[float] = None

    def attach(self) -> None:
        bm = parse_osu(self.osu_path)
        self._bm = bm
        self._hit_objects = [
            HitObject(
                idx=i, t=float(h.t), x=float(h.x), y=float(h.y),
                kind=h.kind, new_combo=bool(h.new_combo),
                end_t=float(h.end_t) if getattr(h, "end_t", None) is not None else None,
            )
            for i, h in enumerate(bm.hit_objects) if h.kind in ("circle", "slider", "spinner")
        ]

        parsed = parse_replay_bytes(self.cursor_bin_path.read_bytes())
        self._cursor_frames = sorted(
            ({"t": f.t, "x": f.x, "y": f.y} for f in parsed.frames),
            key=lambda f: f["t"],
        )

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

        self._wall_t0 = time.perf_counter()

    def _current_bt(self) -> float:
        if self._wall_t0 is None:
            return self.start_offset_ms
        wall_ms = (time.perf_counter() - self._wall_t0) * 1000.0
        return self.start_offset_ms + wall_ms * self.speed

    def _cursor_at(self, t: float) -> tuple[float, float]:
        cf = self._cursor_frames
        if not cf:
            return (256.0, 192.0)
        if t <= cf[0]["t"]:
            return float(cf[0]["x"]), float(cf[0]["y"])
        if t >= cf[-1]["t"]:
            return float(cf[-1]["x"]), float(cf[-1]["y"])
        # binary search
        lo, hi = 0, len(cf) - 1
        while lo < hi - 1:
            mid = (lo + hi) // 2
            if cf[mid]["t"] <= t:
                lo = mid
            else:
                hi = mid
        a, b = cf[lo], cf[hi]
        if b["t"] == a["t"]:
            return float(a["x"]), float(a["y"])
        f = (t - a["t"]) / (b["t"] - a["t"])
        return (
            float(a["x"] + f * (b["x"] - a["x"])),
            float(a["y"] + f * (b["y"] - a["y"])),
        )

    def read(self) -> GameState:
        bt = self._current_bt()
        cx, cy = self._cursor_at(bt)
        return GameState(
            time_ms=bt, cursor_x=cx, cursor_y=cy,
            mods=self.mods,
            hit_objects=self._hit_objects,
            is_playing=True, is_paused=False,
            beatmap_id=self.beatmap_id, player_id=self.player_id,
            **self._session_fields,
        )

    def detach(self) -> None:
        self._wall_t0 = None
