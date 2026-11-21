"""Reader abstraction. Defines the GameState shape and the Reader protocol.

Any concrete reader (mock, hemlock-live, whatever) implements the same
interface so the rest of the pipeline doesn't care where state came from.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol


@dataclass
class HitObject:
    """Minimal hit-object representation the model + rules need."""
    idx: int             # position in the beatmap (0-based)
    t: float             # beatmap time ms
    x: float             # playfield x (0..512)
    y: float             # playfield y (0..384)
    kind: str            # "circle" | "slider" | "spinner"
    new_combo: bool
    end_t: Optional[float] = None  # slider end time, if applicable

    def __repr__(self) -> str:
        return f"HO(idx={self.idx} t={self.t:.0f} {self.kind} nc={int(self.new_combo)})"


@dataclass
class GameState:
    """A single-moment snapshot of what the game is doing.

    All times are in beatmap time (ms). Reader is responsible for converting
    from whatever time base the game exposes (osu! itself uses bt).
    """
    # Core per-frame state
    time_ms: float              # current beatmap time
    cursor_x: float             # cursor position in playfield coords
    cursor_y: float

    # Session-stable fields (same from song-start to song-end)
    mods: int                   # mods bitmask (DT=64, HD=8, HR=16, ...)
    cs: float                   # circle size
    od: float                   # overall difficulty
    hit_objects: List[HitObject] = field(default_factory=list)
    hw300: float = 0.0          # computed from od + mods
    hw100: float = 0.0
    hw50: float = 0.0
    circle_radius_px: float = 0.0
    speed_mult: float = 1.0     # 1.5 for DT, 0.75 for HT, 1.0 otherwise

    # Live state
    is_playing: bool = True
    is_paused: bool = False
    beatmap_id: Optional[int] = None
    player_id: Optional[int] = None  # for style-match lookups


class Reader(Protocol):
    """A Reader produces a GameState on demand.

    Implementations:
      - MockReader: drives state from a pre-loaded .osu + .bin
      - HemlockReader (stub): reads osu! memory via hypercalls

    The Reader is polled; it must be safe to call `read()` from the main
    inference thread at ~1 ms cadence with no blocking I/O. Any slow work
    (file reads, network) happens in `attach()`.
    """
    def attach(self) -> None: ...
    def read(self) -> GameState: ...
    def detach(self) -> None: ...
