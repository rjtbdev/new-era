"""Sink abstractions. A sink outputs a key press/release event.

Concrete sinks:
    MockSink     — writes a (time, key, action) row to a log. For testing.
    HemlockSink  — stub. Will hypercall into hemlock to write osu!'s
                   current-frame keyboard-state memory field.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Protocol


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


class HemlockSink:
    """Stub. The hypercall write target isn't resolved yet.

    Target shape (TBD): hemlock_write_gpa(gpa, 1-byte current-k-state, size=1).
    The GPA comes from VaToPa(cr3, va_of_current_keyboard_input_field).

    OR: we hook earlier and overwrite the return of osu!'s "read keyboard"
    call. Different approach — decide once we have the sig.
    """
    def __init__(self):
        raise NotImplementedError(
            "HemlockSink is a stub. Needs the sig for osu!'s current-frame "
            "keyboard input field (or the function that reads it)."
        )

    def press(self, key: str) -> None: ...
    def release(self, key: str) -> None: ...
