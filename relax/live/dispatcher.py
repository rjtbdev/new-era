"""Dispatcher — holds a queue of PressDecisions and fires each via the Sink
at the right absolute beatmap time.

This runs in the main inference loop. On each tick:
  1. Accept any new decisions from ModelService.plan()
  2. Check the queue: any decision whose press_time_ms is now reached →
     fire via the Sink. Compute a short hold (~60ms default) and fire
     the release after the hold.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from .model_service import PressDecision
from .sink import KeySink


@dataclass
class _PendingRelease:
    release_t_ms: float
    key: str


class Dispatcher:
    def __init__(self, sink: KeySink, *, default_hold_ms: float = 85.0):
        self.sink = sink
        self.default_hold_ms = default_hold_ms
        self._pending: List[PressDecision] = []       # waiting to fire a press
        self._to_release: List[_PendingRelease] = []  # waiting to fire a release

    def submit(self, decisions: List[PressDecision]) -> None:
        self._pending.extend(decisions)
        self._pending.sort(key=lambda d: d.press_time_ms)

    def tick(self, now_bt_ms: float) -> None:
        # Fire any releases that are due
        kept_releases: List[_PendingRelease] = []
        for r in self._to_release:
            if now_bt_ms >= r.release_t_ms:
                self.sink.release(r.key)
            else:
                kept_releases.append(r)
        self._to_release = kept_releases

        # Fire any presses that are due. Presses are time-sorted.
        fired = 0
        for d in self._pending:
            if now_bt_ms >= d.press_time_ms:
                self.sink.press(d.key)
                # slider holds should use d.press_time_ms + slider_duration; for
                # circles use the short default. We default here because we don't
                # carry kind on the decision; model_service could add it.
                self._to_release.append(_PendingRelease(
                    release_t_ms=d.press_time_ms + self.default_hold_ms,
                    key=d.key,
                ))
                fired += 1
            else:
                break
        if fired:
            self._pending = self._pending[fired:]
