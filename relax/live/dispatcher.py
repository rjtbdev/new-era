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
from typing import Callable, List, Optional, Tuple

from .model_service import PressDecision
from .sink import KeySink


@dataclass
class _PendingRelease:
    release_t_ms: float
    key: str


# A "ho getter" callable returns the (x, y, radius_px) of a planned HO by
# center_idx, or None if unknown. Dispatcher uses this with cursor_fn to
# gate presses on "cursor is on circle".
HOGetter = Callable[[int], Optional[Tuple[float, float, float]]]
CursorFn = Callable[[], Optional[Tuple[float, float]]]


class Dispatcher:
    def __init__(
        self,
        sink: KeySink,
        *,
        default_hold_ms: float = 85.0,
        press_lead_ms: float = 0.0,
        cursor_fn: Optional[CursorFn] = None,
        ho_getter: Optional[HOGetter] = None,
        cursor_gate_window_ms: float = 60.0,
        cursor_radius_mult: float = 1.0,
    ):
        """
        press_lead_ms: fire presses N ms earlier than their press_time_ms.

        cursor_fn: () -> (x, y) | None. If given and ho_getter is also given,
            dispatcher gates each press on "cursor near HO center".
            Specifically: when a press is scheduled at press_time_ms, it
            fires early as soon as cursor is within cursor_radius_mult *
            circle_radius of the HO center, but no earlier than
            (press_time_ms - cursor_gate_window_ms). If cursor still isn't
            near by (press_time_ms + cursor_gate_window_ms), fires anyway
            (late-fallback — don't miss just because aim is behind).

        ho_getter: (center_idx) -> (x, y, radius_px) | None.

        cursor_gate_window_ms: half-width of the window in which cursor
            gating is active. Outside this window, fire by schedule.

        cursor_radius_mult: 1.0 means "fire when cursor crosses into circle
            hit radius". 1.5 means a bit outside — more lenient.
        """
        self.sink = sink
        self.default_hold_ms = default_hold_ms
        self.press_lead_ms = press_lead_ms
        self.cursor_fn = cursor_fn
        self.ho_getter = ho_getter
        self.cursor_gate_window_ms = cursor_gate_window_ms
        self.cursor_radius_mult = cursor_radius_mult
        self._pending: List[PressDecision] = []
        self._to_release: List[_PendingRelease] = []
        self._press_deltas: List[float] = []

    def submit(self, decisions: List[PressDecision]) -> None:
        self._pending.extend(decisions)
        self._pending.sort(key=lambda d: d.press_time_ms)

    def tick(self, now_bt_ms: float) -> None:
        # Fire any releases that are due (compensated with the same lead so
        # hold duration stays what the model asked for).
        kept_releases: List[_PendingRelease] = []
        for r in self._to_release:
            if now_bt_ms >= r.release_t_ms - self.press_lead_ms:
                self.sink.release(r.key)
            else:
                kept_releases.append(r)
        self._to_release = kept_releases

        # Fire any presses that are due. Presses are time-sorted.
        #
        # Cursor-gating logic (per-press):
        #   - If cursor_fn + ho_getter available and we're within
        #     [press_time - gate_window, press_time + gate_window]:
        #       - If cursor is within `cursor_radius_mult * radius` of
        #         HO center -> FIRE NOW (even if before press_time)
        #       - Else -> wait (don't fire yet, don't break — maybe next tick)
        #   - If past the late-end of the window -> FIRE ANYWAY (don't miss)
        #   - If no cursor data -> fall back to schedule-only.
        gated = (self.cursor_fn is not None and self.ho_getter is not None)
        cursor_xy = self.cursor_fn() if gated else None

        fired = 0
        skipped = 0  # count of presses we chose NOT to fire this tick
        for d in self._pending:
            schedule_fire_time = d.press_time_ms - self.press_lead_ms
            gate_early = d.press_time_ms - self.cursor_gate_window_ms
            gate_late = d.press_time_ms + self.cursor_gate_window_ms

            if now_bt_ms < gate_early:
                # Too early for any kind of fire. All later presses are
                # later still.
                break

            fire = False

            if gated and cursor_xy is not None and now_bt_ms < gate_late:
                # Try cursor gating within the window.
                ho = self.ho_getter(d.center_idx)
                if ho is not None:
                    hx, hy, radius = ho
                    cx, cy = cursor_xy
                    dx = cx - hx; dy = cy - hy
                    dist_sq = dx * dx + dy * dy
                    r = radius * self.cursor_radius_mult
                    if dist_sq <= r * r:
                        fire = True  # cursor on circle -> fire now
                    elif now_bt_ms >= gate_late:
                        fire = True  # out of time -> fire anyway
                    # else: wait another tick
                elif now_bt_ms >= schedule_fire_time:
                    # No HO data; fall back to schedule.
                    fire = True
            else:
                # Not gated, OR past the late-end of window, OR no cursor.
                # Use schedule.
                if now_bt_ms >= gate_late:
                    # Past late-end: fire regardless (don't miss).
                    fire = True
                elif not gated and now_bt_ms >= schedule_fire_time:
                    # Not gated: fire by schedule.
                    fire = True
                elif gated and cursor_xy is None and now_bt_ms >= schedule_fire_time:
                    # Gated mode but no cursor data: fall back to schedule.
                    fire = True

            if fire:
                delta = now_bt_ms - d.press_time_ms
                self._press_deltas.append(delta)
                if len(self._press_deltas) > 500:
                    self._press_deltas = self._press_deltas[-500:]
                self.sink.press(d.key)
                hold = getattr(d, "hold_ms", self.default_hold_ms)
                # Release is anchored to actual fire time, not press_time,
                # so slider holds remain correct even on early fires.
                effective_press_t = min(d.press_time_ms, now_bt_ms)
                self._to_release.append(_PendingRelease(
                    release_t_ms=effective_press_t + hold,
                    key=d.key,
                ))
                fired += 1
            else:
                # Didn't fire this press. But we may have fired earlier ones
                # in this loop (unlikely with sorted queue, but defensive).
                # If we didn't fire, it means we're waiting for cursor — all
                # LATER presses are later, so we should break; but the ones
                # already fired in this loop should be removed from pending.
                # Easier: break and let next tick handle this one again.
                skipped += 1
                break

        if fired:
            # Consume only the ones we actually fired. Since we break on a
            # skip, fired = how many got fired consecutively from the front.
            self._pending = self._pending[fired:]
