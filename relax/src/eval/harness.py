"""End-to-end eval harness.

Given a DataFrame of human clicks and a DataFrame of generated clicks, produce a
report of detection-metric parity. Also provides `detection_verdict` which gives
a single "suspicious / probably clean" verdict based on the metrics most
commonly used by public tooling.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pandas as pd

from src.eval.metrics import compare, full_report


# Thresholds taken from circleguard / public tooling heuristics.
# These are intentionally conservative: being over any of them is a flag, not a verdict.
DETECTION_THRESHOLDS: Dict[str, float] = {
    "cvUR_low": 50.0,         # sub-50 cvUR for a mid-level player is suspicious
    "cvUR_very_low": 60.0,    # circleguard's "relax" threshold is around 60-70
    "edge_hit_rate_high": 0.65,  # consistently edge-hitting is a relax tell
    "hold_mean_low": 20.0,    # <20ms mean hold time is suspicious
    "miss_rate_zero": 0.001,  # perfect accuracy on a hard map
    "same_key_streak_max_high": 20,  # 20+ consecutive same-key presses is odd
}


def detection_verdict(report: Dict[str, Any]) -> Dict[str, Any]:
    flags: List[str] = []
    if not (report["cvUR"] != report["cvUR"]):  # not NaN
        if report["cvUR"] < DETECTION_THRESHOLDS["cvUR_very_low"]:
            flags.append(f"cvUR={report['cvUR']:.1f} <= very_low")
        elif report["cvUR"] < DETECTION_THRESHOLDS["cvUR_low"]:
            flags.append(f"cvUR={report['cvUR']:.1f} <= low")
    if report["edge_hit_rate@0.85"] >= DETECTION_THRESHOLDS["edge_hit_rate_high"]:
        flags.append(f"edge_hit_rate={report['edge_hit_rate@0.85']:.2f}")
    if report["hold"]["mean"] < DETECTION_THRESHOLDS["hold_mean_low"]:
        flags.append(f"hold_mean={report['hold']['mean']:.1f}")
    if report["outcome"]["miss_rate"] <= DETECTION_THRESHOLDS["miss_rate_zero"] and report["n_clicks"] > 200:
        flags.append("perfect_play")
    if report["same_key_streaks"]["max"] >= DETECTION_THRESHOLDS["same_key_streak_max_high"]:
        flags.append(f"max_same_key_streak={report['same_key_streaks']['max']}")
    return {"flagged": bool(flags), "flags": flags}


def evaluate(human_df: pd.DataFrame, gen_df: pd.DataFrame) -> Dict[str, Any]:
    comp = compare(human_df, gen_df)
    comp["verdict_human"] = detection_verdict(comp["human"])
    comp["verdict_generated"] = detection_verdict(comp["generated"])
    return comp
