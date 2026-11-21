"""Smoke tests for the data pipeline.

Run with: python -m pytest tests -q (if pytest installed) or just plain:
    python tests/test_pipeline.py
"""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.data.parse import parse_replay_text, extract_key_events
from src.data.beatmap import parse_osu


def test_replay_parser_handles_negative_seek_frame():
    # Mini synthetic frame text that has the init pattern plus one real click.
    text = "0|256|-500|0,9146|256|-500|0,-9124|100|100|1,10|100|100|1,20|100|100|0,-12345|0|0|42"
    pr = parse_replay_text(text)
    # Parser discards the two setup frames (0 and 9146); gameplay starts at the
    # first negative-dt frame (at absolute time 22).
    assert len(pr.frames) == 3, f"expected 3, got {len(pr.frames)}"
    assert pr.frames[0].t == 22, f"expected 22, got {pr.frames[0].t}"
    assert pr.frames[1].t == 32
    assert pr.frames[2].t == 52
    assert pr.rng_seed == 42


def test_key_event_extraction():
    text = "0|100|100|0,10|100|100|5,10|100|100|5,10|100|100|0,10|100|100|10,10|100|100|0"
    pr = parse_replay_text(text)
    kev = extract_key_events(pr)
    types = [k.key_type for k in kev]
    assert "K1" in types, types
    assert "K2" in types, types


def test_osu_parser_on_sample():
    sample = pathlib.Path(__file__).resolve().parents[1] / "data" / "raw" / "beatmaps"
    files = list(sample.glob("*.osu"))
    if not files:
        print("no .osu files to test")
        return
    bm = parse_osu(files[0])
    assert bm.ar >= 0
    assert bm.od >= 0
    assert bm.cs >= 0
    assert len(bm.hit_objects) > 0
    hw = bm.hit_windows_ms()
    assert all(h > 0 for h in hw), hw


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            print(f"running {name}...", end=" ")
            fn()
            print("OK")
